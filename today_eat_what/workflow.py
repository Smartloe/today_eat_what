import json
import logging
from datetime import datetime
from typing import Any, Dict, List, TypedDict

import requests
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from .clients import CostTracker, ModelClient
from .config import HOWTOCOOK_MCP_URL, MODEL_CONFIG, XHS_MCP_URL, load_api_keys
from .models import AuditResult, PublishResult, Recipe, RecipeStep
from .utils import load_dotenv, run_with_retry, setup_logging

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict, total=False):
    current_time: str
    meal_type: str
    recipe_data: Dict[str, Any]
    content: str
    audit_result: bool
    images: List[str]
    publish_result: Dict[str, Any]
    cost: float


# ---------- Core domain helpers ----------


def get_meal_type(now: datetime | None = None) -> str:
    now = now or datetime.now()
    hour = now.hour
    if 6 <= hour <= 10:
        return "早餐"
    if 11 <= hour <= 14:
        return "午餐"
    if 17 <= hour <= 21:
        return "晚餐"
    return "小吃"


def call_how_to_cook(meal_type: str) -> Dict[str, Any]:
    payload = {"meal_type": meal_type}
    if not HOWTOCOOK_MCP_URL:
        # Fallback stub.
        return {
            "recipe": {
                "name": f"{meal_type}活力套餐",
                "description": "简单易做的家常菜，快速补充能量。",
                "ingredients": ["鸡胸肉 150g", "西兰花 1颗", "米饭 1碗", "橄榄油 1勺", "盐、黑胡椒 适量"],
                "steps": [
                    {"order": 1, "instruction": "鸡胸肉切片，撒盐和黑胡椒腌5分钟。"},
                    {"order": 2, "instruction": "西兰花切小朵焯水，备用。"},
                    {"order": 3, "instruction": "热锅倒油，煎熟鸡胸肉，加入西兰花翻炒。"},
                    {"order": 4, "instruction": "盛出搭配米饭，淋少许橄榄油。"},
                ],
            }
        }
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(HOWTOCOOK_MCP_URL, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # pragma: no cover - external service
        logger.error("HowToCook MCP failed: %s", exc)
        raise


def fetch_recipe(meal_type: str, qwen: ModelClient, cost: CostTracker) -> Recipe:
    # Ask Qwen to pick a recipe and enrich with HowToCook data.
    messages = [
        {
            "role": "system",
            "content": "你是美食助理，专注家常菜谱。请输出 JSON 对象 {\"recipe\": {...}}，"
            "包含 name/description/ingredients(list)/steps(list: {order, instruction})。",
        },
        {"role": "user", "content": f"餐次：{meal_type}，请给出适合这一餐次的1道菜。"},
    ]
    cost.add("qwen")
    llm_resp = qwen.invoke_chat(messages, extra={"temperature": 0.4})
    recipe_data = llm_resp.get("recipe") if isinstance(llm_resp, dict) else None

    if not recipe_data and isinstance(llm_resp, dict):
        content = (
            llm_resp.get("content")
            or llm_resp.get("text")
            or llm_resp.get("output")
            or (llm_resp.get("choices") or [{}])[0].get("message", {}).get("content")
        )
        if content:
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    recipe_data = parsed.get("recipe") or parsed
            except json.JSONDecodeError:
                logger.warning("Qwen 返回非JSON内容，回退到 MCP：%s", content[:50])

    if not recipe_data:
        recipe_data = call_how_to_cook(meal_type).get("recipe", {})

    # Merge fields and build model.
    name = recipe_data.get("name") or f"{meal_type}推荐"
    description = recipe_data.get("description") or "轻松上手的美味搭配"
    ingredients = recipe_data.get("ingredients") or ["根据口味准备常用食材"]
    steps_raw = recipe_data.get("steps") or [{"order": 1, "instruction": "按常规方法烹饪至熟。"}]
    steps = [RecipeStep(order=s.get("order", i + 1), instruction=s.get("instruction", "")) for i, s in enumerate(steps_raw)]
    return Recipe(name=name, description=description, ingredients=ingredients, steps=steps, meal_type=meal_type)


def generate_content(recipe: Recipe, deepseek: ModelClient, cost: CostTracker) -> str:
    title_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是小红书美食创作者，写20字内的吸睛标题，带1个表情。"),
            ("human", "菜名：{name}，餐次：{meal_type}"),
        ]
    )
    body_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "生成小红书风格文案，包含食材、步骤亮点、口味描述，配2-3个话题标签，使用表情符号。控制在180字以内。",
            ),
            ("human", "菜谱：{description}；主要食材：{ingredients}；步骤：{steps}"),
        ]
    )

    cost.add("deepseek")
    title_resp = deepseek.invoke(title_prompt.format(name=recipe.name, meal_type=recipe.meal_type))
    body_resp = deepseek.invoke(
        body_prompt.format(
            description=recipe.description,
            ingredients=", ".join(recipe.ingredients),
            steps=" / ".join([s.instruction for s in recipe.steps]),
        )
    )
    title = title_resp.get("text") or title_resp.get("output") or "美味上线"
    body = body_resp.get("text") or body_resp.get("output") or ""
    return f"{title}\n{body}"


def content_audit(content: str, longcat: ModelClient, cost: CostTracker) -> AuditResult:
    audit_prompt = "审查以下内容是否包含敏感或违规信息，只返回JSON: {\"ok\": true/false, \"reasons\": []}\n" + content
    cost.add("longcat")
    resp = longcat.invoke(audit_prompt)
    ok = bool(resp.get("ok", True))
    reasons = resp.get("reasons") if isinstance(resp, dict) else None
    return AuditResult(ok=ok, reasons=reasons)


def generate_images(recipe: Recipe, doubao: ModelClient, cost: CostTracker) -> List[str]:
    prompt = (
        f"生成符合小红书风格的菜品图片：{recipe.name}，描述：{recipe.description}。"
        "再生成按步骤的制作过程插画，每步1张。输出JSON {'urls': [list]}"
    )
    cost.add("doubao")
    resp = doubao.invoke(prompt)
    urls = resp.get("urls") if isinstance(resp, dict) else None
    if not urls:
        urls = [f"https://imgs.local/{recipe.name}_cover.png"] + [
            f"https://imgs.local/{recipe.name}_step_{step.order}.png" for step in recipe.steps
        ]
    return urls


def publish_to_xiaohongshu(content: str, images: List[str], gpt4: ModelClient, cost: CostTracker) -> PublishResult:
    publish_prompt = (
        "你是小红书助手，准备发布以下内容，返回JSON {\"success\": true/false, \"post_id\": \"...\"}。"
        f"正文：{content}\n图片：{images}"
    )
    cost.add("gpt4")
    resp = gpt4.invoke(publish_prompt)
    success = bool(resp.get("success", True))
    post_id = resp.get("post_id") or resp.get("id") or "mock-post-id"

    # Optionally call MCP endpoint if configured.
    if XHS_MCP_URL:
        try:
            mcp_resp = requests.post(XHS_MCP_URL, json={"content": content, "images": images}, timeout=10)
            mcp_resp.raise_for_status()
            publish_data = mcp_resp.json()
            success = bool(publish_data.get("success", success))
            post_id = publish_data.get("post_id", post_id)
        except Exception as exc:  # pragma: no cover - external service
            logger.error("Xiaohongshu MCP call failed: %s", exc)

    return PublishResult(success=success, post_id=post_id, detail=resp if isinstance(resp, dict) else None)


# ---------- LangGraph wiring ----------


def build_app() -> StateGraph:
    load_dotenv()
    setup_logging()
    api_keys = load_api_keys()
    cost_tracker = CostTracker()

    qwen_client = ModelClient("qwen", api_keys.qwen, default_model=MODEL_CONFIG.get("qwen", {}).get("model"))
    deepseek_client = ModelClient("deepseek", api_keys.deepseek)
    longcat_client = ModelClient("longcat", api_keys.longcat)
    doubao_client = ModelClient("doubao", api_keys.doubao)
    gpt4_client = ModelClient("gpt4", api_keys.gpt4)

    graph_builder = StateGraph(WorkflowState)

    def node_determine_meal(state: WorkflowState) -> WorkflowState:
        now = datetime.now()
        meal_type = get_meal_type(now)
        logger.info("当前时间 %s，推荐餐次：%s", now.isoformat(timespec="minutes"), meal_type)
        return {
            **state,
            "current_time": now.isoformat(),
            "meal_type": meal_type,
            "cost": cost_tracker.total_cost,
        }

    def node_recipe(state: WorkflowState) -> WorkflowState:
        recipe = fetch_recipe(state["meal_type"], qwen_client, cost_tracker)
        logger.info("生成菜谱：%s", recipe.name)
        return {**state, "recipe_data": recipe.model_dump(), "cost": cost_tracker.total_cost}

    def node_content(state: WorkflowState) -> WorkflowState:
        recipe = Recipe(**state["recipe_data"])
        content = generate_content(recipe, deepseek_client, cost_tracker)
        return {**state, "content": content, "cost": cost_tracker.total_cost}

    def node_audit(state: WorkflowState) -> WorkflowState:
        result = content_audit(state["content"], longcat_client, cost_tracker)
        logger.info("审核结果：%s", result.ok)
        return {**state, "audit_result": result.ok, "audit_detail": result.model_dump(), "cost": cost_tracker.total_cost}

    def node_rewrite(state: WorkflowState) -> WorkflowState:
        # Re-run content generation with a safe guardrail.
        recipe = Recipe(**state["recipe_data"])
        safe_prompt = (
            "以安全、温和的表达重写文案，避免任何可能违规的描述，保持小红书风格和表情。"
        )
        cost_tracker.add("deepseek")
        resp = deepseek_client.invoke(safe_prompt + "\n原文：" + state["content"])
        content = resp.get("text") or resp.get("output") or state["content"]
        return {**state, "content": content, "audit_result": True, "cost": cost_tracker.total_cost}

    def node_images(state: WorkflowState) -> WorkflowState:
        recipe = Recipe(**state["recipe_data"])
        imgs = generate_images(recipe, doubao_client, cost_tracker)
        return {**state, "images": imgs, "cost": cost_tracker.total_cost}

    def node_publish(state: WorkflowState) -> WorkflowState:
        content = state["content"]
        imgs = state.get("images") or []
        result = publish_to_xiaohongshu(content, imgs, gpt4_client, cost_tracker)
        logger.info("发布结果：%s %s", result.success, result.post_id)
        return {
            **state,
            "publish_result": result.model_dump(),
            "cost": cost_tracker.total_cost,
        }

    graph_builder.add_node("determine_meal", node_determine_meal)
    graph_builder.add_node("generate_recipe", node_recipe)
    graph_builder.add_node("generate_content", node_content)
    graph_builder.add_node("audit_content", node_audit)
    graph_builder.add_node("rewrite_content", node_rewrite)
    graph_builder.add_node("generate_images", node_images)
    graph_builder.add_node("publish", node_publish)

    graph_builder.set_entry_point("determine_meal")
    graph_builder.add_edge("determine_meal", "generate_recipe")
    graph_builder.add_edge("generate_recipe", "generate_content")
    graph_builder.add_edge("generate_content", "audit_content")

    def audit_decision(state: WorkflowState) -> str:
        return "generate_images" if state.get("audit_result") else "rewrite_content"

    graph_builder.add_conditional_edges("audit_content", audit_decision, {"generate_images": "generate_images", "rewrite_content": "rewrite_content"})
    graph_builder.add_edge("rewrite_content", "generate_images")
    graph_builder.add_edge("generate_images", "publish")
    graph_builder.add_edge("publish", END)

    return graph_builder.compile()


def run_workflow() -> WorkflowState:
    app = build_app()
    # Initial state is minimal; graph computes everything else.
    initial_state: WorkflowState = {}
    final_state = app.invoke(initial_state)
    logger.info("流程完成，总成本估算：%.4f", float(final_state.get("cost", 0)))
    return final_state
