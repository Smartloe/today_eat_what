import logging
from datetime import datetime
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from .agents import AuditAgent, ContentAgent, ImageAgent, PublishAgent, RecipeAgent
from .clients import CostTracker, ModelClient
from .config import MODEL_CONFIG, load_api_keys
from .models import Recipe
from .services import get_meal_type
from .utils import load_dotenv, setup_logging

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict, total=False):
    current_time: str
    meal_type: str
    recipe_data: Dict[str, Any]
    content: Dict[str, str]
    audit_result: bool
    rewrite_attempted: bool
    images: List[str]
    publish_result: Dict[str, Any]
    cost: float


def build_app() -> StateGraph:
    load_dotenv()
    setup_logging()
    api_keys = load_api_keys()
    cost_tracker = CostTracker()

    qwen_client = ModelClient("qwen", api_keys.qwen, default_model=MODEL_CONFIG.get("qwen", {}).get("model"))
    deepseek_client = ModelClient("deepseek", api_keys.deepseek)
    longcat_client = ModelClient("longcat", api_keys.longcat)
    doubao_client = ModelClient("doubao", api_keys.doubao)
    gpt4_client = ModelClient("glm", api_keys.glm)

    recipe_agent = RecipeAgent(qwen_client, cost_tracker)
    content_agent = ContentAgent(deepseek_client, cost_tracker)
    audit_agent = AuditAgent(longcat_client, cost_tracker)
    image_agent = ImageAgent(doubao_client, cost_tracker)
    publish_agent = PublishAgent(gpt4_client, cost_tracker)

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

    def _normalize_recipe_data(data: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(data)
        ingredients = normalized.get("ingredients") or []
        if any(isinstance(i, dict) for i in ingredients):
            ing_list: List[str] = []
            for item in ingredients:
                if isinstance(item, dict):
                    name = item.get("ingredient") or item.get("name") or ""
                    qty = item.get("quantity") or item.get("qty") or ""
                    combined = f"{name} {qty}".strip()
                    if combined:
                        ing_list.append(combined)
                else:
                    ing_list.append(str(item))
            normalized["ingredients"] = ing_list
        steps = normalized.get("steps") or []
        fixed_steps: List[Dict[str, Any]] = []
        for idx, s in enumerate(steps, start=1):
            if isinstance(s, dict):
                order = s.get("order") or idx
                instr = s.get("instruction") or s.get("step") or ""
                fixed_steps.append({"order": order, "instruction": instr})
            else:
                fixed_steps.append({"order": idx, "instruction": str(s)})
        normalized["steps"] = fixed_steps
        dishes = normalized.get("dishes") or []
        fixed_dishes: List[Dict[str, Any]] = []
        for dish in dishes:
            if not isinstance(dish, dict):
                continue
            d = dict(dish)
            d_ings = d.get("ingredients") or []
            if any(isinstance(i, dict) for i in d_ings):
                ing_list: List[str] = []
                for item in d_ings:
                    if isinstance(item, dict):
                        name = item.get("ingredient") or item.get("name") or ""
                        qty = item.get("quantity") or item.get("qty") or ""
                        combo = f"{name} {qty}".strip()
                        if combo:
                            ing_list.append(combo)
                    else:
                        ing_list.append(str(item))
                d["ingredients"] = ing_list
            d_steps = d.get("steps") or []
            ds_fixed: List[Dict[str, Any]] = []
            for idx, s in enumerate(d_steps, start=1):
                if isinstance(s, dict):
                    order = s.get("order") or idx
                    instr = s.get("instruction") or s.get("step") or ""
                    ds_fixed.append({"order": order, "instruction": instr})
                else:
                    ds_fixed.append({"order": idx, "instruction": str(s)})
            d["steps"] = ds_fixed
            fixed_dishes.append(d)
        if fixed_dishes:
            normalized["dishes"] = fixed_dishes
            if not normalized.get("ingredients"):
                combined_ings: List[str] = []
                for dish in fixed_dishes:
                    combined_ings.extend(dish.get("ingredients", []))
                normalized["ingredients"] = combined_ings
        return normalized

    def node_recipe(state: WorkflowState) -> WorkflowState:
        recipe_data = recipe_agent.generate_recipe_tool.invoke({"meal_type": state["meal_type"]})
        recipe_data = _normalize_recipe_data(recipe_data)
        recipe = Recipe(**recipe_data)
        logger.info("生成菜谱：%s", recipe.name)
        return {**state, "recipe_data": recipe_data, "cost": cost_tracker.total_cost}

    def node_content(state: WorkflowState) -> WorkflowState:
        content = content_agent.generate_content_tool.invoke({"recipe": state["recipe_data"]})
        return {**state, "content": content, "cost": cost_tracker.total_cost}

    def node_audit(state: WorkflowState) -> WorkflowState:
        content_text = state["content"].get("content") if isinstance(state.get("content"), dict) else state["content"]
        result = audit_agent.audit_content_tool.invoke({"content": content_text})
        logger.info("审核结果：%s", result.get("ok"))
        return {**state, "audit_result": result.get("ok"), "audit_detail": result, "cost": cost_tracker.total_cost}

    def node_rewrite(state: WorkflowState) -> WorkflowState:
        # Re-run content generation with a safe guardrail.
        recipe = Recipe(**state["recipe_data"])
        safe_prompt = (
            "以安全、温和的表达重写文案，避免任何可能违规的描述，保持小红书风格和表情。"
        )
        cost_tracker.add("deepseek")
        original = state["content"].get("content") if isinstance(state["content"], dict) else str(state["content"])
        resp = deepseek_client.invoke(safe_prompt + "\n原文：" + original)
        new_body = resp.get("text") or resp.get("output") or original
        rewritten = {
            "title": state["content"].get("title", "安全改写") if isinstance(state["content"], dict) else "安全改写",
            "body": new_body,
            "content": f"{state['content'].get('title', '安全改写')}\n{new_body}"
            if isinstance(state["content"], dict)
            else new_body,
        }
        return {
            **state,
            "content": rewritten,
            "audit_result": False,
            "rewrite_attempted": True,
            "cost": cost_tracker.total_cost,
        }

    def node_images(state: WorkflowState) -> WorkflowState:
        imgs_result = image_agent.generate_images_tool.invoke({"recipe": state["recipe_data"]})
        imgs = imgs_result.get("images") if isinstance(imgs_result, dict) else imgs_result
        return {**state, "images": imgs, "cost": cost_tracker.total_cost}

    def node_publish(state: WorkflowState) -> WorkflowState:
        content = state["content"].get("content") if isinstance(state["content"], dict) else state["content"]
        imgs = state.get("images") or []
        tags = ["今天吃什么呢"]
        result = publish_agent.publish_tool.invoke({"content": content, "images": imgs, "tags": tags})
        logger.info("发布结果：%s %s", result.get("success"), result.get("post_id"))
        return {
            **state,
            "publish_result": result,
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
        if state.get("audit_result"):
            return "generate_images"
        if state.get("rewrite_attempted"):
            # 已重写过一次仍未通过，直接进入生成图片/发布，避免死循环
            return "generate_images"
        return "rewrite_content"

    graph_builder.add_conditional_edges("audit_content", audit_decision, {"generate_images": "generate_images", "rewrite_content": "rewrite_content"})
    graph_builder.add_edge("rewrite_content", "audit_content")
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
