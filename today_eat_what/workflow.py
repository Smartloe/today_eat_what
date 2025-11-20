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
    content: str
    audit_result: bool
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
    gpt4_client = ModelClient("gpt4", api_keys.gpt4)

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

    def node_recipe(state: WorkflowState) -> WorkflowState:
        recipe_data = recipe_agent.generate_recipe_tool.invoke({"meal_type": state["meal_type"]})
        recipe = Recipe(**recipe_data)
        logger.info("生成菜谱：%s", recipe.name)
        return {**state, "recipe_data": recipe_data, "cost": cost_tracker.total_cost}

    def node_content(state: WorkflowState) -> WorkflowState:
        content = content_agent.generate_content_tool.invoke({"recipe": state["recipe_data"]})
        return {**state, "content": content, "cost": cost_tracker.total_cost}

    def node_audit(state: WorkflowState) -> WorkflowState:
        result = audit_agent.audit_content_tool.invoke({"content": state["content"]})
        logger.info("审核结果：%s", result.get("ok"))
        return {**state, "audit_result": result.get("ok"), "audit_detail": result, "cost": cost_tracker.total_cost}

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
        imgs = image_agent.generate_images_tool.invoke({"recipe": state["recipe_data"]})
        return {**state, "images": imgs, "cost": cost_tracker.total_cost}

    def node_publish(state: WorkflowState) -> WorkflowState:
        content = state["content"]
        imgs = state.get("images") or []
        result = publish_agent.publish_tool.invoke({"content": content, "images": imgs})
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
