import json
import logging
from typing import Any, Dict, List

from langchain_core.tools import tool

from ..clients import CostTracker, ModelClient
from ..models import Recipe, RecipeStep
from ..services import call_how_to_cook

logger = logging.getLogger(__name__)


class RecipeAgent:
    def __init__(self, qwen_client: ModelClient, cost: CostTracker) -> None:
        self.qwen = qwen_client
        self.cost = cost

    @tool("generate_recipe", return_direct=True)
    def generate_recipe_tool(self, meal_type: str) -> Dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": "你是美食助理，输出 JSON 对象 {\"recipe\": {...}}，"
                "包含 name/description/ingredients(list)/steps(list: {order, instruction})。",
            },
            {"role": "user", "content": f"餐次：{meal_type}，请给出适合这一餐次的1道菜。"},
        ]
        self.cost.add("qwen")
        llm_resp = self.qwen.invoke_chat(messages, extra={"temperature": 0.4})
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
                    logger.warning("Qwen 返回非JSON内容，尝试使用 HowToCook MCP：%s", content[:50])

        if not recipe_data:
            recipe_data = call_how_to_cook(meal_type).get("recipe", {})

        recipe = self._build_recipe(recipe_data, meal_type)
        return recipe.model_dump()

    def _build_recipe(self, recipe_data: Dict[str, Any], meal_type: str) -> Recipe:
        name = recipe_data.get("name") or f"{meal_type}推荐"
        description = recipe_data.get("description") or "轻松上手的美味搭配"
        ingredients: List[str] = recipe_data.get("ingredients") or ["根据口味准备常用食材"]
        steps_raw = recipe_data.get("steps") or [{"order": 1, "instruction": "按常规方法烹饪至熟。"}]
        steps = [RecipeStep(order=s.get("order", i + 1), instruction=s.get("instruction", "")) for i, s in enumerate(steps_raw)]
        return Recipe(name=name, description=description, ingredients=ingredients, steps=steps, meal_type=meal_type)
