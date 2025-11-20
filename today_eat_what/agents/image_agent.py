from typing import List

from langchain_core.tools import tool

from ..clients import CostTracker, ModelClient
from ..models import Recipe


class ImageAgent:
    def __init__(self, doubao_client: ModelClient, cost: CostTracker) -> None:
        self.doubao = doubao_client
        self.cost = cost

    @tool("generate_images", return_direct=True)
    def generate_images_tool(self, recipe: dict) -> List[str]:
        recipe_obj = Recipe(**recipe)
        prompt = (
            f"生成符合小红书风格的菜品图片：{recipe_obj.name}，描述：{recipe_obj.description}。"
            "再生成按步骤的制作过程插画，每步1张。输出JSON {'urls': [list]}"
        )
        self.cost.add("doubao")
        resp = self.doubao.invoke(prompt)
        urls = resp.get("urls") if isinstance(resp, dict) else None
        if not urls:
            urls = [f"https://imgs.local/{recipe_obj.name}_cover.png"] + [
                f"https://imgs.local/{recipe_obj.name}_step_{step.order}.png" for step in recipe_obj.steps
            ]
        return urls
