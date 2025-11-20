from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from ..clients import CostTracker, ModelClient
from ..models import Recipe


class ContentAgent:
    def __init__(self, deepseek_client: ModelClient, cost: CostTracker) -> None:
        self.deepseek = deepseek_client
        self.cost = cost

    @tool("generate_content", return_direct=True)
    def generate_content_tool(self, recipe: dict) -> str:
        recipe_obj = Recipe(**recipe)
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

        self.cost.add("deepseek")
        title_resp = self.deepseek.invoke(title_prompt.format(name=recipe_obj.name, meal_type=recipe_obj.meal_type))
        body_resp = self.deepseek.invoke(
            body_prompt.format(
                description=recipe_obj.description,
                ingredients=", ".join(recipe_obj.ingredients),
                steps=" / ".join([s.instruction for s in recipe_obj.steps]),
            )
        )
        title = title_resp.get("text") or title_resp.get("output") or "美味上线"
        body = body_resp.get("text") or body_resp.get("output") or ""
        return f"{title}\n{body}"
