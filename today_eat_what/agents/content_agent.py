import os
import sys
from pathlib import Path
from typing import Dict
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Ensure package import works when running as a script
ROOT = Path(__file__).resolve().parents[1].parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from today_eat_what.clients import CostTracker, ModelClient
from today_eat_what.models import Recipe


class ContentAgent:
    def __init__(self, deepseek_client: ModelClient, cost: CostTracker) -> None:
        self.deepseek = deepseek_client
        self.cost = cost
        self.generate_content_tool = tool("generate_content", return_direct=True)(self._generate_content)
        self._agent = None

    def _generate_content(self, recipe: dict) -> Dict[str, str]:
        """生成小红书风格文案，返回 JSON: {title, body, content}。"""
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
        try:
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
            if not body or not title:
                raise ValueError("LLM returned empty content")
            return {"title": title, "body": body, "content": f"{title}\n{body}"}
        except Exception as exc:  # pragma: no cover - LLM failure guard
            # Fallback本地模板，确保推理不中断。
            fallback = self._fallback_copy(recipe_obj)
            return fallback

    def _fallback_copy(self, recipe_obj: Recipe) -> Dict[str, str]:
        tags = ["#家常菜", "#当季食材", f"#{recipe_obj.meal_type}灵感"]
        body = (
            f"{recipe_obj.name} | {recipe_obj.meal_type}灵感\n"
            f"食材：{', '.join(recipe_obj.ingredients)}\n"
            f"步骤亮点：{' / '.join([s.instruction for s in recipe_obj.steps[:3]])}\n"
            f"口味：{recipe_obj.description}\n"
            f"{' '.join(tags)}"
        )
        title = "美味上线 🍴"
        return {"title": title, "body": body, "content": f"{title}\n{body}"}

    def get_agent(self):
        """使用 LangChain create_agent 包装为完整智能体（以 DeepSeek/Hunyuan/OpenAI 兼容接口为模型）。"""
        if self._agent:
            return self._agent
        # 尝试用 DeepSeek API，如果未配置则回退到 OpenAI 兼容参数。
        llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url=os.environ.get("DEEPSEEK_BASE_URL"),
            temperature=0.7,
        )
        system_prompt = "你是小红书美食创作者，擅长写吸睛标题和口味、步骤亮点描述。"
        self._agent = create_agent(model=llm, tools=[self.generate_content_tool], system_prompt=system_prompt)
        return self._agent
