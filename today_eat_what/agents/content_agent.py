import os
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime
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
        dishes = recipe.get("dishes") or []
        dish_names = [d.get("name") for d in dishes if isinstance(d, dict) and d.get("name")]
        if not dish_names:
            dish_names = [recipe_obj.name]
        weekday = ["一", "二", "三", "四", "五", "六", "日"][datetime.now().weekday()]

        summary_parts: List[str] = []
        for dish in dishes:
            if not isinstance(dish, dict):
                continue
            name = dish.get("name") or ""
            desc = dish.get("description") or ""
            ing = dish.get("ingredients") or []
            summary_parts.append(f"{name}：{desc}｜食材：{', '.join(ing[:4])}")
        if not summary_parts:
            summary_parts.append(f"{recipe_obj.name}：{recipe_obj.description}")

        title_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "你是小红书美食创作者，写20字内的吸睛标题，带1个表情。"),
                ("human", "餐次：{meal_type}，菜品：{dishes}"),
            ]
        )
        body_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "用小红书口吻写一段文案，包含：今天周几+餐次开场、每道菜的搭配理由/亮点、2-3个话题标签，带表情符号，控制在180字内。",
                ),
                ("human", "餐次：{meal_type}，菜品详情：{dish_summary}"),
            ]
        )

        self.cost.add("deepseek")
        try:
            title_resp = self.deepseek.invoke(
                title_prompt.format(meal_type=recipe_obj.meal_type, dishes=" + ".join(dish_names))
            )
            body_resp = self.deepseek.invoke(
                body_prompt.format(meal_type=recipe_obj.meal_type, dish_summary=" / ".join(summary_parts))
            )
            title = title_resp.get("text") or title_resp.get("output") or "美味上线"
            body = body_resp.get("text") or body_resp.get("output") or ""
            if not body or not title:
                raise ValueError("LLM returned empty content")
            body = f"今天周{weekday} | {recipe_obj.meal_type}\n" + body
            return {"title": title, "body": body, "content": f"{title}\n{body}"}
        except Exception:
            # Fallback本地模板，确保推理不中断。
            fallback = self._fallback_copy(recipe_obj, dish_names, summary_parts, weekday)
            return fallback

    def _fallback_copy(self, recipe_obj: Recipe, dish_names: List[str], summary_parts: List[str], weekday: str) -> Dict[str, str]:
        tags = ["#家常菜", "#当季食材", f"#{recipe_obj.meal_type}灵感"]
        body = (
            f"今天周{weekday} | {recipe_obj.meal_type}\n"
            f"{'；'.join(summary_parts)}\n"
            f"{' '.join(tags)}"
        )
        title = f"{' + '.join(dish_names[:3])} | 今日餐单 🍽️"
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
