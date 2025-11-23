import os
import re
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
from today_eat_what.config import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL_DEFAULT
from today_eat_what.models import Recipe
from today_eat_what.utils import load_dotenv


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
            body_raw = body_resp.get("text") or body_resp.get("output") or ""
            if not body_raw or not title:
                raise ValueError("LLM returned empty content")
            header = f"今天周{weekday} | {recipe_obj.meal_type}"
            body_main = re.sub(r"^今天?周[一二三四五六日天][^\n]*\n?", "", body_raw.strip())
            body = f"{header}\n{body_main}".strip()
            body = self._normalize_weekday(body, weekday, recipe_obj.meal_type)
            base_tags = [
                "今日吃什么呢",
                f"{recipe_obj.meal_type}灵感",
                "快速上桌",
                "营养均衡",
            ]
            content_text, tags = self._split_tags(body, base_tags=base_tags)
            content = f"{title}\n{content_text}"
            return {"title": title, "content": content, "tags": tags}
        except Exception:
            # Fallback本地模板，确保推理不中断。
            fallback = self._fallback_copy(recipe_obj, dish_names, summary_parts, weekday)
            return fallback

    def _fallback_copy(self, recipe_obj: Recipe, dish_names: List[str], summary_parts: List[str], weekday: str) -> Dict[str, str]:
        tags = ["今天吃什么呢", "当季食材", f"{recipe_obj.meal_type}灵感"]
        body = f"今天周{weekday} | {recipe_obj.meal_type}\n" f"{'；'.join(summary_parts)}"
        title = f"{' + '.join(dish_names[:3])} | 今日餐单 🍽️"
        return {"title": title, "content": f"{title}\n{body}", "tags": tags}

    @staticmethod
    def _normalize_weekday(body: str, weekday: str, meal_type: str) -> str:
        """确保正文中的周几与当前一致，避免模型胡写。"""
        correct = f"今天周{weekday} | {meal_type}"
        # 替换开头任何“周X”表述为正确行。
        body = re.sub(r"^今天?周[一二三四五六日天][^\n]*", correct, body.strip())
        # 将正文其他出现的“周X”统一替换为当前周几，避免矛盾。
        body = re.sub(r"周[一二三四五六日天]", f"周{weekday}", body)
        return body

    @staticmethod
    def _split_tags(text: str, base_tags: List[str]) -> (str, List[str]):
        """从正文中抽取 #标签，移除后返回纯文本和标签列表。"""
        hashtags = re.findall(r"#([^\s#]+)", text)
        merged: List[str] = []
        for t in [*base_tags, *hashtags]:
            if t and t not in merged:
                merged.append(t)
        # 保证标签数量不少于5个，填充默认口味/场景标签。
        filler_pool = ["家常好菜", "下饭菜", "快手菜", "暖心餐", "今日份美食", "解馋必备"]
        for t in filler_pool:
            if len(merged) >= 6:
                break
            if t not in merged:
                merged.append(t)
        cleaned = re.sub(r"#([^\s#]+)", "", text)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        return cleaned, merged

    def get_agent(self):
        """使用 LangChain create_agent 包装为完整智能体（以 DeepSeek/Hunyuan/OpenAI 兼容接口为模型）。"""
        if self._agent:
            return self._agent
        # 尝试用 DeepSeek API，如果未配置则回退到 OpenAI 兼容参数。
        model_name = DEEPSEEK_MODEL_DEFAULT
        if not model_name:
            raise RuntimeError("DEEPSEEK_MODEL 未设置，无法生成文案")
        llm = ChatOpenAI(
            model=model_name,
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url=DEEPSEEK_BASE_URL or None,
            temperature=0.7,
        )
        system_prompt ="""
        你是一位非常受欢迎的小红书美食创作达人，你的笔记标题总是能让人忍不住想点击，内容描述能让人立刻收藏。
请为 [你的菜品] 创作一篇笔记。
核心指令：
1. 标题：必须包含一个爆款关键词（如：绝了、封神、尖叫、求你们去做），并巧妙搭配1-2个相关Emoji（如：🔥、🍳、💥）。
2. 口味描述：不使用“好吃”等空洞词汇，而是从 口感（如：外酥里嫩、入口即化）、风味（如：蒜香浓郁、酱香回甘）、香气（如：满屋飘香） 三个维度进行刻画。
3. 步骤提炼：不写完整菜谱，只提炼1-2个最关键、最能让读者感觉“简单又厉害”的步骤亮点，并点明它为谁省了事（如：打工族/宝妈/懒人），例如“10分钟搞定”、“免烤箱”、“一锅出”。
4. 平台话术：在描述中自然融入“真的巨巨巨…”、“我不允许还有人没吃过…”等小红书特色语气。
"""
        self._agent = create_agent(model=llm, tools=[self.generate_content_tool], system_prompt=system_prompt)
        return self._agent


if __name__ == "__main__":
    import json
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    load_dotenv()
    sample_recipe = {
        "name": "麻辣香锅",
        "description": "热辣鲜香",
        "meal_type": "晚餐",
        "dishes": [
            {
                "name": "麻辣香锅",
                "description": "麻辣鲜香",
                "ingredients": ["土豆片", "藕片", "牛肉", "辣椒"],
                "steps": [
                    {"order": 1, "instruction": "处理食材切片"},
                    {"order": 2, "instruction": "锅中炒制底料"},
                    {"order": 3, "instruction": "下入食材翻炒入味"},
                ],
            }
        ],
    }

    deepseek_client = ModelClient("deepseek", api_key=os.environ.get("DEEPSEEK_API_KEY"))
    agent = ContentAgent(deepseek_client, CostTracker())
    result = agent.generate_content_tool.invoke({"recipe": sample_recipe})
    print(json.dumps(result, ensure_ascii=False, indent=2))
