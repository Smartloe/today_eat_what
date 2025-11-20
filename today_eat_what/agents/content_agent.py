import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Ensure package import works when running as a script
ROOT = Path(__file__).resolve().parents[1].parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from today_eat_what.config import MODEL_CONFIG, load_api_keys
from today_eat_what.utils import load_dotenv, setup_logging

from today_eat_what.clients import CostTracker, ModelClient
from today_eat_what.models import Recipe


class ContentAgent:
    def __init__(self, deepseek_client: ModelClient, cost: CostTracker) -> None:
        self.deepseek = deepseek_client
        self.cost = cost
        self.generate_content_tool = tool("generate_content", return_direct=True)(self._generate_content)
        self._agent = None

    def _generate_content(self, recipe: dict) -> str:
        """生成小红书风格文案，包含标题与正文。"""
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
            return f"{title}\n{body}"
        except Exception as exc:  # pragma: no cover - LLM failure guard
            # Fallback本地模板，确保推理不中断。
            fallback = self._fallback_copy(recipe_obj)
            return fallback

    def _fallback_copy(self, recipe_obj: Recipe) -> str:
        tags = ["#家常菜", "#当季食材", f"#{recipe_obj.meal_type}灵感"]
        body = (
            f"{recipe_obj.name} | {recipe_obj.meal_type}灵感\n"
            f"食材：{', '.join(recipe_obj.ingredients)}\n"
            f"步骤亮点：{' / '.join([s.instruction for s in recipe_obj.steps[:3]])}\n"
            f"口味：{recipe_obj.description}\n"
            f"{' '.join(tags)}"
        )
        return f"美味上线 🍴\n{body}"

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


if __name__ == "__main__":
    import argparse

    load_dotenv()
    setup_logging()
    parser = argparse.ArgumentParser(description="Test ContentAgent with a sample recipe.")
    parser.add_argument("--recipe-path", help="JSON文件路径，内容为 recipe 字典")
    parser.add_argument("--use-agent", action="store_true", help="使用 create_agent 包装的智能体调用")
    args = parser.parse_args()

    sample_recipe = {
        "name": "秋季限定小吃组合",
        "meal_type": "小吃",
        "description": "当季南瓜、西兰花等食材，小吃属性，时间适中。",
        "ingredients": [
            "西兰花1颗（约300g）",
            "蒜末15g",
            "蚝油10ml",
            "盐3g",
            "面粉200g",
            "芝麻50g",
            "酵母3g",
            "温水100ml",
            "猪里脊肉200g",
            "苹果1个",
            "白糖30g",
            "白醋15ml",
            "南瓜500g",
            "冰糖20g",
            "枸杞10g",
        ],
        "steps": [
            {"order": 1, "instruction": "蒜蓉西兰花：焯水过冷水，蒜末爆香翻炒，淋蚝油。"},
            {"order": 2, "instruction": "芝麻烧饼：面团发酵后擀平抹油，撒芝麻刷蛋液，烤制。"},
            {"order": 3, "instruction": "糖醋里脊：腌制裹粉油炸，调糖醋汁淋上。"},
            {"order": 4, "instruction": "蒸南瓜：切块蒸熟后加冰糖再蒸，撒枸杞。"},
        ],
    }

    recipe_data = sample_recipe
    if args.recipe_path:
        path = Path(args.recipe_path)
        recipe_data = json.loads(path.read_text())

    keys = load_api_keys()
    cost = CostTracker()
    deepseek_client = ModelClient("deepseek", keys.deepseek, default_model=MODEL_CONFIG.get("deepseek", {}).get("model"))
    agent = ContentAgent(deepseek_client, cost)
    if args.use_agent:
        lc_agent = agent.get_agent()
        output = lc_agent.invoke({"messages": [{"role": "user", "content": f"请为以下菜谱写小红书文案：{recipe_data}"}]})
        if hasattr(output, "messages"):
            content = output.messages[-1].content
        else:
            content = output
    else:
        output = agent.generate_content_tool.invoke({"recipe": recipe_data})
        content = output
    print("------ 文案输出 ------")
    print(content)
    print("成本估算：", cost.total_cost)
