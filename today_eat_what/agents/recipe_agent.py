import os
import json
import asyncio
import dotenv
import logging
from datetime import datetime
from typing import Dict, List, Optional
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

"""
菜谱智能体（类封装）：
- 优先调用 mcp_howtocook_whatToEat 按时段/季节推荐组合
- 如不合适则调用 mcp_howtocook_getAllRecipes，筛选符合当前季节和餐次
- 最终返回菜谱详情（菜名、食材、步骤、时间、贴士）
"""

dotenv.load_dotenv()
_mcp_logger = logging.getLogger("mcp.client.stdio")
_mcp_logger.setLevel(logging.CRITICAL)
_mcp_logger.propagate = False


def _now() -> datetime:
    return datetime.now()


def get_meal_type(now: Optional[datetime] = None) -> str:
    now = now or _now()
    h = now.hour
    if 6 <= h <= 10:
        return "早餐"
    if 11 <= h <= 14:
        return "午餐"
    if 17 <= h <= 21:
        return "晚餐"
    return "小吃"


def get_season(now: Optional[datetime] = None) -> str:
    now = now or _now()
    m = now.month
    if m in (3, 4, 5):
        return "春季"
    if m in (6, 7, 8):
        return "夏季"
    if m in (9, 10, 11):
        return "秋季"
    return "冬季"


class RecipeAgent:
    def __init__(self, people: int = 1, dislikes: str = "无偏好") -> None:
        self.people = people
        self.dislikes = dislikes
        self.agent = None

    def _init_model(self) -> ChatOpenAI:
        """初始化 Qwen 模型（SiliconFlow 兼容接口）。"""
        return ChatOpenAI(
            model=os.environ.get("QWEN_MODEL", "Qwen/Qwen3-8B"),
            api_key=os.environ.get("SILICONFLOW_API_KEY"),
            base_url=os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
            temperature=0.35,
            max_tokens=1800,
        )

    async def _load_tools(self) -> List:
        tools: List = []
        try:
            client = MultiServerMCPClient(
                {
                    "howtocook": {
                        "transport": "stdio",
                        "command": "npx",
                        "args": ["-y", "howtocook-mcp"],
                    }
                }
            )
            tools = await asyncio.wait_for(client.get_tools(), timeout=20)
            if not tools:
                print("⚠️ MCP 未返回工具，将以无工具模式运行。")
        except Exception as exc:  # pragma: no cover - external MCP
            print(f"⚠️ MCP 连接警告: {exc}")
            print("将以离线模式运行（无 HowToCook 工具）")
            tools = []
        return tools

    async def setup(self) -> None:
        """初始化 Agent，仅调用一次。"""
        model = self._init_model()
        tools = await self._load_tools()
        system_prompt = f"""
你是一个专业的烹饪助手。优先使用 MCP 工具，策略如下：
- 优先调用 mcp_howtocook_whatToEat 根据人数/饮食偏好/当前餐次/季节直接给出组合。
- 若组合不符合当前季节或餐次，调用 mcp_howtocook_getAllRecipes 拉取全部菜谱，再筛选出符合“{get_season()}”和“{get_meal_type()}”的菜谱。
- 输出严格 JSON，字段：name, description, ingredients(List[str]), steps(List[{{order:int, instruction:str}}]), meal_type(当前餐次)。
- 仅输出 JSON，不要多余文字。
- 若无工具可用，直接用模型生成符合季节与餐次的家常菜谱，并按上述 JSON 返回。
"""
        self.agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt.strip(),
        )

    def _parse_recipe(self, content: str, meal_type: str) -> Optional[Dict]:
        """解析模型/工具返回的文本为结构化 recipe dict，失败返回 None。"""
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                parsed.setdefault("meal_type", meal_type)
                return parsed
        except Exception:
            pass
        return None

    def _generate_local_recipe(self, meal: str, season: str, people: int, dislikes: str) -> Dict:
        """当工具返回不可解析时，直接用模型生成 JSON 菜谱。"""
        llm = self._init_model()
        prompt = f"""
请生成一份 JSON 菜谱，字段：
- name: 菜名
- description: 20字内简介
- ingredients: 食材数组（带数量）
- steps: 数组，每项 {{"order":编号, "instruction":步骤描述}}
- meal_type: "{meal}"
约束：适合 {people} 人，当前餐次 {meal}，季节 {season}，忌口/过敏：{dislikes}。
仅输出 JSON，不要多余文字。
"""
        try:
            resp = llm.invoke(prompt).content
            parsed = self._parse_recipe(resp, meal)
            if parsed:
                return parsed
        except Exception:
            logger.error("本地模型生成菜谱失败", exc_info=True)
        return {}

    async def generate_recipe(self, people: Optional[int] = None, dislikes: Optional[str] = None) -> Dict:
        """生成菜谱（优先 MCP 工具，带季节/餐次约束）。"""
        if not self.agent:
            await self.setup()
        meal = get_meal_type()
        season = get_season()
        people = people or self.people
        dislikes = dislikes or self.dislikes

        user_message = (
            f"请推荐适合 {people} 人的菜品组合，当前餐次：{meal}，季节：{season}，"
            f"忌口/过敏：{dislikes}。"
            "优先使用 mcp_howtocook_whatToEat，若不合适再用 mcp_howtocook_getAllRecipes 过滤符合餐次+季节的菜。"
        )

        print(f"\n{'='*50}\n正在生成 {meal} 菜单（{season}，人数 {people}，忌口 {dislikes}）\n{'='*50}\n")

        try:
            result: Dict = await self.agent.ainvoke(
                {
                    "messages": [{"role": "user", "content": user_message}]
                }
            )
            if result.get("messages"):
                final_message = result["messages"][-1]
                content = final_message.content
                print(f"✅ 菜谱生成结果:\n{content}\n")
                parsed = self._parse_recipe(content, meal_type=meal)
                if parsed:
                    return parsed
                print("⚠️ 未解析出有效 JSON，改为直接模型生成。")
                return self._generate_local_recipe(meal, season, people, dislikes)
            return self._generate_local_recipe(meal, season, people, dislikes)
        except Exception as exc:  # pragma: no cover - runtime guardrail
            print(f"❌ 生成菜谱失败: {exc!r}")
            return self._generate_local_recipe(meal, season, people, dislikes)


async def main():
    print("🍳 HowToCook 菜谱智能体启动...\n")
    agent = RecipeAgent(people=1, dislikes="无偏好")
    recipe = await agent.generate_recipe()
    print("最终 JSON 菜谱：")
    print(json.dumps(recipe, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
