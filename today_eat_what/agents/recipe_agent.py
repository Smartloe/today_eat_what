import os
import json
import asyncio
import dotenv
import logging
from datetime import datetime
from typing import Dict, List, Optional
import sys
from pathlib import Path
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import tool

ROOT = Path(__file__).resolve().parents[1].parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from today_eat_what.clients import CostTracker, ModelClient
from today_eat_what.config import QWEN_BASE_URL, QWEN_MODEL_DEFAULT

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
logger = logging.getLogger(__name__)


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
    def __init__(
        self,
        model_client: Optional[ModelClient] = None,
        cost: Optional[CostTracker] = None,
        people: int = 1,
        dislikes: str = "无偏好",
    ) -> None:
        self.people = people
        self.dislikes = dislikes
        self.agent = None
        self._llm = None
        self._llm_unavailable = False
        self._setup_done = False
        self.model_client = model_client
        self.cost = cost or CostTracker()
        self.generate_recipe_tool = tool("generate_recipe", return_direct=True)(self._generate_recipe_sync)

    def _init_model(self) -> Optional[ChatOpenAI]:
        """初始化 Qwen 模型（SiliconFlow 兼容接口）。缺失配置时回退模板。"""
        if self._llm is not None or self._llm_unavailable:
            return self._llm
        model_name = (
            QWEN_MODEL_DEFAULT
            or os.environ.get("QWEN_MODEL")
            or os.environ.get("Qwen_MODEL")
            or os.environ.get("Qwen_MODEL")
        )
        if not model_name:
            logger.warning("QWEN_MODEL/Qwen_MODEL 未设置，将使用内置菜谱模板。")
            self._llm_unavailable = True
            return None
        api_key = os.environ.get("QWEN_API_KEY") or os.environ.get("Qwen_API_KEY") or os.environ.get("Qwen_API_KEY")
        if not api_key:
            logger.warning("QWEN_API_KEY/Qwen_API_KEY 未设置，将使用内置菜谱模板。")
            self._llm_unavailable = True
            return None
        try:
            self._llm = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url=QWEN_BASE_URL,
                temperature=0.35,
                max_tokens=1800,
            )
            return self._llm
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning("初始化 Qwen 模型失败，将使用内置菜谱模板：%s", exc)
            self._llm_unavailable = True
            return None

    async def _load_tools(self) -> List:
        # 支持 streamable_http MCP (优先环境变量 URL)，否则回退到 stdio npx。
        from today_eat_what.config import HOWTOCOOK_MCP_URL

        tools: List = []
        server_name = "howtocook"
        connections = (
            {server_name: {"transport": "streamable_http", "url": HOWTOCOOK_MCP_URL}}
            if HOWTOCOOK_MCP_URL
            else {server_name: {"transport": "stdio", "command": "npx", "args": ["-y", "howtocook-mcp"]}}
        )
        try:
            client = MultiServerMCPClient(connections)
            tools = await asyncio.wait_for(client.get_tools(server_name=server_name), timeout=20)
            if not tools:
                logger.warning("MCP 未返回工具，将以无工具模式运行。")
        except Exception as exc:  # pragma: no cover - external MCP
            logger.warning("MCP 连接警告: %s，将以离线模式运行（无 HowToCook 工具）", exc)
            tools = []
        return tools

    async def setup(self) -> None:
        """初始化 Agent，仅调用一次。"""
        if self._setup_done:
            return
        self._setup_done = True
        model = self._init_model()
        if not model:
            return
        tools = await self._load_tools()
        system_prompt = f"""
你是一个专业的烹饪助手。优先使用 MCP 工具，策略如下：
- 优先调用 mcp_howtocook_whatToEat 根据人数/饮食偏好/当前餐次/季节直接给出组合。
- 若组合不符合当前季节或餐次，调用 mcp_howtocook_getAllRecipes 拉取全部菜谱，再筛选出符合“{get_season()}”和“{get_meal_type()}”的菜谱。
- 输出严格 JSON，字段：
  name: 餐单标题（例如“{get_meal_type()}双拼组合”）
  description: 20字内餐单简介
  meal_type: 当前餐次
  dishes: List，长度 2-3；每项 {{"name": 菜名, "description": 简介, "ingredients": List[str], "steps": List[{{"order":int,"instruction":str}}]}}
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
请生成一份 JSON 餐单，字段：
- name: 餐单标题
- description: 20字内简介
- meal_type: "{meal}"
- dishes: 长度2-3的数组，每项 {{"name":菜名, "description":简介, "ingredients":食材数组（带数量）, "steps":数组，每项 {{"order":编号,"instruction":步骤描述}}}}
约束：适合 {people} 人，当前餐次 {meal}，季节 {season}，忌口/过敏：{dislikes}。仅输出 JSON，不要多余文字。
"""
        if not llm:
            raise RuntimeError("未配置 Qwen 模型，且 MCP 输出不可用，无法生成菜谱。")
        resp = llm.invoke(prompt).content
        parsed = self._parse_recipe(resp, meal)
        if parsed:
            return parsed
        raise RuntimeError("Qwen 返回内容无法解析为 JSON 菜谱，请重试或检查提示词。")

    async def generate_recipe(self, people: Optional[int] = None, dislikes: Optional[str] = None, meal_type: Optional[str] = None) -> Dict:
        """生成菜谱（优先 MCP 工具，带季节/餐次约束）。"""
        await self.setup()
        meal = meal_type or get_meal_type()
        season = get_season()
        people = people or self.people
        dislikes = dislikes or self.dislikes

        user_message = (
            f"请推荐适合 {people} 人的菜品组合，当前餐次：{meal}，季节：{season}，"
            f"忌口/过敏：{dislikes}。"
            "优先使用 mcp_howtocook_whatToEat，若不合适再用 mcp_howtocook_getAllRecipes 过滤符合餐次+季节的菜。"
        )

        if self.agent:
            try:
                result: Dict = await self.agent.ainvoke(
                    {
                        "messages": [{"role": "user", "content": user_message}]
                    }
                )
                if result.get("messages"):
                    final_message = result["messages"][-1]
                    content = final_message.content
                    parsed = self._parse_recipe(content, meal_type=meal)
                    if parsed:
                        return parsed
                    logger.warning("未解析出有效 JSON，改为直接模型生成。")
            except Exception as exc:  # pragma: no cover - runtime guardrail
                logger.error("生成菜谱失败: %s", exc)

        return self._generate_local_recipe(meal, season, people, dislikes)


    def _generate_recipe_sync(self, meal_type: Optional[str] = None, people: Optional[int] = None, dislikes: Optional[str] = None) -> Dict:
        """同步包装：生成菜谱 JSON，参数可选 meal_type/people/dislikes。"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.generate_recipe(people=people, dislikes=dislikes, meal_type=meal_type))
        finally:
            loop.close()



