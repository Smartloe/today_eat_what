import os
import sys
from pathlib import Path
from typing import List

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Ensure package import works when running as a script
ROOT = Path(__file__).resolve().parents[1].parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from today_eat_what.clients import CostTracker, ModelClient
from today_eat_what.models import PublishResult
from today_eat_what.services import publish_to_xiaohongshu


class PublishAgent:
    def __init__(self, gpt4_client: ModelClient, cost: CostTracker) -> None:
        self.gpt4 = gpt4_client
        self.cost = cost
        self.publish_tool = tool("publish_to_xhs", return_direct=True)(self._publish)
        self._agent = None

    def _publish(self, content: str, images: List[str]) -> dict:
        """发布到小红书（或模拟），返回结果字典。"""
        result = publish_to_xiaohongshu(content, images, self.gpt4, self.cost)
        return PublishResult(**result.model_dump()).model_dump()

    def get_agent(self):
        """使用 LangChain create_agent 包装为发布智能体。"""
        if self._agent:
            return self._agent
        llm = ChatOpenAI(
            model=os.environ.get("GPT4_MODEL", "gpt-4o-mini"),
            api_key=os.environ.get("GPT4_API_KEY"),
            base_url=os.environ.get("GPT4_BASE_URL"),
            temperature=0.2,
        )
        system_prompt = "你是小红书发布助手，负责调用发布工具并汇总发布结果。"
        self._agent = create_agent(model=llm, tools=[self.publish_tool], system_prompt=system_prompt)
        return self._agent


if __name__ == "__main__":
    import argparse
    from today_eat_what.config import MODEL_CONFIG, load_api_keys
    from today_eat_what.utils import load_dotenv, setup_logging

    load_dotenv()
    setup_logging()
    parser = argparse.ArgumentParser(description="Test PublishAgent.")
    parser.add_argument("--use-agent", action="store_true", help="通过 create_agent 调用发布")
    parser.add_argument("--content", default="测试文案")
    parser.add_argument("--images", nargs="*", default=["https://example.com/demo.png"])
    args = parser.parse_args()

    keys = load_api_keys()
    cost = CostTracker()
    gpt4_client = ModelClient("gpt4", keys.gpt4, default_model=MODEL_CONFIG.get("gpt4", {}).get("model"))
    agent = PublishAgent(gpt4_client, cost)

    if args.use_agent:
        lc_agent = agent.get_agent()
        output = lc_agent.invoke({"messages": [{"role": "user", "content": f"请发布：{args.content}"}]})
        if hasattr(output, "messages"):
            print(output.messages[-1].content)
        else:
            print(output)
    else:
        result = agent.publish_tool.invoke({"content": args.content, "images": args.images})
        print(result)
    print("成本估算：", cost.total_cost)
