from typing import List

from langchain_core.tools import tool

from ..clients import CostTracker, ModelClient
from ..models import PublishResult
from ..services import publish_to_xiaohongshu


class PublishAgent:
    def __init__(self, gpt4_client: ModelClient, cost: CostTracker) -> None:
        self.gpt4 = gpt4_client
        self.cost = cost

    @tool("publish_to_xhs", return_direct=True)
    def publish_tool(self, content: str, images: List[str]) -> dict:
        # reuse publish helper for MCP + LLM flow
        result = publish_to_xiaohongshu(content, images, self.gpt4, self.cost)
        return PublishResult(**result.model_dump()).model_dump()
