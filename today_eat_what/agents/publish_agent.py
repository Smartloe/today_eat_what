import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

# Ensure package import works when running as a script
ROOT = Path(__file__).resolve().parents[1].parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from today_eat_what.clients import CostTracker, ModelClient
from today_eat_what.config import XHS_MCP_URL
from today_eat_what.models import PublishResult

logger = logging.getLogger(__name__)


class PublishAgent:
    def __init__(self, gpt4_client: ModelClient, cost: CostTracker, mcp_url: Optional[str] = None) -> None:
        self.gpt4 = gpt4_client
        self.cost = cost
        self.publish_tool = tool("publish_to_xhs", return_direct=True)(self._publish)
        self._agent = None
        self._mcp_client: Optional[MultiServerMCPClient] = None
        self._mcp_publish_tool = None
        self._mcp_tool_checked = False
        self._mcp_server_name = (
            os.environ.get("XHS_MCP_SERVER")
            or os.environ.get("XIAOHONGSHU_MCP_SERVER")
            or "xiaohongshu-mcp"
        )
        self._mcp_error: Optional[str] = None
        self._mcp_url = (
            mcp_url
            or os.environ.get("XIAOHONGSHU_MCP_URL")
            or os.environ.get("XHS_MCP_URL")
            or os.environ.get("XHS_MCP_BASE_URL")
            or XHS_MCP_URL
            or "http://120.55.6.39:18060/mcp"
        )

    def _init_mcp_client(self) -> Optional[MultiServerMCPClient]:
        if self._mcp_client or not self._mcp_url:
            return self._mcp_client
        try:
            self._mcp_client = MultiServerMCPClient(
                {self._mcp_server_name: {"transport": "streamable_http", "url": self._mcp_url}}
            )
            logger.info("使用的小红书 MCP URL: %s", self._mcp_url)
            return self._mcp_client
        except Exception as exc:  # pragma: no cover - MCP 依赖环境
            self._mcp_error = f"初始化小红书 MCP 失败：{exc}"
            logger.error(self._mcp_error)
            raise

    async def _load_mcp_tool(self) -> Optional[Any]:
        client = self._init_mcp_client()
        if not client:
            raise RuntimeError("小红书 MCP 客户端不可用。")
        try:
            tools = await asyncio.wait_for(client.get_tools(server_name=self._mcp_server_name), timeout=20)
            if not tools:
                raise RuntimeError("小红书 MCP 未返回任何工具，请检查服务。")
            # 优先精确 publish_content，其次 publish_with_video，再兜底包含 publish 的工具。
            name_map = {t.name: t for t in tools}
            if "publish_content" in name_map:
                return name_map["publish_content"]
            if "publish_with_video" in name_map:
                return name_map["publish_with_video"]
            preferred = next((t for t in tools if "publish" in t.name.lower()), None)
            return preferred or tools[0]
        except Exception as exc:  # pragma: no cover - 外部 MCP
            self._mcp_error = f"拉取小红书 MCP 工具失败：{exc}"
            logger.error(self._mcp_error)
            raise

    def _ensure_mcp_tool(self) -> Optional[Any]:
        if self._mcp_tool_checked:
            return self._mcp_publish_tool
        self._mcp_tool_checked = True
        if not self._mcp_url:
            raise RuntimeError(
                "未配置 XIAOHONGSHU_MCP_URL（或 --mcp-url），无法发布。示例：http://localhost:18060/mcp"
            )
        loop = asyncio.new_event_loop()
        try:
            self._mcp_publish_tool = loop.run_until_complete(self._load_mcp_tool())
        finally:
            loop.close()
        return self._mcp_publish_tool

    def _publish_via_mcp(self, content: str, images: List[str], tags: Optional[List[str]] = None) -> Optional[PublishResult]:
        publish_tool = self._ensure_mcp_tool()
        if not publish_tool:
            msg = self._mcp_error or "小红书 MCP 工具不可用，请检查服务。"
            raise RuntimeError(msg)

        tags = tags or []
        if "今天吃什么呢" not in tags:
            tags = ["今天吃什么呢", *tags]

        args: dict[str, Any] = {}
        schema = getattr(publish_tool, "args_schema", None)
        schema_fields = getattr(schema, "__fields__", {}) if schema else {}
        if "content" in schema_fields:
            args["content"] = content
        if "images" in schema_fields:
            args["images"] = images
        if "tags" in schema_fields:
            args["tags"] = tags
        if not args:
            args = {"content": content, "images": images, "tags": tags}
        # 一些 MCP 工具要求 title 字段，尝试自动补全。
        if "title" not in args:
            args["title"] = self._infer_title(content)
        logger.info("调用 MCP 工具 %s，传参 keys=%s", getattr(publish_tool, "name", "unknown"), list(args.keys()))

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(asyncio.wait_for(publish_tool.ainvoke(args), timeout=30))
        except Exception as exc:  # pragma: no cover - 外部 MCP
            self._mcp_error = f"调用小红书 MCP 发布失败：{exc}"
            logger.error(self._mcp_error)
            raise
        finally:
            loop.close()

        if isinstance(result, PublishResult):
            return result
        if isinstance(result, dict):
            return PublishResult(
                success=bool(result.get("success", True)),
                post_id=result.get("post_id") or result.get("id"),
                detail=result,
            )
        return PublishResult(success=True, post_id=str(result), detail={"output": result})

    def _publish(self, content: str, images: List[str], tags: Optional[List[str]] = None) -> dict:
        """仅通过 MCP 发布小红书，如失败则抛出异常。"""
        mcp_result = self._publish_via_mcp(content, images, tags=tags)
        if not mcp_result:
            raise RuntimeError("小红书 MCP 发布返回空结果。")
        if not mcp_result.success:
            raise RuntimeError(f"小红书发布失败：{mcp_result.detail or mcp_result.post_id}")
        return mcp_result.model_dump()

    @staticmethod
    def _infer_title(content: str) -> str:
        """根据正文简单提取标题，必要时截断。"""
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        title = lines[0] if lines else content.strip()
        if len(title) > 30:
            title = title[:27] + "..."
        return title or "自动发布"

    def get_agent(self):
        """使用 LangChain create_agent 包装为发布智能体。"""
        if self._agent:
            return self._agent
        llm = ChatOpenAI(
            model=os.environ.get("GLM_MODEL") or "THUDM/GLM-4-9B-0414",
            api_key=os.environ.get("GLM_API_KEY"),
            base_url=os.environ.get("GLM_BASE_URL"),
            temperature=0.2,
        )
        system_prompt = "你是小红书发布助手，负责调用发布工具并汇总发布结果。"
        tools = [self.publish_tool]
        mcp_tool = self._ensure_mcp_tool()
        if mcp_tool:
            tools.insert(0, mcp_tool)
        self._agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)
        return self._agent
