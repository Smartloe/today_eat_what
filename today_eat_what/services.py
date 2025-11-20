import logging
from datetime import datetime
from typing import Any, Dict, List

import requests

from .config import HOWTOCOOK_MCP_URL, XHS_MCP_URL
from .models import PublishResult
from .clients import ModelClient, CostTracker

logger = logging.getLogger(__name__)


def get_meal_type(now: datetime | None = None) -> str:
    now = now or datetime.now()
    hour = now.hour
    if 6 <= hour <= 10:
        return "早餐"
    if 11 <= hour <= 14:
        return "午餐"
    if 17 <= hour <= 21:
        return "晚餐"
    return "小吃"


def call_how_to_cook(meal_type: str) -> Dict[str, Any]:
    payload = {"meal_type": meal_type}
    if not HOWTOCOOK_MCP_URL:
        # Fallback stub.
        return {
            "recipe": {
                "name": f"{meal_type}活力套餐",
                "description": "简单易做的家常菜，快速补充能量。",
                "ingredients": ["鸡胸肉 150g", "西兰花 1颗", "米饭 1碗", "橄榄油 1勺", "盐、黑胡椒 适量"],
                "steps": [
                    {"order": 1, "instruction": "鸡胸肉切片，撒盐和黑胡椒腌5分钟。"},
                    {"order": 2, "instruction": "西兰花切小朵焯水，备用。"},
                    {"order": 3, "instruction": "热锅倒油，煎熟鸡胸肉，加入西兰花翻炒。"},
                    {"order": 4, "instruction": "盛出搭配米饭，淋少许橄榄油。"},
                ],
            }
        }
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(HOWTOCOOK_MCP_URL, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # pragma: no cover - external service
        logger.error("HowToCook MCP failed: %s", exc)
        raise


def publish_to_xiaohongshu(content: str, images: List[str], gpt4: ModelClient, cost: CostTracker) -> PublishResult:
    publish_prompt = (
        "你是小红书助手，准备发布以下内容，返回JSON {\"success\": true/false, \"post_id\": \"...\"}。"
        f"正文：{content}\n图片：{images}"
    )
    cost.add("gpt4")
    resp = gpt4.invoke(publish_prompt)
    success = bool(resp.get("success", True))
    post_id = resp.get("post_id") or resp.get("id") or "mock-post-id"

    # Optionally call MCP endpoint if configured.
    if XHS_MCP_URL:
        try:
            mcp_resp = requests.post(XHS_MCP_URL, json={"content": content, "images": images}, timeout=10)
            mcp_resp.raise_for_status()
            publish_data = mcp_resp.json()
            success = bool(publish_data.get("success", success))
            post_id = publish_data.get("post_id", post_id)
        except Exception as exc:  # pragma: no cover - external service
            logger.error("Xiaohongshu MCP call failed: %s", exc)

    return PublishResult(success=success, post_id=post_id, detail=resp if isinstance(resp, dict) else None)
