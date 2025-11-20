import os
from dataclasses import dataclass
from typing import Dict, Optional

# Model endpoints and nominal costs; endpoints are meant to be overridden via env.
MODEL_CONFIG: Dict[str, Dict[str, object]] = {
    "qwen": {"endpoint": os.environ.get("QWEN_ENDPOINT", ""), "cost_per_call": 0.01},
    "deepseek": {"endpoint": os.environ.get("DEEPSEEK_ENDPOINT", ""), "cost_per_call": 0.02},
    "longcat": {"endpoint": os.environ.get("LONGCAT_ENDPOINT", ""), "cost_per_call": 0.005},
    "doubao": {"endpoint": os.environ.get("DOUBAO_ENDPOINT", ""), "cost_per_call": 0.03},
    "gpt4": {"endpoint": os.environ.get("GPT4_ENDPOINT", ""), "cost_per_call": 0.05},
}

XHS_MCP_URL = os.environ.get("XIAOHONGSHU_MCP_URL", "")
HOWTOCOOK_MCP_URL = os.environ.get("HOWTOCOOK_MCP_URL", "")  # optional override


@dataclass
class ApiKeys:
    qwen: Optional[str]
    deepseek: Optional[str]
    longcat: Optional[str]
    doubao: Optional[str]
    gpt4: Optional[str]


def load_api_keys() -> ApiKeys:
    return ApiKeys(
        qwen=os.environ.get("QWEN_API_KEY"),
        deepseek=os.environ.get("DEEPSEEK_API_KEY"),
        longcat=os.environ.get("LONGCAT_API_KEY"),
        doubao=os.environ.get("DOUBAO_API_KEY"),
        gpt4=os.environ.get("GPT4_API_KEY"),
    )
