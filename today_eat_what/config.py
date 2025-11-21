import os
from dataclasses import dataclass
from typing import Dict, Optional

_deepseek_base = os.environ.get("DEEPSEEK_BASE_URL", "")
_deepseek_endpoint = os.environ.get("DEEPSEEK_ENDPOINT", "")
if _deepseek_base and not _deepseek_endpoint:
    # OpenAI 兼容路径
    _deepseek_endpoint = _deepseek_base.rstrip("/") + "/chat/completions"
DEEPSEEK_MODEL_DEFAULT = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

_longcat_base = os.environ.get("LONGCAT_BASE_URL", "")
_longcat_endpoint = os.environ.get("LONGCAT_ENDPOINT", "")
if _longcat_base and not _longcat_endpoint:
    # 如果给的是 /openai 或 /v1 等基地址，自动拼上 chat/completions
    base_sanitized = _longcat_base.rstrip("/")
    if not base_sanitized.endswith("chat/completions"):
        if base_sanitized.endswith(("openai", "v1")):
            _longcat_endpoint = base_sanitized + "/chat/completions"
        else:
            _longcat_endpoint = base_sanitized
LONGCAT_MODEL_DEFAULT = os.environ.get("LONGCAT_MODEL", "LongCat-Flash-Chat")

SILICONFLOW_BASE_URL = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
QWEN_MODEL_DEFAULT = os.environ.get("QWEN_MODEL", os.environ.get("SILICONFLOW_MODEL", "Qwen/Qwen3-8B"))
QWEN_ENDPOINT_DEFAULT = os.environ.get("QWEN_ENDPOINT") or f"{SILICONFLOW_BASE_URL.rstrip('/')}/chat/completions"

# Model endpoints and nominal costs; endpoints are meant to be overridden via env.
MODEL_CONFIG: Dict[str, Dict[str, object]] = {
    "qwen": {"endpoint": QWEN_ENDPOINT_DEFAULT, "model": QWEN_MODEL_DEFAULT, "cost_per_call": 0.01},
    "deepseek": {"endpoint": _deepseek_endpoint, "model": DEEPSEEK_MODEL_DEFAULT, "cost_per_call": 0.02},
    "longcat": {
        "endpoint": _longcat_endpoint,
        "model": LONGCAT_MODEL_DEFAULT,
        "cost_per_call": 0.005,
    },
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
        qwen=os.environ.get("QWEN_API_KEY") or os.environ.get("SILICONFLOW_API_KEY"),
        deepseek=os.environ.get("DEEPSEEK_API_KEY"),
        longcat=os.environ.get("LONGCAT_API_KEY"),
        doubao=os.environ.get("DOUBAO_API_KEY"),
        gpt4=os.environ.get("GPT4_API_KEY"),
    )
