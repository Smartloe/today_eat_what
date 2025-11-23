import os
from dataclasses import dataclass
from typing import Dict, Optional


def _env_first(*keys: str) -> Optional[str]:
    for key in keys:
        value = os.environ.get(key)
        if value:
            return value
    return None


_deepseek_base = os.environ.get("DEEPSEEK_BASE_URL", "")
_deepseek_endpoint = os.environ.get("DEEPSEEK_ENDPOINT", "")
if _deepseek_base and not _deepseek_endpoint:
    # OpenAI 兼容路径
    _deepseek_endpoint = _deepseek_base.rstrip("/") + "/chat/completions"
DEEPSEEK_BASE_URL = _deepseek_base
DEEPSEEK_MODEL_DEFAULT = _env_first("DEEPSEEK_MODEL", "DEEPSEEK__MODEL")

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
LONGCAT_MODEL_DEFAULT = _env_first("LONGCAT_MODEL")

QWEN_BASE_URL = _env_first("QWEN_BASE_URL", "Qwen_BASE_URL", "Qwen_BASE_URL") or "https://api.siliconflow.cn/v1"
Qwen_BASE_URL = QWEN_BASE_URL  # backward compat export
QWEN_MODEL_DEFAULT = _env_first("QWEN_MODEL", "Qwen_MODEL", "Qwen_MODEL")
QWEN_ENDPOINT_DEFAULT = os.environ.get("QWEN_ENDPOINT") or f"{QWEN_BASE_URL.rstrip('/')}/chat/completions"
DOUBAO_BASE_URL = os.environ.get("DOUBAO_BASE_URL", "")
GLM_BASE_URL = os.environ.get("GLM_BASE_URL", "")
GLM_MODEL_DEFAULT = _env_first("GLM_MODEL")

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
    "glm": {
        "endpoint": os.environ.get("GLM_ENDPOINT")
        or f"{GLM_BASE_URL.rstrip('/')}/chat/completions" if GLM_BASE_URL else "",
        "model": GLM_MODEL_DEFAULT,
        "cost_per_call": 0.05,
    },
}

XHS_MCP_URL = os.environ.get("XIAOHONGSHU_MCP_URL", "")
HOWTOCOOK_MCP_URL = os.environ.get("HOWTOCOOK_MCP_URL", "")  # optional override


@dataclass
class ApiKeys:
    qwen: Optional[str]
    deepseek: Optional[str]
    longcat: Optional[str]
    doubao: Optional[str]
    glm: Optional[str]


def load_api_keys() -> ApiKeys:
    return ApiKeys(
        qwen=_env_first("QWEN_API_KEY", "Qwen_API_KEY", "Qwen_API_KEY"),
        deepseek=os.environ.get("DEEPSEEK_API_KEY"),
        longcat=os.environ.get("LONGCAT_API_KEY"),
        doubao=os.environ.get("DOUBAO_API_KEY"),
        glm=os.environ.get("GLM_API_KEY"),
    )
