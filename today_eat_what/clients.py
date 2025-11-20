import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from .config import MODEL_CONFIG
from .utils import run_with_retry, run_with_timeout

logger = logging.getLogger(__name__)


@dataclass
class CostTracker:
    total_cost: float = 0.0
    breakdown: Dict[str, float] = field(default_factory=dict)

    def add(self, vendor: str) -> None:
        cost = float(MODEL_CONFIG.get(vendor, {}).get("cost_per_call", 0))
        self.total_cost += cost
        self.breakdown[vendor] = self.breakdown.get(vendor, 0) + cost


class ModelClient:
    def __init__(self, vendor: str, api_key: Optional[str], default_model: Optional[str] = None) -> None:
        self.vendor = vendor
        self.api_key = api_key
        self.endpoint = str(MODEL_CONFIG.get(vendor, {}).get("endpoint", "")).rstrip("/")
        self.default_model = default_model or MODEL_CONFIG.get(vendor, {}).get("model")
        if not self.endpoint:
            logger.warning("Endpoint for %s not configured; calls will return mock data.", vendor)

    def _post_json(self, payload: Dict[str, Any], timeout: float = 10.0) -> Dict[str, Any]:
        if not self.endpoint:
            return {"mock": True, "payload": payload, "vendor": self.vendor}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        def do_request() -> Dict[str, Any]:
            resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()

        return run_with_retry(lambda: run_with_timeout(do_request, timeout))

    def invoke(self, prompt: str, extra: Optional[Dict[str, Any]] = None, timeout: float = 10.0) -> Dict[str, Any]:
        payload = {"prompt": prompt}
        if extra:
            payload.update(extra)
        try:
            return self._post_json(payload, timeout=timeout)
        except Exception as exc:
            logger.error("Model %s call failed: %s", self.vendor, exc)
            raise

    def invoke_chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"messages": messages}
        if model or self.default_model:
            payload["model"] = model or self.default_model
        if extra:
            payload.update(extra)
        try:
            return self._post_json(payload, timeout=timeout)
        except Exception as exc:
            logger.error("Model %s chat call failed: %s", self.vendor, exc)
            raise
