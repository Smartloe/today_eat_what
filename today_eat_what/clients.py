import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

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
        if not self.endpoint:
            base = os.environ.get(f"{vendor.upper()}_BASE_URL", "")
            if base:
                self.endpoint = base.rstrip("/") + "/chat/completions"
        self.default_model = default_model or MODEL_CONFIG.get(vendor, {}).get("model")
        if not self.endpoint:
            logger.warning("Endpoint for %s not configured; calls will return mock data.", vendor)
        self.base_url = self._derive_base_url(self.endpoint)
        self._chat = self._init_chat_llm()

    @staticmethod
    def _derive_base_url(endpoint: str) -> str:
        if not endpoint:
            return ""
        for suffix in ("/chat/completions", "/v1/chat/completions"):
            if endpoint.endswith(suffix):
                return endpoint[: -len(suffix)].rstrip("/")
        return endpoint

    def _init_chat_llm(self) -> Optional[ChatOpenAI]:
        """Prefer ChatOpenAI for all OpenAI兼容端点，缺失时回退为 HTTP 请求。"""
        if not self.base_url or not self.api_key:
            return None
        try:
            return ChatOpenAI(
                model=self.default_model or "gpt-4o-mini",
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=0.4,
            )
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning("Init ChatOpenAI failed for %s: %s", self.vendor, exc)
            return None

    def _messages_to_langchain(self, messages: List[Dict[str, str]]) -> List:
        converted = []
        for msg in messages:
            role = msg.get("role", "") if isinstance(msg, dict) else ""
            content = msg.get("content") if isinstance(msg, dict) else str(msg)
            if role in ("system",):
                converted.append(SystemMessage(content=content))
            elif role in ("assistant", "ai"):
                converted.append(AIMessage(content=content))
            else:
                converted.append(HumanMessage(content=content))
        return converted

    def _normalize_text_response(self, content: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            parsed = None
        return {"text": content, "output": content, "content": content}

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
        if self._chat:
            model_name = (extra or {}).get("model") if extra else None
            def do_invoke() -> Dict[str, Any]:
                chat_llm = self._chat
                if model_name and model_name != getattr(chat_llm, "model_name", None):
                    chat_llm = ChatOpenAI(
                        model=model_name,
                        api_key=self.api_key,
                        base_url=self.base_url,
                        temperature=getattr(self._chat, "temperature", 0.4),
                    )
                msg = chat_llm.invoke(prompt)
                content = msg.content if hasattr(msg, "content") else str(msg)
                return self._normalize_text_response(content)

            return run_with_retry(lambda: run_with_timeout(do_invoke, timeout))

        # If hitting OpenAI-compatible chat, wrap prompt into messages and include model.
        if self.endpoint.endswith("/chat/completions"):
            payload: Dict[str, Any] = {"messages": [{"role": "user", "content": prompt}]}
            if self.default_model:
                payload["model"] = self.default_model
        else:
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
        if self._chat:
            def do_invoke() -> Dict[str, Any]:
                llm = self._chat
                target_model = model or (extra or {}).get("model") or self.default_model
                if target_model and target_model != getattr(llm, "model_name", None):
                    llm = ChatOpenAI(
                        model=target_model,
                        api_key=self.api_key,
                        base_url=self.base_url,
                        temperature=getattr(self._chat, "temperature", 0.4),
                    )
                lc_messages = self._messages_to_langchain(messages)
                msg = llm.invoke(lc_messages)
                content = msg.content if hasattr(msg, "content") else str(msg)
                return self._normalize_text_response(content)

            return run_with_retry(lambda: run_with_timeout(do_invoke, timeout))

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
