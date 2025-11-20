from langchain_core.tools import tool

from ..clients import CostTracker, ModelClient
from ..models import AuditResult


class AuditAgent:
    def __init__(self, longcat_client: ModelClient, cost: CostTracker) -> None:
        self.longcat = longcat_client
        self.cost = cost

    @tool("audit_content", return_direct=True)
    def audit_content_tool(self, content: str) -> dict:
        audit_prompt = "审查以下内容是否包含敏感或违规信息，只返回JSON: {\"ok\": true/false, \"reasons\": []}\n" + content
        self.cost.add("longcat")
        resp = self.longcat.invoke(audit_prompt)
        ok = bool(resp.get("ok", True))
        reasons = resp.get("reasons") if isinstance(resp, dict) else None
        return AuditResult(ok=ok, reasons=reasons).model_dump()
