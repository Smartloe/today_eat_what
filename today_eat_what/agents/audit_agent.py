import sys
from pathlib import Path

from langchain_core.tools import tool

# Ensure package import works when running as a script
ROOT = Path(__file__).resolve().parents[1].parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from today_eat_what.clients import CostTracker, ModelClient
from today_eat_what.models import AuditResult
from today_eat_what.config import MODEL_CONFIG, load_api_keys
from today_eat_what.utils import load_dotenv, setup_logging


class AuditAgent:
    def __init__(self, longcat_client: ModelClient, cost: CostTracker) -> None:
        self.longcat = longcat_client
        self.cost = cost
        self.audit_content_tool = tool("audit_content", return_direct=True)(self._audit_content)

    def _audit_content(self, content: str) -> dict:
        """调用 LongCat 模型审核文本，返回 ok/reasons/risk_level 结构。"""
        audit_prompt = """
你是内容安全审核员，严格执行中国法律法规、公序良俗和平台规范，对以下文本进行审核。

【必须拦截】
- 脏话、人身攻击、辱骂（如：你妈、去死、傻逼等）
- 淫秽色情、性暗示、性器官描述
- 暴力恐怖、自杀自残、血腥
- 非法/违法活动（毒品、赌博、武器、犯罪）
- 政治敏感（国家领导人、分裂、极端言论等）
- 歧视仇恨（种族/民族/宗教/性别/地域），网络霸凌
- 隐私泄露（身份证、电话、住址等个人信息）
- 未成年人不宜内容、错误健康/医疗建议

【处理标准】
- 任一违规点 => ok=false，简明列出原因，risk_level=高/中。
- 边界/潜在风险（轻微脏话、暗示、煽动） => ok=false，risk_level=中，写明风险。
- 无问题 => ok=true，reasons=[]，risk_level=无。

        【输出】
        仅返回 JSON：
        {"ok": true/false, "reasons": ["原因1", ...], "risk_level": "无/低/中/高"}
        不要输出其他文字。

待审核内容：
        """ + content
        self.cost.add("longcat")
        resp = self.longcat.invoke(audit_prompt, extra={"model": MODEL_CONFIG.get("longcat", {}).get("model")}, timeout=10)

        # 尝试解析模型返回的 JSON（兼容 content/choices/message.content）。
        parsed = None
        if isinstance(resp, dict):
            if "ok" in resp:
                parsed = resp
            else:
                content_text = resp.get("content") or resp.get("text") or resp.get("output")
                if not content_text and resp.get("choices"):
                    content_text = resp["choices"][0].get("message", {}).get("content")
                if content_text:
                    try:
                        import json

                        parsed = json.loads(content_text)
                    except Exception:
                        parsed = None

        ok = bool(parsed.get("ok", False)) if parsed else False
        reasons = parsed.get("reasons") if isinstance(parsed, dict) else None
        risk = parsed.get("risk_level") if isinstance(parsed, dict) else None
        return AuditResult(ok=ok, reasons=reasons or [risk] if risk else None).model_dump()
