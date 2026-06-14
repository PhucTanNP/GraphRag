"""LLM Fallback — dùng Gemini khi semantic matching confidence thấp.

Hai chức năng:
  1. ``normalize(query)`` → (intent, normalized_query) — chuẩn hoá câu hỏi mơ hồ
  2. ``answer(query)`` → direct answer khi Cypher/Neo4j không trả về kết quả
"""

import json
import logging
from app.services import LLMClient

logger = logging.getLogger(__name__)

# ─── System prompt cho LLM fallback ─────────────────────────────────────
SYSTEM_PROMPT = """Bạn là chuyên viên tư vấn lốp xe máy.

Hướng dẫn:
1. Trả lời bằng tiếng Việt, thân thiện, chuyên nghiệp
2. Nếu khách hỏi về thông số kỹ thuật, hãy yêu cầu họ cung cấp kích thước lốp
3. Nếu khách hỏi về giá, yêu cầu họ cho biết kích thước hoặc thương hiệu
4. Nếu không chắc chắn, đề nghị kết nối với nhân viên hỗ trợ
5. Chỉ tư vấn về lốp xe máy và dịch vụ liên quan
"""

NORMALIZE_PROMPT = """Bạn là hệ thống phân loại câu hỏi về lốp xe máy.
Phân tích câu hỏi sau và chọn intent phù hợp nhất từ danh sách.

Các intent có sẵn:
  THONG_SO   — hỏi thông số kỹ thuật chung của lốp
  SPEED      — hỏi tốc độ tối đa
  LOAD       — hỏi tải trọng
  PRICE      — hỏi giá cả
  COMPARE    — so sánh hai lốp
  PRESSURE   — hỏi áp suất / bơm hơi
  BRAND      — hỏi thương hiệu / so sánh hãng
  MAX_LOAD   — lốp chịu tải cao nhất
  MAX_SPEED  — lốp nhanh nhất
  MAX_PRICE  — lốp đắt nhất
  DRAINAGE   — lốp thoát nước / đi mưa
  DURABILITY — lốp bền nhất
  TUBE       — có săm / không săm
  SERVICE    — dịch vụ thay lốp / lắp đặt

Trả về JSON (không markdown, không ```):
  {{"intent": "TÊN_INTENT", "normalized": "câu hỏi đã chuẩn hoá có size/brand nếu có"}}

Ví dụ:
  "cho tôi hỏi giá" → {{"intent": "PRICE", "normalized": "giá lốp bao nhiêu"}}
  "lốp 120/70-17 chạy nhanh không" → {{"intent": "SPEED", "normalized": "lốp 120/70-17 tốc độ bao nhiêu"}}
"""


class LLMFallback:
    """Handle low-confidence queries: normalise or answer directly.

    Usage:
        fallback = LLMFallback()
        intent, normalized = fallback.normalize("cho tôi hỏi giá")
        answer = fallback.answer("lốp 120/70-17 giá bao nhiêu?")
    """

    def __init__(self):
        self.llm = LLMClient(model_name="models/gemini-3.1-flash-lite-preview")

    # ── Normalize ──────────────────────────────────────────────────────

    def normalize(self, query: str) -> tuple[str | None, str]:
        """Normalise vague query into (intent, normalized_query).

        Args:
            query: Raw user query.

        Returns:
            Tuple of (intent_str or None, normalized_query_string).
            Falls back to (None, original_query) on error.
        """
        if not query or not query.strip():
            return None, query

        prompt = f"""{NORMALIZE_PROMPT}

Câu hỏi: {query}
"""
        try:
            raw = self.llm.chat(prompt)
            if raw:
                raw = raw.strip()
                # Strip markdown code fence if present
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                data = json.loads(raw)
                intent = data.get("intent") or None
                normalized = data.get("normalized", query)
                logger.info("[LLMFallback] normalize → intent=%s, normalized=%s", intent, normalized)
                return intent, normalized
        except Exception as e:
            logger.warning("[LLMFallback] normalize failed: %s", e)

        return None, query

    # ── Direct answer ──────────────────────────────────────────────────

    def answer(self, query: str) -> str:
        """Generate direct answer using LLM (fallback when Neo4j has no data)."""
        if not query or not query.strip():
            return self._default_message()

        prompt = f"""{SYSTEM_PROMPT}

Khách hàng hỏi: {query}

Hãy trả lời thân thiện, hữu ích.
Nếu cần thêm thông tin, hãy hỏi lại.
Nếu không thể trả lời, đề nghị kết nối nhân viên hỗ trợ.
"""
        try:
            response = self.llm.chat(prompt)
            if response and response.strip():
                return response.strip()
        except Exception as e:
            logger.error("[LLMFallback] Gemini error: %s", e)

        return self._default_message()

    def answer_with_context(self, query: str, context: dict = None) -> str:
        """Generate answer with extra context (intent, size, brand)."""
        ctx = ""
        if context:
            ctx = "\nThông tin thêm:\n" + "\n".join(f"- {k}: {v}" for k, v in context.items() if v)

        prompt = f"""{SYSTEM_PROMPT}{ctx}

Khách hàng hỏi: {query}
"""
        try:
            response = self.llm.chat(prompt)
            if response and response.strip():
                return response.strip()
        except Exception as e:
            logger.error("[LLMFallback] Gemini error (ctx): %s", e)

        return self._default_message()

    # ── Default ────────────────────────────────────────────────────────

    @staticmethod
    def _default_message() -> str:
        return """❌ **Mình rất tiếc, chưa thể trả lời câu hỏi của bạn ngay bây giờ.**

💡 **Bạn có thể thử:**
- Hỏi rõ ràng hơn với kích thước lốp (VD: "Lốp 120/70-17 giá bao nhiêu?")
- Gọi hotline: **1900 XXXX** để được tư vấn trực tiếp
- Để lại số điện thoại, nhân viên sẽ gọi lại cho bạn

📞 **Cần hỗ trợ gấp?** Hãy yêu cầu kết nối với nhân viên hỗ trợ!"""
