"""Answer builder — formats structured data into human-readable responses.

Sử dụng ``TemplateManager`` để render các template .md theo topic.
"""
from __future__ import annotations

from app.services import LLMClient
from app.response.normalizer import normalize_data
from app.templates import get_template_manager


class AnswerGenerator:
    def __init__(self):
        self.llm = LLMClient(model_name="models/gemini-3.1-flash-lite-preview")
        self.templates = get_template_manager()

    def generate(self, query, data, plan=None):
        data = normalize_data(data)
        plan_type = plan.get("type") if plan else None

        # ── rule-based formatters ────────────────────────────────────────
        fmt = None
        if plan_type in ("SPEED", "LOAD", "PRICE", "PRESSURE"):
            fmt = self._format_simple(plan_type, data)
        elif plan_type in ("MAX_SPEED", "MAX_LOAD", "MAX_PRICE"):
            fmt = self._format_max(plan_type, data)
        elif plan_type == "COMPARE":
            fmt = self._format_compare(data)
        elif plan_type == "MULTI_HOP":
            fmt = self._format_list(data)
        elif plan_type == "NO_MATCH":
            return self._format_no_match(query)

        if fmt:
            return fmt

        return self._llm_generate(query, data, plan)

    # ── private formatters ───────────────────────────────────────────────

    def _format_no_match(self, query):
        return self.templates.render("no_match") or "Xin lỗi, tôi chưa hiểu yêu cầu của bạn."

    def _format_simple(self, plan_type, data):
        if isinstance(data, list):
            data = data[0] if data else {}
        if not isinstance(data, dict) or not data:
            return self._format_no_match("")

        size = data.get("size", "kích thước này")
        brand = data.get("brand", "")
        brand_str = f" ({brand})" if brand else ""
        topic = plan_type.lower()

        return self.templates.render(topic, **{
            "size": size,
            "brand": brand,
            "brand_str": brand_str,
            **data,
        }) or self._llm_generate(plan_type, data, {"type": plan_type})

    def _format_max(self, plan_type, data):
        if not data:
            return self._format_no_match("")
        if isinstance(data, dict):
            data = [data]

        lines = []
        value_key = {"MAX_SPEED": "max_speed", "MAX_LOAD": "max_load", "MAX_PRICE": "price"}.get(plan_type, "")
        unit = {"MAX_SPEED": "km/h", "MAX_LOAD": "kg", "MAX_PRICE": " VNĐ"}.get(plan_type, "")
        emoji = {"MAX_SPEED": "⚡", "MAX_LOAD": "💪", "MAX_PRICE": "💰"}.get(plan_type, "")

        for idx, item in enumerate(data, 1):
            size = item.get("size", "N/A")
            brand = item.get("brand", "")
            brand_str = f" ({brand})" if brand else ""
            value = item.get(value_key)
            if value is not None:
                if plan_type == "MAX_PRICE":
                    lines.append(f"{idx}. **Lốp {size}{brand_str}**: {emoji} **{value:,.0f}{unit}**")
                else:
                    lines.append(f"{idx}. **Lốp {size}{brand_str}**: {emoji} **{value} {unit}**")

        if not lines:
            return self._format_no_match("")

        items_str = "\n".join(lines)
        return self.templates.render("max", items=items_str) or "\n".join(lines)

    def _format_compare(self, data):
        if not data:
            return self._format_no_match("")
        return self._llm_generate("compare", data, {"type": "COMPARE"})

    def _format_list(self, data):
        if not data:
            return self._format_no_match("")
        lines = []
        for i, item in enumerate(data, 1):
            size = item.get("size", "N/A")
            brand = item.get("brand", "")
            info = " | ".join(f"{k}: {v}" for k, v in item.items() if k not in ("size", "brand"))
            lines.append(f"{i}. **{size}** {f'({brand})' if brand else ''}: {info}")

        items_str = "\n".join(lines)
        return self.templates.render("list", items=items_str) or "\n".join(
            ["📋 **Danh sách kết quả:**\n", items_str, "\n\n💡 Bạn muốn biết thêm chi tiết về mẫu nào?"]
        )

    # ── LLM fallback ─────────────────────────────────────────────────────

    def _llm_generate(self, query, data, plan):
        prompt = self._build_prompt(query, data, plan)
        try:
            return self.llm.chat(prompt)
        except Exception:
            return "Xin lỗi, tôi không thể tạo câu trả lời ngay lúc này."

    def _build_prompt(self, query, data, plan):
        import json
        plan_type = plan.get("type", "UNKNOWN") if plan else "UNKNOWN"
        data_str = json.dumps(data, ensure_ascii=False, indent=2) if data else "Không có dữ liệu"
        return (
            f"Bạn là chuyên gia tư vấn lốp xe. Hãy trả lời câu hỏi sau bằng tiếng Việt.\n\n"
            f"Câu hỏi: {query}\n"
            f"Loại truy vấn: {plan_type}\n"
            f"Dữ liệu: {data_str}\n\n"
            "Hãy đưa ra câu trả lời thân thiện, có cấu trúc rõ ràng (dùng **bold** cho số liệu quan trọng)."
        )
