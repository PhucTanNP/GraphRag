"""Answer builder — formats structured data into human-readable responses.

Sử dụng ``TemplateManager`` để render các template .md theo topic.
"""
from __future__ import annotations

import logging

from app.services import LLMClient
from app.response.normalizer import normalize_data
from app.templates import get_template_manager

logger = logging.getLogger(__name__)


class AnswerGenerator:
    def __init__(self):
        self.llm = LLMClient(model_name="models/gemini-3.1-flash-lite-preview")
        self.templates = get_template_manager()

    def generate(self, query, data, plan=None):
        data = normalize_data(data)
        plan_type = plan.get("type") if plan else None

        # ── rule-based formatters ────────────────────────────────────────
        fmt = None
        if plan_type in ("SPEED", "LOAD", "PRICE", "PRESSURE", "SPECS"):
            fmt = self._format_simple(plan_type, data)
        elif plan_type in ("MAX_SPEED", "MAX_LOAD", "MAX_PRICE"):
            fmt = self._format_max(plan_type, data)
        elif plan_type == "COMPARE":
            fmt = self._format_compare(data)
        elif plan_type in ("BRAND", "TUBE"):
            fmt = self._format_item_list(plan_type, data)
        elif plan_type == "MULTI_HOP":
            fmt = self._format_list(data)
        elif plan_type == "NO_MATCH":
            return self._format_no_match(query)

        if fmt:
            return fmt

        logger.info("[AnswerGenerator] No rule-based formatter matched (type=%s) → dùng LLM", plan_type)
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

        # Xử lý các trường có thể None trước khi vào template
        safe_data = dict(data)
        if "price" in safe_data:
            p = safe_data.get("price")
            safe_data["price_display"] = f"{p:,.0f} VNĐ" if p is not None else "Liên hệ"
        if "max_speed" in safe_data:
            v = safe_data.get("max_speed")
            safe_data["max_speed_display"] = f"{v} km/h" if v is not None else "Không có dữ liệu"
        if "max_load" in safe_data:
            v = safe_data.get("max_load")
            safe_data["max_load_display"] = f"{v} kg" if v is not None else "Không có dữ liệu"
        if "pressure" in safe_data:
            v = safe_data.get("pressure")
            safe_data["pressure_display"] = f"{v} psi" if v is not None else "Không có dữ liệu"

        result = self.templates.render(topic, **{
            "size": size,
            "brand": brand,
            "brand_str": brand_str,
            **safe_data,
        })
        if result:
            return result
        logger.info("[AnswerGenerator] Template %s returned None → dùng LLM", topic)
        return self._llm_generate(plan_type, data, {"type": plan_type})

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
        if isinstance(data, dict):
            data = [data]

        lines = []
        for idx, item in enumerate(data, 1):
            size = item.get("size", "N/A")
            brand = item.get("brand", "")
            brand_str = f" ({brand})" if brand else ""
            specs = []
            if item.get("max_speed") is not None:
                specs.append(f"⚡{item['max_speed']} km/h")
            if item.get("max_load") is not None:
                specs.append(f"💪{item['max_load']} kg")
            if item.get("price") is not None:
                specs.append(f"💰{item['price']:,.0f} VNĐ")
            if item.get("pressure") is not None:
                specs.append(f"🔧{item['pressure']} psi")
            info = " | ".join(specs)
            lines.append(f"{idx}. **Lốp {size}{brand_str}**: {info}")

        items_str = "\n".join(lines)
        return self.templates.render("compare", items=items_str) or "\n".join(lines)

    def _format_item_list(self, plan_type, data):
        """Generic formatter cho BRAND và TUBE — render list items."""
        if not data:
            return self._format_no_match("")
        if isinstance(data, dict):
            data = [data]

        lines = []
        topic = plan_type.lower()

        for idx, item in enumerate(data, 1):
            size = item.get("size", "N/A")
            brand = item.get("brand", "")
            brand_str = f" ({brand})" if brand else ""
            specs = []
            if item.get("max_speed") is not None:
                specs.append(f"⚡{item['max_speed']} km/h")
            if item.get("max_load") is not None:
                specs.append(f"💪{item['max_load']} kg")
            if item.get("price") is not None:
                p = item['price']
                specs.append(f"💰{p:,.0f} VNĐ" if p else "💰Liên hệ")
            if item.get("pressure") is not None:
                specs.append(f"🔧{item['pressure']} psi")
            info = " | ".join(specs) if specs else ""
            lines.append(f"{idx}. **Lốp {size}{brand_str}**" + (f": {info}" if info else ""))

        items_str = "\n".join(lines)

        # Render với template riêng
        first_size = data[0].get("size", "") if data else ""
        first_brand = data[0].get("brand", "") if data else ""
        result = self.templates.render(topic,
            items=items_str,
            brand=first_brand,
            size_str=f" ({first_size})" if first_size else "",
        )
        if result:
            return result
        return f"📋 **Kết quả:**\n{items_str}"

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
        logger.info("[AnswerGenerator] 🤖 _llm_generate — query=%s, type=%s",
                     str(query)[:50], plan.get("type", "N/A") if plan else "N/A")
        try:
            response = self.llm.chat(prompt)
            logger.info("[AnswerGenerator] 🤖 LLM response OK — length=%s", len(response or ""))
            return response
        except Exception as e:
            logger.error("[AnswerGenerator] 🤖 LLM error: %s", e)
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
