"""Pipeline Orchestrator V5 — Text2Cypher thuần (không neo4j-graphrag).

Pipeline Flow:
  User Query
  ↓
  Gemini sinh Cypher (prompt: schema + examples + question)
  ↓
  Execute Cypher trên Neo4j
  ↓
  Gemini sinh câu trả lời từ kết quả
  ↓
  Return answer

Usage:
    chatbot = GraphRAGV5()
    answer = chatbot.run("lốp 120/70-17 giá bao nhiêu?")

Interface tương thích V4:
    - run(query) → str
    - reset_context()
    - is_healthy() → bool
"""

import json
import logging
import re
import time
from typing import Any

from rapidfuzz import fuzz

from app.config import LLM_MOCK, LLM_MODELS
from app.services.llm_service import LLMClient
from app.services.neo4j_service import Neo4jService

# ── Import schema, examples, prompts từ file riêng ───────────────────────
from app.pipeline.schema import SCHEMA
from app.pipeline.examples import EXAMPLES
from app.pipeline.prompts import CYPHER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Intent Detection + Templates (question-based)
#  ═══════════════════════════════════════════════════════════════════════════
#  Phát hiện intent từ câu hỏi gốc (tiếng Việt), không phụ thuộc Cypher.
#  Format kết quả theo template, KHÔNG gọi Gemini.

def _fmt_price(v):
    """Format giá: 322491 → 322.491"""
    if v is None:
        return "?"
    try:
        return f"{int(v):,}".replace(",", ".")
    except (ValueError, TypeError):
        return str(v)


def _clean_keys(records: list[dict]) -> list[dict]:
    """Strip alias prefix từ keys: 't.tire_size' → 'tire_size', 'v.name' → 'name'."""
    cleaned = []
    for rec in records:
        new_rec = {}
        for k, v in rec.items():
            clean_k = k.split(".", 1)[-1] if "." in k else k
            new_rec[clean_k] = v
        cleaned.append(new_rec)
    return cleaned


def _has_any_keyword(text: str, keywords: list[str]) -> bool:
    """Kiểm tra text có chứa bất kỳ từ khoá nào không."""
    t = text.lower()
    for kw in keywords:
        if kw in t:
            return True
    return False


# ── Greeting Detection (không gọi LLM) ───────────────────────────────────

GREETING_PATTERNS: list[tuple[list[str], str, int]] = [
    # (patterns, response, min_word_count) — min_word_count: query cần >= từ này
    (
        ["xin chào", "chào bạn", "chào em", "chào buổi", "good morning",
         "good afternoon", "good evening", "có ai không", "ê bạn", "chào tạm biệt"],
        "Dạ chào anh/chị! Em là trợ lý tư vấn lốp xe của DRC Tires. "
        "Em có thể tư vấn lốp, săm theo xe, so sánh hoa lốp, báo giá — anh/chị cần em hỗ trợ gì ạ? 😊",
        1
    ),
    (
        ["hello bạn", "hello em", "hello"],
        "Dạ chào anh/chị! Em là trợ lý tư vấn lốp xe của DRC Tires. "
        "Em có thể tư vấn lốp, săm theo xe, so sánh hoa lốp, báo giá — anh/chị cần em hỗ trợ gì ạ? 😊",
        1
    ),
    (
        ["hi bạn", "hi em", "hi"],
        "Dạ chào anh/chị! Em là trợ lý tư vấn lốp xe của DRC Tires. "
        "Em có thể tư vấn lốp, săm theo xe, so sánh hoa lốp, báo giá — anh/chị cần em hỗ trợ gì ạ? 😊",
        1
    ),
    (
        ["alo"],
        "Dạ em nghe! Em là trợ lý tư vấn lốp xe DRC Tires. Anh/chị cần em hỗ trợ gì ạ? 😊",
        1
    ),
    (
        ["cảm ơn", "cám ơn", "thanks", "tks", "c.ơn", "c.on"],
        "Dạ không có gì ạ! Nếu anh/chị cần tư vấn thêm về lốp, săm hay gì cứ hỏi em nhé. "
        "Chúc anh/chị một ngày tốt lành! 😊",
        1
    ),
    (
        ["tạm biệt", "bái bai", "goodbye", "see you", "see ya"],
        "Dạ chào anh/chị! Cảm ơn đã ghé DRC Tires. Khi nào cần tư vấn lốp cứ nhắn em nhé. "
        "Đi đường cẩn thận ạ! 👋😊",
        1
    ),
    (
        ["bye"],
        "Dạ chào anh/chị! Cảm ơn đã ghé DRC Tires. Khi nào cần tư vấn lốp cứ nhắn em nhé. "
        "Đi đường cẩn thận ạ! 👋😊",
        1
    ),
    (
        ["bạn làm gì", "bán gì", "có những gì", "giới thiệu", "biết gì", "có thể làm"],
        "Dạ, em là trợ lý tư vấn của DRC Tires — chuyên về lốp xe máy. "
        "Em có thể:\n"
        "• Tra lốp, săm theo từng dòng xe (Vision, Air Blade, SH, Exciter...)\n"
        "• So sánh các hoa lốp DRC vs DPLUS\n"
        "• Tư vấn lốp theo nhu cầu (đường trường, thể thao, đi mưa...)\n"
        "• Báo giá chi tiết\n\n"
        "Anh/chị muốn xem lốp cho xe nào ạ?",
        1
    ),
    (
        ["tuyệt", "hay quá", "giỏi", "pro", "nice", "wow"],
        "Dạ em cảm ơn anh/chị! Có gì anh/chị cứ hỏi, em sẵn lòng tư vấn ạ. 😊",
        1
    ),
    (
        ["liên hệ", "hotline", "số điện thoại", "số máy", "gọi điện", "gọi số",
         "địa chỉ", "ở đâu", "thông tin liên hệ", "cách liên hệ", "tư vấn qua điện thoại",
         "alo em", "gọi em", "nhắn em", "zalo"],
        "Dạ, thông tin liên hệ của DRC Tires:\n"
        "📞 **Hotline: 0905 033 776**\n"
        "📍 **409 Trường Chinh, An Khê, Thanh Khê, TP.Đà Nẵng**\n"
        "📧 Email: minhphat.ltd@gmail.com\n"
        "🕐 Thứ 2 - Thứ 7: 07:30 - 17:00\n\n"
        "Anh/chị gọi hotline để được tư vấn và đặt hàng nhanh nhất ạ! 😊",
        1
    ),
    (
        ["đặt hàng", "mua hàng", "mua lốp", "order", "mua ở đâu", "cửa hàng",
         "đại lý", "showroom", "mua chỗ nào"],
        "Dạ, anh/chị có thể:\n"
        "📞 Gọi hotline **0905 033 776** để đặt hàng trực tiếp\n"
        "📍 Hoặc ghé trực tiếp cửa hàng tại **409 Trường Chinh, An Khê, Thanh Khê, TP.Đà Nẵng**\n"
        "📧 Email: minhphat.ltd@gmail.com\n"
        "🕐 Thứ 2 - Thứ 7: 07:30 - 17:00\n\n"
        "Cần tư vấn thêm lốp nào anh/chị cứ hỏi em nhé! 😊",
        1
    ),
]


def _detect_greeting(query: str) -> str | None:
    """Kiểm tra nếu câu hỏi là greeting/thanks/bye — không cần gọi LLM.

    Dùng word-boundary matching để tránh false positive
    (vd: "hi" trong "nhiêu", "bye" trong "bye" tách biệt).

    Returns:
        Câu trả lời có sẵn nếu match, None nếu không phải greeting.
    """
    q = query.lower().strip().strip("?!., ")
    q_words = q.split()
    q_len = len(q_words)

    for patterns, response, min_words in GREETING_PATTERNS:
        for pat in patterns:
            pat_lower = pat.lower()
            pat_words = pat_lower.split()
            pat_len = len(pat_words)

            # Nếu query có ít hơn min_words → không match
            if q_len < min_words:
                continue

            # Pattern nhiều từ: cần match chính xác cụm từ
            if pat_len > 1:
                if pat_lower in q:
                    return response
                continue

            # Pattern 1 từ: cần word boundary (tránh "hi" trong "nhiêu")
            pat_word = pat_lower.strip()
            for word in q_words:
                # Xoá dấu câu dính vào từ
                clean_word = word.strip("?!.,;:")
                if clean_word == pat_word:
                    return response

    return None


# ── Intent detection từ câu hỏi gốc ──────────────────────────────────────

def _detect_intent_from_question(question: str) -> str | None:
    """Phát hiện intent từ câu hỏi gốc (tiếng Việt).

    Dùng keyword patterns, thứ tự ưu tiên: cụ thể → tổng quát.
    Returns tên intent hoặc None nếu không xác định được.
    """
    q = question.lower().strip()

    # 1. So sánh lốp
    if _has_any_keyword(q, ["so sánh", "khác nhau", " so với ", " vs "]):
        return "so_sanh"

    # 2. Săm
    if _has_any_keyword(q, ["săm", "ruột"]):
        return "sam"

    # 3. Hoa: lợi ích / ưu điểm / phù hợp
    if _has_any_keyword(q, ["hoa", "gai"]):
        if _has_any_keyword(q, ["lợi ích", "ưu điểm", "phù hợp", "tốt không", "điều kiện", "tác dụng"]):
            return "hoa_loi_ich"
        return "hoa_lop"

    # 4. Thương hiệu xe → danh sách xe
    if _has_any_keyword(q, ["honda", "yamaha", "piaggio", "sym", "vespa", "thương hiệu"]):
        if _has_any_keyword(q, ["xe", "dòng", "nào", "những"]):
            return "thuong_hieu_xe"

    # 5. Lốp cho loại xe (xe tay ga / xe số) — DISTINCT
    if _has_any_keyword(q, ["xe tay ga", "tay ga", "xe ga", "xe số", "xe sô"]):
        return "lop_xe"

    # 6. Tra lốp theo xe cụ thể
    if _has_any_keyword(q, ["lốp cho", "dùng lốp", "lốp gì", "xài lốp", "chạy lốp"]):
        return "xe_lop"

    # 7. Thống kê (ưu tiên hơn "gia" vì "giá bao nhiêu" có thể trùng)
    if _has_any_keyword(q, ["bao nhiêu", "tổng số", "mấy"]) and \
       _has_any_keyword(q, ["lốp", "xe", "cái", "loại", "sản phẩm"]):
        if not _has_any_keyword(q, ["giá", "tiền"]):
            return "thong_ke"

    # 8. Danh sách size
    if _has_any_keyword(q, ["size nào", "kích cỡ", "size có", "size của"]):
        return "ds_size"

    # 9. Thông số kỹ thuật
    if _has_any_keyword(q, ["tốc độ", "tải trọng", "thông số", "nặng bao", "kg", "km/h", "đường kính", "chiều rộng"]):
        return "thong_so"

    # 10. Giá cả
    if _has_any_keyword(q, ["giá", "bao nhiêu", "đắt", "rẻ", "tiền"]):
        return "gia"

    # 11. Mặc định: hỏi chung về xe / lốp
    if _has_any_keyword(q, ["xe"]):
        return "xe_lop"
    if _has_any_keyword(q, ["lốp"]):
        return "gia"

    return None


# ── Fallback: detect từ Cypher ───────────────────────────────────────────

def _detect_intent_from_cypher(cypher: str) -> str:
    """Fallback: phát hiện intent từ câu Cypher khi question detection không ra."""
    c = cypher.upper()

    if "TUBE" in c:
        return "sam"
    if "VEHICLEBRAND" in c:
        return "thuong_hieu_xe"
    if "COUNT(" in c:
        return "thong_ke"
    if "DISTINCT" in c and "TIRE_SIZE" in c:
        return "ds_size"
    if "DÙNG_LỐP_TRƯỚC" in c or "DÙNG_LỐP_SAU" in c:
        return "xe_lop"
    if "TIREPATTERN" in c:
        if "LOI_ICH" in c or "PHU_HOP" in c:
            return "hoa_loi_ich"
        return "hoa_lop"
    if "IN [" in c and "TIRE" in c:
        return "so_sanh"
    if "TIRE" in c:
        if "MAX_SPEED" in c or "MAX_LOAD" in c:
            return "thong_so"
        return "gia"
    if "VEHICLE" in c:
        return "xe_lop"
    return "general"


# ── Main intent detection ────────────────────────────────────────────────

def _detect_intent(question: str, cypher: str, records: list[dict]) -> str:
    """Xác định intent từ câu hỏi gốc, fallback sang Cypher nếu cần.

    Chiến lược:
      1. Phân tích câu hỏi gốc (keyword patterns)
      2. Nếu không match → dùng Cypher để suy luận
    """
    intent = _detect_intent_from_question(question)
    if intent:
        return intent
    return _detect_intent_from_cypher(cypher)


# ═══════════════════════════════════════════════════════════════════════════
#  Template Formatting — dispatch pattern
#  ═══════════════════════════════════════════════════════════════════════════

def _format_results(intent: str, question: str, records: list[dict]) -> str:
    """Format kết quả theo intent, không gọi Gemini.

    Dùng dispatch table để dễ maintain và mở rộng.
    """
    if not records:
        return "Tôi không tìm thấy thông tin phù hợp."

    handlers = {
        "gia": _format_gia,
        "thong_so": _format_thong_so,
        "so_sanh": _format_so_sanh,
        "xe_lop": _format_xe_lop,
        "lop_xe": _format_lop_xe,
        "hoa_lop": _format_hoa_lop,
        "hoa_loi_ich": _format_hoa_loi_ich,
        "sam": _format_sam,
        "thuong_hieu_xe": _format_thuong_hieu_xe,
        "thong_ke": _format_thong_ke,
        "ds_size": _format_ds_size,
    }

    handler = handlers.get(intent, _format_general)
    return handler(records, question=question)


# ── Handler: Giá lốp ─────────────────────────────────────────────────────

def _format_gia(records: list[dict], question: str | None = None) -> str:
    lines = []
    for r in records:
        size = r.get("tire_size") or r.get("size") or "?"
        brand = r.get("brand") or ""
        pattern = r.get("pattern_code") or r.get("code") or ""
        price = _fmt_price(r.get("sale_price_inc_vat") or r.get("gia") or r.get("price"))
        text = f"• **{brand}** **{size}**"
        if pattern:
            text += f" (hoa **{pattern}**)"
        text += f" — **{price}đ**"
        lines.append(text)
    return "📊 **Giá lốp**\n" + "\n".join(lines)


# ── Handler: Thông số kỹ thuật ───────────────────────────────────────────

def _format_thong_so(records: list[dict], question: str | None = None) -> str:
    lines = []
    for r in records:
        size = r.get("tire_size") or "?"
        brand = r.get("brand") or ""
        specs = []
        if r.get("max_speed_kmh"):
            specs.append(f"tốc độ **{r['max_speed_kmh']} km/h**")
        if r.get("max_load_kg"):
            specs.append(f"tải **{r['max_load_kg']} kg**")
        if r.get("outer_diameter_mm"):
            specs.append(f"đường kính **{r['outer_diameter_mm']} mm**")
        if r.get("overall_width_mm"):
            specs.append(f"rộng **{r['overall_width_mm']} mm**")
        if r.get("sale_price_inc_vat"):
            specs.append(f"giá **{_fmt_price(r['sale_price_inc_vat'])}đ**")
        lines.append(f"• **{brand}** **{size}**: {', '.join(specs)}")
    return "📐 **Thông số kỹ thuật**\n" + "\n".join(lines)


# ── Handler: So sánh lốp ────────────────────────────────────────────────

def _format_so_sanh(records: list[dict], question: str | None = None) -> str:
    lines = []
    for r in records:
        size = r.get("tire_size") or "?"
        brand = r.get("brand") or ""
        price = _fmt_price(r.get("sale_price_inc_vat"))
        speed = r.get("max_speed_kmh")
        load = r.get("max_load_kg")
        specs = []
        if price != "?":
            specs.append(f"**{price}đ**")
        if speed:
            specs.append(f"**{speed} km/h**")
        if load:
            specs.append(f"tải **{load}kg**")
        lines.append(f"• **{brand}** **{size}**: {', '.join(specs)}")
    return "⚖️ **So sánh lốp**\n" + "\n".join(lines)


# ── Handler: Tra lốp theo xe cụ thể ─────────────────────────────────────

def _format_xe_lop(records: list[dict], question: str | None = None) -> str:
    """Xe cụ thể: phân tích records để hiển thị tự nhiên."""
    first = records[0]

    # Detect vị trí được hỏi từ câu hỏi
    asked_pos = None
    if question:
        q = question.lower()
        if _has_any_keyword(q, ["lốp sau", "bánh sau", "lốp phía sau", "hư lốp sau"]):
            asked_pos = "sau"
        elif _has_any_keyword(q, ["lốp trước", "bánh trước", "lốp phía trước"]):
            asked_pos = "trước"

    # Helper: lấy giá trị từ nhiều key khả dĩ
    def _get(r, *keys):
        for k in keys:
            v = r.get(k)
            if v:
                return v
        return None

    lines = []

    # Case A: front_tire/rear_tire (thông tin lốp từ property Vehicle)
    if "front_tire" in first or "rear_tire" in first:
        for r in records:
            name = r.get("name") or "?"
            ft = r.get("front_tire") or "?"
            rt = r.get("rear_tire") or "?"
            extra = ""
            if r.get("tire_type"):
                extra += f" — {r['tire_type']}"
            if r.get("motorcycle_type"):
                loai = {"Scooter": "tay ga", "Manual": "xe số"}.get(r["motorcycle_type"], r["motorcycle_type"])
                extra += f", {loai}"
            lines.append(f"🚗 {name}{extra}")
            lines.append(f"   Trước: {ft}  |  Sau: {rt}")
        if asked_pos:
            val = records[0].get(f"{asked_pos}_tire")
            if val:
                lines.append(f"💡 Bạn hỏi lốp **{asked_pos}**: size {val}")
        return "\n".join(lines)

    # Case B: vi_tri (DÙNG_LỐP_TRƯỚC/SAU)
    if "vi_tri" in first:
        groups: dict[str, list] = {}
        for r in records:
            name = r.get("name") or "?"
            groups.setdefault(name, []).append(r)
        for name, tires in groups.items():
            lines.append(f"🚗 **{name}:**")
            for t in tires:
                pos = t.get("vi_tri", "")
                pos_vn = {"DÙNG_LỐP_TRƯỚC": "Trước", "DÙNG_LỐP_SAU": "Sau"}.get(pos, pos)
                size = _get(t, "tire_size", "size") or "?"
                brand = t.get("brand") or ""
                pattern = t.get("pattern_code") or ""
                price = _fmt_price(t.get("sale_price_inc_vat"))
                text = f"   **{pos_vn}**: **{brand}** **{size}**"
                if pattern:
                    text += f" (hoa **{pattern}**)"
                text += f" — **{price}đ**"
                lines.append(text)
        if asked_pos:
            pos_filter = {"trước": "DÙNG_LỐP_TRƯỚC", "sau": "DÙNG_LỐP_SAU"}.get(asked_pos)
            if pos_filter:
                matched = [r for r in records if r.get("vi_tri") == pos_filter]
                if matched:
                    lines.append(f"💡 Xe này dùng lốp **{asked_pos}** size {matched[0].get('tire_size', '?')}")
        return "\n".join(lines)

    # Case C: có name + các field lốp (complex query, alias custom như size_lop_sau)
    if "name" in first:
        groups: dict[str, list] = {}
        for r in records:
            n = r.get("name") or "?"
            groups.setdefault(n, []).append(r)
        for name, tires in groups.items():
            lines.append(f"🚗 **{name}:**")
            for t in tires:
                size = _get(t, "tire_size", "size_lop_sau", "size_lop_truoc", "size", "tire_size") or "?"
                brand = t.get("brand") or ""
                pattern = t.get("pattern_code") or ""
                loai_hoa = t.get("loai") or ""
                loi_ich = t.get("loi_ich") or ""
                price = _fmt_price(t.get("sale_price_inc_vat"))
                text = f"   **{brand}** **{size}**"
                if pattern:
                    text += f" (hoa **{pattern}**"
                    if loai_hoa:
                        text += f" — {loai_hoa}"
                    text += ")"
                if loi_ich:
                    # Lấy ngắn gọn benefit: chỉ lấy phần trước dấu ; đầu tiên
                    short = loi_ich.split(";")[0].strip()
                    text += f": {short}"
                text += f" — **{price}đ**"
                lines.append(text)
        if asked_pos:
            lines.append(f"💡 Bạn hỏi lốp **{asked_pos}** cho xe này.")
        return "\n".join(lines)

    # Fallback
    return _format_general(records, question)


# ── Handler: Lốp cho loại xe (xe tay ga / xe số) ────────────────────────

def _format_lop_xe(records: list[dict], question: str | None = None) -> str:
    seen = set()
    lines = []
    for r in records:
        size = r.get("tire_size") or "?"
        brand = r.get("brand") or ""
        pattern = r.get("pattern_code") or ""
        price = _fmt_price(r.get("sale_price_inc_vat"))
        key = (size, brand, pattern)
        if key in seen:
            continue
        seen.add(key)
        text = f"• **{brand}** **{size}**"
        if pattern:
            text += f" (hoa **{pattern}**)"
        text += f" — **{price}đ**"
        lines.append(text)
    return "🔧 **Lốp phù hợp**\n" + "\n".join(lines)


# ── Handler: Hoa lốp (lốp → kiểu hoa / kiểu hoa → lốp) ─────────────────

def _format_hoa_lop(records: list[dict], question: str | None = None) -> str:
    first = records[0]

    # Chỉ có code → danh sách kiểu hoa
    if "tire_size" not in first and "code" in first:
        lines = [f"• **{r.get('code','?')}** ({r.get('loai','')})" for r in records if r.get('code')]
        return "🎨 **Các kiểu hoa lốp**\n" + "\n".join(lines)

    # Có tire_size → từng lốp + hoa
    lines = []
    for r in records:
        size = r.get("tire_size") or "?"
        brand = r.get("brand") or ""
        code = r.get("code") or r.get("pattern_code") or ""
        loai = r.get("loai") or ""
        price = _fmt_price(r.get("sale_price_inc_vat"))
        text = f"• **{brand}** **{size}** (hoa **{code}**"
        if loai:
            text += f" — {loai}"
        text += f") — **{price}đ**"
        lines.append(text)
    return "🎨 **Lốp theo kiểu hoa**\n" + "\n".join(lines)


# ── Handler: Lợi ích hoa lốp ────────────────────────────────────────────

def _format_hoa_loi_ich(records: list[dict], question: str | None = None) -> str:
    lines = []
    for r in records:
        code = r.get("code") or "?"
        loai = r.get("loai") or ""
        loi_ich = r.get("loi_ich") or ""
        phu_hop = r.get("phu_hop") or ""
        dieu_kien = r.get("dieu_kien_duong") or ""
        text = f"• **{code}**: {loai}"
        if loi_ich:
            text += f" | {loi_ich}"
        if phu_hop:
            text += f" ✓ {phu_hop}"
        if dieu_kien:
            text += f" ({dieu_kien})"
        lines.append(text)
    return "🌿 **Thông tin hoa lốp**\n" + "\n".join(lines)


# ── Handler: Săm ────────────────────────────────────────────────────────

def _format_sam(records: list[dict], question: str | None = None) -> str:
    lines = []
    for r in records:
        size = r.get("tube_size") or "?"
        price = _fmt_price(r.get("sale_price_inc_vat"))
        text = f"• Săm **{size}** — **{price}đ**"
        if r.get("tire_size"):
            text += f" (dùng lốp **{r['tire_size']}**)"
        lines.append(text)
    return "🛞 **Săm xe**\n" + "\n".join(lines)


# ── Handler: Thương hiệu xe → danh sách xe ──────────────────────────────

def _format_thuong_hieu_xe(records: list[dict], question: str | None = None) -> str:
    lines = []
    for r in records:
        name = r.get("name") or "?"
        ft = r.get("front_tire") or ""
        rt = r.get("rear_tire") or ""
        spec = f" (**{ft}**/**{rt}**)" if ft and rt else ""
        lines.append(f"• **{name}**{spec}")
    return "🏢 **Danh sách xe**\n" + "\n".join(lines)


# ── Handler: Thống kê ───────────────────────────────────────────────────

def _format_thong_ke(records: list[dict], question: str | None = None) -> str:
    if not records or not records[0]:
        return "📊 **Kết quả:** 0"
    key = list(records[0].keys())[0]
    val = records[0].get(key) or 0
    return f"📊 **Kết quả:** {val}"


# ── Handler: Danh sách size ─────────────────────────────────────────────

def _format_ds_size(records: list[dict], question: str | None = None) -> str:
    sizes = [r.get("tire_size") or "?" for r in records if r.get("tire_size")]
    return f"📋 **Các size có sẵn**\n" + "\n".join(f"• {s}" for s in sizes)


# ── Handler: Fallback ───────────────────────────────────────────────────

def _format_general(records: list[dict], question: str | None = None) -> str:
    """Fallback: hiển thị dữ liệu gọn, đẹp với key tiếng Việt."""
    LABEL_MAP = {
        "name": "xe", "tire_size": "size", "size_lop_sau": "lốp sau",
        "size_lop_truoc": "lốp trước", "brand": "hãng", "pattern_code": "hoa",
        "code": "hoa", "loai": "loại", "sale_price_inc_vat": "giá",
        "max_speed_kmh": "tốc độ", "max_load_kg": "tải", "tube_size": "size săm",
        "loi_ich": "lợi ích", "phu_hop": "phù hợp", "dieu_kien_duong": "điều kiện",
        "motorcycle_type": "loại xe", "tire_type": "loại lốp",
        "front_tire": "lốp trước", "rear_tire": "lốp sau",
        "outer_diameter_mm": "đường kính", "overall_width_mm": "rộng",
        "vi_tri": "vị trí",
    }
    # Các key quan trọng cần in đậm
    BOLD_KEYS = {"tire_size", "sale_price_inc_vat", "gia", "price", "brand", "pattern_code", "code", "name", "tube_size", "front_tire", "rear_tire", "size_lop_sau", "size_lop_truoc"}
    lines = []
    for r in records[:10]:
        parts = []
        for k, v in r.items():
            if v is None:
                continue
            label = LABEL_MAP.get(k, k)
            val = v
            if k in ("sale_price_inc_vat", "gia", "price"):
                val = f"**{_fmt_price(v)}đ**"
            elif k in ("max_speed_kmh",):
                val = f"**{v} km/h**"
            elif k in ("max_load_kg",):
                val = f"**{v} kg**"
            elif k == "motorcycle_type":
                val = {"Scooter": "tay ga", "Manual": "xe số"}.get(v, v)
            elif k == "vi_tri":
                val = {"DÙNG_LỐP_TRƯỚC": "trước", "DÙNG_LỐP_SAU": "sau"}.get(v, v)
            elif k in BOLD_KEYS:
                val = f"**{v}**"
            else:
                val = str(v)
            parts.append(f"{label} {val}")
        lines.append("• " + " | ".join(parts))
    if len(records) > 10:
        lines.append(f"... +{len(records) - 10} kết quả nữa")
    return "📊 **Kết quả**\n" + "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  GraphRAGV5
# ═══════════════════════════════════════════════════════════════════════════

class GraphRAGV5:
    """Pipeline V5 — Text2Cypher thuần + Template (không Gemini paraphrase).

    Dùng Gemini sinh Cypher → Neo4j → detect intent → format template.
    Chỉ gọi Gemini 1 lần duy nhất, tiết kiệm 50% token so với V5 cũ.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or LLM_MODELS[0]

        # LLM client cho Text2Cypher (gọi 1 lần duy nhất)
        # Tự động fallback qua các model + API key dự phòng
        self.llm = LLMClient(model_name=self.model_name, temperature=0.0)

        # Neo4j service
        self.neo4j_service = Neo4jService()

        # Format examples cho prompt
        self._examples_text = "\n\n".join(
            f"Câu hỏi: {e['question']}\nCypher: {e['cypher']}"
            for e in EXAMPLES
        )

        # Cache cho fuzzy retry
        self._vehicle_names: list[str] | None = None

        logger.info("[V5] Khởi tạo xong với %d examples, model=%s", len(EXAMPLES), model_name)

    # ── Private ─────────────────────────────────────────────────────────

    def _extract_cypher(self, text: str) -> str:
        """Trích xuất câu Cypher từ response của LLM."""
        # Nếu có ```cypher ... ``` hoặc ``` ... ```
        m = re.search(r"```(?:cypher)?\s*\n?(.*?)```", text, re.DOTALL)
        if m:
            cypher = m.group(1).strip()
        else:
            cypher = text.strip()

        # Xử lý multi-statement: Neo4j chỉ chạy 1 câu
        # Nếu có nhiều câu cách nhau bằng ; → lấy câu cuối cùng (có RETURN)
        parts = [p.strip() for p in cypher.split(";") if p.strip()]
        if len(parts) > 1:
            # Tìm câu có RETURN → đó là câu chính
            for p in reversed(parts):
                if "RETURN" in p.upper():
                    cypher = p
                    break
            else:
                cypher = parts[-1]

        # Xoá ; ở cuối nếu còn
        cypher = cypher.rstrip(";").strip()
        return cypher

    def _generate_cypher(self, question: str, history: list | None = None) -> tuple[str | None, str | None]:
        """Gọi Gemini sinh Cypher từ câu hỏi + history context.

        Returns:
            (cypher, raw_response) — nếu lỗi thì (None, error_msg)
        """
        history = history or []
        history_context = ""
        if history:
            history_lines = []
            for h in history[-4:]:
                role = "Người dùng" if (isinstance(h, dict) and h.get("role") == "user") else "Trợ lý"
                text = h.get("text", h.get("content", "")) if isinstance(h, dict) else str(h)
                history_lines.append(f"{role}: {text}")
            history_context = "LỊCH SỬ HỘI THOẠI:\n" + "\n".join(history_lines) + "\n\n"

        prompt = CYPHER_PROMPT_TEMPLATE.format(
            schema=SCHEMA,
            examples=self._examples_text,
            history_context=history_context,
            question=question,
        )

        if LLM_MOCK:
            # Trả về Cypher mẫu để test
            return "MATCH (t:Tire) WHERE t.tire_size = '100/90-10' RETURN t.tire_size, t.brand, t.pattern_code, t.sale_price_inc_vat", prompt

        try:
            raw = self.llm.chat(prompt)
            if not raw:
                return None, "LLM trả về rỗng"
            cypher = self._extract_cypher(raw)
            if not cypher:
                return None, f"Không trích được Cypher từ:\n{raw[:200]}"
            return cypher, raw
        except Exception as e:
            logger.exception("[V5] LLM generate Cypher error")
            return None, str(e)

    def _execute_cypher(self, cypher: str) -> tuple[list[dict] | None, str | None]:
        """Execute Cypher trên Neo4j.

        Returns:
            (records, None) nếu thành công
            (None, error_msg) nếu lỗi
        """
        try:
            client = self.neo4j_service
            records = client.query(cypher)
            return records, None
        except Exception as e:
            logger.exception("[V5] Cypher execution error")
            return None, str(e)

    def _answer_from_context(self, question: str, cypher: str, records: list[dict], history: list | None = None) -> str:
        """Format kết quả dựa trên intent — KHÔNG gọi Gemini."""
        if not records:
            # Có history thì gợi ý user thử lại
            if history:
                return "Tôi không tìm thấy thông tin phù hợp. Bạn có thể thử hỏi theo cách khác hoặc kiểm tra lại tên xe nhé."
            return "Tôi không tìm thấy thông tin phù hợp."

        # Chuẩn hoá keys: 't.tire_size' → 'tire_size'
        records = _clean_keys(records)

        intent = _detect_intent(question, cypher, records)
        logger.info("[V5] Intent: %s (%s records)", intent, len(records))
        return _format_results(intent, question, records)

    def _answer_with_gemini(self, question: str, template_answer: str, records: list[dict], history: list | None = None) -> str | None:
        """Deep mode: dùng Gemini paraphrase câu trả lời cho tự nhiên hơn.

        Gọi Gemini với ANSWER_PROMPT_TEMPLATE để viết lại câu trả lời
        theo phong cách tư vấn viên, có gợi ý và recommend.
        """
        from app.pipeline.prompts import ANSWER_PROMPT_TEMPLATE

        # Inject history context nếu có
        history_context = ""
        if history:
            history_lines = []
            for h in history[-2:]:
                role = "Người dùng" if (isinstance(h, dict) and h.get("role") == "user") else "Trợ lý"
                text = h.get("text", h.get("content", "")) if isinstance(h, dict) else str(h)
                history_lines.append(f"{role}: {text}")
            if history_lines:
                history_context = "HỘI THOẠI TRƯỚC ĐÓ:\n" + "\n".join(history_lines) + "\n\n"

        data_str = json.dumps(records[:5], ensure_ascii=False, indent=2)
        prompt = ANSWER_PROMPT_TEMPLATE.format(
            history_context=history_context,
            data=data_str,
            question=question,
            template_answer=template_answer,
        )

        try:
            raw = self.llm.chat(prompt)
            if not raw or len(raw.strip()) < 10:
                return None
            return raw.strip()
        except Exception as e:
            logger.warning("[V5] Gemini paraphrase failed: %s", e)
            return None

    # ── Fuzzy Vehicle Name Retry ──────────────────────────────────────

    def _load_vehicle_names(self) -> list[str]:
        """Lấy danh sách tất cả tên xe từ Neo4j (có cache)."""
        if self._vehicle_names is not None:
            return self._vehicle_names
        try:
            rows = self.neo4j_service.query(
                "MATCH (v:Vehicle) RETURN v.name ORDER BY v.name"
            )
            names = [r["v.name"] for r in rows if r.get("v.name")]
            self._vehicle_names = names
            logger.info("[V5] Loaded %d vehicle names for fuzzy matching", len(names))
        except Exception as e:
            logger.warning("[V5] Failed to load vehicle names: %s", e)
            self._vehicle_names = []
        return self._vehicle_names

    def _fuzzy_retry_vehicle(self, query: str, cypher: str) -> tuple[list[dict] | None, str | None]:
        """Khi 0 records, thử fuzzy match tên xe và retry Cypher.

        Returns:
            (records, corrected_cypher) nếu retry thành công,
            (None, original_cypher) nếu không match được.
        """
        vehicle_names = self._load_vehicle_names()
        if not vehicle_names:
            return None, cypher

        q_lower = query.lower()

        # Step 1: Tìm vehicle name match tốt nhất bằng fuzzy
        best_match = None
        best_score = 0

        for name in vehicle_names:
            name_lower = name.lower()
            # partial_ratio: tên xe có xuất hiện trong query không?
            score = fuzz.partial_ratio(name_lower, q_lower)
            if score > best_score:
                best_score = score
                best_match = name

            # Hoặc ngược lại: query match vào tên xe dài
            score2 = fuzz.partial_ratio(q_lower, name_lower)
            if score2 > best_score:
                best_score = score2
                best_match = name

            # Token sort: bỏ qua thứ tự từ
            score3 = fuzz.token_sort_ratio(name_lower, q_lower)
            if score3 > best_score:
                best_score = score3
                best_match = name

        # Ngưỡng: >= 60 là đủ tốt để retry
        if best_score < 60 or not best_match:
            logger.info("[V5] Fuzzy retry: không tìm thấy xe nào match (best=%.0f%%)", best_score)
            return None, cypher

        logger.info("[V5] Fuzzy retry: phát hiện xe '%s' (score=%.0f%%) trong query '%s'",
                     best_match, best_score, query)

        # Step 2: Dùng LLM retry — CHỈ sửa tên xe, KHÔNG thêm filter benefit
        retry_prompt = f"""Bạn đang sửa câu Cypher bị sai tên xe.

CYPHR CŨ:
{cypher}
Lý do sai: 0 records vì tên xe KHÔNG ĐÚNG.
Tên xe ĐÚNG trong DB: "{best_match}"

NHIỆM VỤ:
- Giữ NGUYÊN cấu trúc Cypher cũ
- CHỈ thay tên xe trong CONTAINS thành "{best_match}"
- KHÔNG thêm WHERE, KHÔNG thêm filter, KHÔNG thêm MATCH nào khác
- Chỉ trả về câu Cypher đã sửa, không giải thích"""

        try:
            raw = self.llm.chat(retry_prompt)
            if not raw:
                logger.warning("[V5] Fuzzy retry: LLM trả về rỗng")
                return None, cypher
            new_cypher = self._extract_cypher(raw)
            if not new_cypher:
                return None, cypher

            logger.info("[V5] Fuzzy retry: Cypher mới = %s", new_cypher)
            records, err = self._execute_cypher(new_cypher)
            if records:
                logger.info("[V5] Fuzzy retry: thành công! %d records", len(records))
                return records, new_cypher
            else:
                logger.warning("[V5] Fuzzy retry: vẫn 0 records sau khi sửa")
                return None, cypher
        except Exception as e:
            logger.warning("[V5] Fuzzy retry lỗi: %s", e)
            return None, cypher

    # ── Public API ──────────────────────────────────────────────────────

    def run(self, query: str, mode: str = "fast", history: list | None = None) -> str:
        """Run pipeline: Gemini sinh Cypher → Neo4j → Template + optional Deep mode.

        Args:
            query: Câu hỏi bằng tiếng Việt.
            mode: "fast" (template, 1 call Gemini) hoặc "deep" (paraphrase, 2 calls Gemini).
            history: Lịch sử hội thoại (mặc định []) — dùng cho context xuyên suốt.

        Returns:
            Câu trả lời dạng text.
        """
        start = time.time()
        history = history or []
        logger.info("=" * 50)
        logger.info("[V5] Query: %s (mode=%s)", query, mode)
        if history:
            logger.info("[V5] History: %d tin nhắn", len(history))
            for i, h in enumerate(history[-4:]):  # log 4 tin gần nhất
                role = h.get("role", h.get("role", "?")) if isinstance(h, dict) else "?"
                text = h.get("text", h.get("content", str(h)[:80])) if isinstance(h, dict) else str(h)[:80]
                logger.info("[V5]   History[%d] %s: %s", i, role, text)
        logger.info("=" * 50)

        if not query or not query.strip():
            return "Vui lòng nhập câu hỏi."

        # ═════════════════════════════════════════════════════════════════
        # STEP 0: Kiểm tra greeting (không gọi LLM)
        # ═════════════════════════════════════════════════════════════════
        greeting_response = _detect_greeting(query)
        if greeting_response:
            logger.info("[V5] Phát hiện greeting, bỏ qua LLM")
            elapsed = time.time() - start
            logger.info("[V5] Done in %.2fs (greeting)", elapsed)
            return greeting_response

        # ═════════════════════════════════════════════════════════════════
        # STEP 1: Gemini sinh Cypher (có history context)
        # ═════════════════════════════════════════════════════════════════
        logger.info("[V5] Step 1: Gemini sinh Cypher...")
        cypher, err = self._generate_cypher(query, history)
        if cypher is None:
            logger.error("[V5] Không sinh được Cypher: %s", err)
            return "Xin lỗi, tôi không thể tạo truy vấn cho câu hỏi này."
        logger.info("[V5] Cypher: %s", cypher)

        # ═════════════════════════════════════════════════════════════════
        # STEP 2: Execute Cypher trên Neo4j
        # ═════════════════════════════════════════════════════════════════
        logger.info("[V5] Step 2: Execute Cypher...")
        records, exec_err = self._execute_cypher(cypher)
        if records is None:
            logger.error("[V5] Execute failed: %s", exec_err)
            return "Xin lỗi, truy vấn dữ liệu thất bại. Vui lòng thử lại."

        logger.info("[V5] Kết quả: %s records", len(records))

        # ═════════════════════════════════════════════════════════════════
        # STEP 2.5: Fuzzy retry nếu 0 records (tên xe sai)
        # ═════════════════════════════════════════════════════════════════
        if not records:
            logger.info("[V5] Step 2.5: Thử fuzzy retry...")
            fuzzy_records, corrected_cypher = self._fuzzy_retry_vehicle(query, cypher)
            if fuzzy_records:
                records = fuzzy_records
                cypher = corrected_cypher or cypher
                logger.info("[V5] Fuzzy retry tìm được %d records", len(records))

        # ═════════════════════════════════════════════════════════════════
        # STEP 3: Detect intent + format template (KHÔNG gọi Gemini)
        # ═════════════════════════════════════════════════════════════════
        logger.info("[V5] Step 3: Format kết quả...")
        answer = self._answer_from_context(query, cypher, records, history)

        # ═════════════════════════════════════════════════════════════════
        # STEP 4 (Deep mode): Gemini paraphrase câu trả lời (có history)
        # ═════════════════════════════════════════════════════════════════
        if mode == "deep" and records:
            logger.info("[V5] Step 4: Gemini paraphrase (deep mode)...")
            enhanced = self._answer_with_gemini(query, answer, records, history)
            if enhanced:
                logger.info("[V5] Deep mode: paraphrase thành công")
                answer = enhanced
            else:
                logger.info("[V5] Deep mode: paraphrase thất bại, giữ template")

        elapsed = time.time() - start
        logger.info("[V5] Done in %.2fs (mode=%s)", elapsed, mode)
        return answer

    def reset_context(self):
        """Reset context (V5 stateless, giữ interface cho tương thích)."""
        pass

    # ── Health ──────────────────────────────────────────────────────────

    def is_healthy(self) -> bool:
        try:
            client = self.neo4j_service.get_client()
            result = client.query("RETURN 1 AS ok")
            return result is not None and len(result) > 0
        except Exception:
            return False

    def get_retriever_info(self) -> dict[str, Any]:
        return {
            "version": "5.0",
            "engine": "Text2Cypher thuần (Gemini)",
            "examples": len(EXAMPLES),
            "model": self.model_name,
        }
