"""Cypher Mapper — map intent + query → Cypher query.

Step 4 trong pipeline:
  intent + raw_query → Cypher query + params

Tự trích xuất size/brand từ raw_query bằng regex, không cần SlotExtractor.
"""

import re
import unicodedata
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Known brands ───────────────────────────────────────────────────────
KNOWN_BRANDS = [
    "dplus", "irc", "maxxis", "bridgestone", "michelin", "pirelli",
    "continental", "dunlop", "goodyear", "kenda", "cst", "cheng shin",
    "shinko", "metzeler", "mrf", "ceat", "vietnam", "casumina",
]

SIZE_PATTERN = r'(\d+(?:\.\d+)?[-/]\d+(?:[-/]R?\d+)?[A-Za-z]?\b)'


def _normalize(text: str) -> str:
    """Lowercase + bỏ dấu."""
    if not text:
        return ""
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.replace("đ", "d")


# ═══════════════════════════════════════════════════════════════════════════
#  CypherMapper
# ═══════════════════════════════════════════════════════════════════════════

class CypherMapper:
    """Map intent + raw query → Cypher query + params.

    Usage:
        mapper = CypherMapper()
        cypher, params = mapper.map(intent="SPEED", query="lốp 120/70-17 tốc độ?")
    """

    # ── Public API ──────────────────────────────────────────────────────

    def map(self, intent: str | None, query: str) -> tuple[Optional[str], Optional[dict]]:
        """Map (intent, query) → (cypher_string, params_dict).

        Args:
            intent: Matched intent (e.g. SPEED, PRICE) or None.
            query: Raw user query string.

        Returns:
            Tuple of (cypher_string, params_dict) or (None, None).
        """
        size = self._extract_size(query)
        brand = self._extract_brand(query)
        compare_sizes = self._extract_compare_sizes(query)
        attribute = self._intent_to_attribute(intent)

        # ── No intent → single lookup if size found ─────────────────
        if not intent and size:
            return self._build_single(size)

        # ── Intent-based routing ────────────────────────────────────────
        builder_map: dict[str, callable] = {
            "SPECS":  lambda: self._build_single(size),
            "SINGLE":    lambda: self._build_single(size),
            "SPEED":     lambda: self._build_speed(size),
            "LOAD":      lambda: self._build_load(size),
            "PRICE":     lambda: self._build_price(size),
            "PRESSURE":  lambda: self._build_pressure(size),
            "MAX_LOAD":  lambda: self._build_max_load(size),
            "MAX_SPEED": lambda: self._build_max_speed(size),
            "MAX_PRICE": lambda: self._build_max_price(size),
            "COMPARE":   lambda: self._build_compare(size, compare_sizes),
            "BRAND":     lambda: self._build_brand(size, brand),
            "DRAINAGE":  lambda: self._build_attribute_search(attribute),
            "DURABILITY": lambda: self._build_attribute_search(attribute),
            "TUBE":      lambda: self._build_attribute_search(attribute),
            "SERVICE":   lambda: self._build_attribute_search(attribute),
        }

        builder = builder_map.get(intent) if intent else None
        if builder:
            result = builder()
            if result[0] is not None:
                return result

        # Fallback
        if size:
            return self._build_single(size)
        return None, None

    def map_from_entities(self, intent: str | None, size: str | None = None,
                           brand: str | None = None,
                           compare_sizes: list[str] | None = None) -> tuple[Optional[str], Optional[dict]]:
        """Map intent + pre-extracted entities → Cypher (bỏ qua regex).

        Dùng khi LLM fallback đã extract entities sẵn,
        hoặc khi dùng matched question từ QuestionBank.
        """
        attribute = self._intent_to_attribute(intent)

        builder_map: dict[str, callable] = {
            "SPECS":  lambda: self._build_single(size),
            "SINGLE":    lambda: self._build_single(size),
            "SPEED":     lambda: self._build_speed(size),
            "LOAD":      lambda: self._build_load(size),
            "PRICE":     lambda: self._build_price(size),
            "PRESSURE":  lambda: self._build_pressure(size),
            "MAX_LOAD":  lambda: self._build_max_load(size),
            "MAX_SPEED": lambda: self._build_max_speed(size),
            "MAX_PRICE": lambda: self._build_max_price(size),
            "COMPARE":   lambda: self._build_compare(size, compare_sizes or []),
            "BRAND":     lambda: self._build_brand(size, brand),
            "DRAINAGE":  lambda: self._build_attribute_search(attribute),
            "DURABILITY": lambda: self._build_attribute_search(attribute),
            "TUBE":      lambda: self._build_attribute_search(attribute),
            "SERVICE":   lambda: self._build_attribute_search(attribute),
        }

        builder = builder_map.get(intent) if intent else None
        if builder:
            result = builder()
            if result[0] is not None:
                return result

        if size:
            return self._build_single(size)
        return None, None

    # ── Inline extractors ───────────────────────────────────────────────

    @staticmethod
    def _extract_size(query: str) -> Optional[str]:
        for pat in (SIZE_PATTERN, r'(\d+[-/]\d+)'):
            m = re.search(pat, query)
            if m:
                return m.group(1)
        return None

    @staticmethod
    def _extract_compare_sizes(query: str) -> list[str]:
        sizes = re.findall(SIZE_PATTERN, query)
        return sizes if len(sizes) >= 2 else []

    @staticmethod
    def _extract_brand(query: str) -> Optional[str]:
        q = _normalize(query)
        for brand in KNOWN_BRANDS:
            if brand in q:
                return brand.upper()
        return None

    @staticmethod
    def _intent_to_attribute(intent: str | None) -> Optional[str]:
        """Map intent → attribute name for attribute_search."""
        if intent in ("DRAINAGE", "DURABILITY", "TUBE", "SERVICE"):
            return intent.lower()
        return None

    # ── Query Builders (standalone — no Slots dependency) ───────────────

    @staticmethod
    def _build_single(size: str) -> tuple[str, dict]:
        if not size:
            return None, None
        return ("""
        MATCH (t:Tire)
        WHERE t.size = $size
        OPTIONAL MATCH (t)-[:CÓ_HOA]->(p:TirePattern)
        RETURN
            t.size AS size, t.brand AS brand,
            t.toc_do_toi_da AS max_speed, t.tai_trong_lon_nhat AS max_load,
            t.noi_ap_tieu_chuan AS pressure, t.gia_ban_co_vat AS price,
            t.duong_kinh_ngoai AS diameter, t.duong_kinh_vanh AS rim,
            t.cau_truc_lop AS structure,
            COLLECT(DISTINCT p.pattern) AS pattern
        LIMIT 1
        """, {"size": size})

    @staticmethod
    def _build_speed(size: str) -> tuple[str, dict]:
        if not size:
            return None, None
        return ("MATCH (t:Tire) WHERE t.size = $size RETURN t.size AS size, t.brand AS brand, t.toc_do_toi_da AS max_speed LIMIT 1", {"size": size})

    @staticmethod
    def _build_load(size: str) -> tuple[str, dict]:
        if not size:
            return None, None
        return ("MATCH (t:Tire) WHERE t.size = $size RETURN t.size AS size, t.tai_trong_lon_nhat AS max_load LIMIT 1", {"size": size})

    @staticmethod
    def _build_price(size: str) -> tuple[str, dict]:
        if not size:
            return None, None
        return ("MATCH (t:Tire) WHERE t.size = $size RETURN t.size AS size, t.brand AS brand, t.gia_ban_co_vat AS price LIMIT 1", {"size": size})

    @staticmethod
    def _build_pressure(size: str) -> tuple[str, dict]:
        if not size:
            return None, None
        return ("MATCH (t:Tire) WHERE t.size = $size RETURN t.size AS size, t.noi_ap_tieu_chuan AS pressure LIMIT 1", {"size": size})

    @staticmethod
    def _build_max_load(size: str = None) -> tuple[str, dict]:
        if size:
            return ("MATCH (t:Tire) WHERE t.size = $size AND t.tai_trong_lon_nhat IS NOT NULL RETURN t.size AS size, t.tai_trong_lon_nhat AS max_load LIMIT 1", {"size": size})
        return ("""
        MATCH (t:Tire) WHERE t.tai_trong_lon_nhat IS NOT NULL
        WITH MAX(t.tai_trong_lon_nhat) AS max_load
        MATCH (t:Tire) WHERE t.tai_trong_lon_nhat = max_load
        RETURN t.size AS size, t.brand AS brand, t.tai_trong_lon_nhat AS max_load LIMIT 10
        """, None)

    @staticmethod
    def _build_max_speed(size: str = None) -> tuple[str, dict]:
        if size:
            return ("MATCH (t:Tire) WHERE t.size = $size AND t.toc_do_toi_da IS NOT NULL RETURN t.size AS size, t.toc_do_toi_da AS max_speed LIMIT 1", {"size": size})
        return ("""
        MATCH (t:Tire) WHERE t.toc_do_toi_da IS NOT NULL
        WITH MAX(t.toc_do_toi_da) AS max_speed
        MATCH (t:Tire) WHERE t.toc_do_toi_da = max_speed
        RETURN t.size AS size, t.brand AS brand, t.toc_do_toi_da AS max_speed LIMIT 10
        """, None)

    @staticmethod
    def _build_max_price(size: str = None) -> tuple[str, dict]:
        if size:
            return ("MATCH (t:Tire) WHERE t.size = $size AND t.gia_ban_co_vat IS NOT NULL RETURN t.size AS size, t.gia_ban_co_vat AS price LIMIT 1", {"size": size})
        return ("""
        MATCH (t:Tire) WHERE t.gia_ban_co_vat IS NOT NULL
        WITH MAX(t.gia_ban_co_vat) AS max_price
        MATCH (t:Tire) WHERE t.gia_ban_co_vat = max_price
        RETURN t.size AS size, t.brand AS brand, t.gia_ban_co_vat AS price LIMIT 10
        """, None)

    @staticmethod
    def _build_compare(size: str, compare_sizes: list[str]) -> tuple[str, dict]:
        all_sizes = compare_sizes if len(compare_sizes) >= 2 else ([size] if size else [])
        if not all_sizes:
            return None, None
        if len(all_sizes) == 1:
            return __class__._build_single(all_sizes[0])

        params = {f"size{i}": s for i, s in enumerate(all_sizes)}
        cond = " OR ".join(f"t.size = $size{i}" for i in range(len(all_sizes)))
        return (f"""
        MATCH (t:Tire) WHERE {cond}
        RETURN t.size AS size, t.brand AS brand,
               t.toc_do_toi_da AS max_speed, t.tai_trong_lon_nhat AS max_load,
               t.gia_ban_co_vat AS price, t.noi_ap_tieu_chuan AS pressure,
               t.duong_kinh_ngoai AS diameter, t.cau_truc_lop AS structure
        """, params)

    @staticmethod
    def _build_brand(size: str, brand: str) -> tuple[str, dict]:
        if brand:
            return ("""
            MATCH (t:Tire) WHERE toLower(t.brand) CONTAINS toLower($brand)
            RETURN t.size AS size, t.brand AS brand,
                   t.toc_do_toi_da AS max_speed, t.tai_trong_lon_nhat AS max_load,
                   t.gia_ban_co_vat AS price LIMIT 10
            """, {"brand": brand})
        if size:
            return __class__._build_single(size)
        return ("MATCH (t:Tire) RETURN DISTINCT t.brand AS brand ORDER BY brand", None)

    @staticmethod
    def _build_attribute_search(attribute: str | None) -> tuple[str, dict]:
        # TUBE — có dữ liệu thực tế trong Neo4j (t.co_sam)
        if attribute == "tube":
            return ("""
            MATCH (t:Tire) WHERE t.co_sam = True
            RETURN t.size AS size, t.brand AS brand, t.gia_ban_co_vat AS price
            LIMIT 10
            """, None)
        # drainage, durability, service — không có dữ liệu trong Neo4j
        # → trả về None để pipeline rơi vào LLM fallback
        return None, None
