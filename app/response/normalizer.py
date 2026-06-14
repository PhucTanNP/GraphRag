"""Normalise DB records to canonical consumer-friendly keys."""


def normalize_record(rec: dict) -> dict:
    if not isinstance(rec, dict):
        return rec
    nr = dict(rec)
    nr["max_speed"] = nr.get("max_speed") or nr.get("toc_do_toi_da") or nr.get("speed")
    nr["max_load"] = nr.get("max_load") or nr.get("tai_trong_lon_nhat") or nr.get("load")
    nr["price"] = nr.get("price") or nr.get("gia_ban_co_vat") or nr.get("gia_ban")
    nr["speed"] = nr.get("speed") or nr.get("max_speed") or nr.get("toc_do_toi_da")
    nr["load"] = nr.get("load") or nr.get("max_load") or nr.get("tai_trong_lon_nhat")
    return nr


def normalize_data(data):
    if data is None:
        return data
    if isinstance(data, dict):
        return normalize_record(data)
    if isinstance(data, list):
        return [normalize_record(d) if isinstance(d, dict) else d for d in data]
    return data


def dedup_data(data: list[dict]) -> list[dict]:
    """Deduplicate a list of dicts, safely handling unhashable values (e.g. lists)."""
    seen = set()
    result = []
    for d in data:
        # Convert all values to strings for hashing
        key = tuple(
            (k, str(v)) for k, v in sorted(d.items())
        )
        if key not in seen:
            seen.add(key)
            result.append(d)
    return result
