"""Value store — caches Tire property values from Neo4j for fast lookup."""
import unicodedata
import os
import pickle

from app.services import Neo4jClient

CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "mapper", "value_store.pkl")


def normalize_text(text: str):
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.replace("đ", "d")


class ValueStore:
    def __init__(self):
        self.client = Neo4jClient()
        self.data = []
        self.columns = []

    def build(self):
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "rb") as f:
                data = pickle.load(f)
                self.data = data["data"]
                self.columns = data["columns"]
                print("[OK] Loaded ValueStore from cache")
                return

        print("[BUILD] Building ValueStore from Neo4j...")
        schema_query = """
        MATCH (t:Tire) WITH keys(t) AS props UNWIND props AS prop RETURN DISTINCT prop
        """
        props = self.client.query(schema_query)
        prop_list = [p["prop"] for p in props]

        for p in prop_list:
            self.columns.append({"column": p})

        for prop in prop_list:
            query = f"""
            MATCH (t:Tire) WHERE t.{prop} IS NOT NULL
            RETURN DISTINCT t.{prop} AS value LIMIT 1000
            """
            try:
                rows = self.client.query(query)
            except Exception:
                continue
            for r in rows:
                raw_val = str(r["value"]).strip()
                norm_val = normalize_text(raw_val)
                if not norm_val:
                    continue
                self.data.append({"value": norm_val, "raw_value": raw_val, "column": prop})

        with open(CACHE_FILE, "wb") as f:
            pickle.dump({"data": self.data, "columns": self.columns}, f)
        print(f"[OK] Loaded {len(self.data)} values from Neo4j")
