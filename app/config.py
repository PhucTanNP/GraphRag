"""Configuration: read from env, files (K8s), or Vault (optional).

Provides a single `settings` object used by the app.
"""
import os
import logging

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class SecretsManager:
    def __init__(self, env_prefix: str = ""):
        self.env_prefix = env_prefix

    def _env_key(self, key: str) -> str:
        return f"{self.env_prefix}{key}"

    def get(self, key: str, default: str = None) -> str:
        env_key = self._env_key(key)
        val = os.environ.get(env_key)
        if val:
            return val

        file_env = os.environ.get(env_key + "_FILE")
        if file_env and os.path.exists(file_env):
            try:
                return open(file_env, "r", encoding="utf-8").read().strip()
            except Exception:
                logger.exception("Failed reading secret file %s", file_env)

        path = f"/var/run/secrets/{key}"
        if os.path.exists(path):
            try:
                return open(path, "r", encoding="utf-8").read().strip()
            except Exception:
                logger.exception("Failed reading secret file %s", path)

        try:
            vault_addr = os.environ.get("VAULT_ADDR")
            vault_token = os.environ.get("VAULT_TOKEN")
            vault_path = os.environ.get("VAULT_SECRET_PATH")
            if vault_addr and vault_token and vault_path:
                try:
                    import hvac
                except Exception:
                    logger.warning("hvac not installed; skipping Vault secret fetch")
                    return default
                try:
                    client = hvac.Client(url=vault_addr, token=vault_token)
                    secret = client.secrets.kv.v2.read_secret_version(path=vault_path)
                    data = secret.get("data", {}).get("data", {})
                    if key in data:
                        return data[key]
                except Exception:
                    logger.exception("Vault secret read failed")
        except Exception:
            logger.exception("Vault fetch setup failed")

        return default


_secrets = SecretsManager()


def get_secret(key: str, default: str = None) -> str:
    return _secrets.get(key, default=default)


# ── Model paths ──────────────────────────────────────────────────────────────
import os as _os
_BASE = _os.path.dirname(_os.path.abspath(__file__))
_MODELS_DIR = _os.path.join(_BASE, "models", "models")

YOLO_SEG_MODEL_PATH = _os.environ.get("YOLO_SEG_MODEL_PATH") or _os.path.join(_MODELS_DIR, "best (5).pt")
YOLO_DETECT_MODEL_PATH = _os.environ.get("YOLO_DETECT_MODEL_PATH") or _os.path.join(_MODELS_DIR, "best (4).pt")
YOLO_CONF_THRESHOLD = float(_os.environ.get("YOLO_CONF_THRESHOLD", "0.25"))

PADDLE_OCR_MODEL_PATH = _os.environ.get("PADDLE_OCR_MODEL_PATH") or _os.path.join(_MODELS_DIR, "tire_infer_v3")
PADDLE_DICT_PATH = _os.environ.get("PADDLE_DICT_PATH") or _os.path.join(_MODELS_DIR, "paddle_dict.txt")

TARGET_OCR_HEIGHT = int(_os.environ.get("TARGET_OCR_HEIGHT", "48"))
MAX_OCR_WIDTH = int(_os.environ.get("MAX_OCR_WIDTH", "320"))
MAX_IMAGE_WIDTH = int(_os.environ.get("MAX_IMAGE_WIDTH", "2268"))
MIN_IMAGE_HEIGHT = int(_os.environ.get("MIN_IMAGE_HEIGHT", "32"))


class _Settings:
    """Settings namespace — cho các model file cũ import 'from app.config import settings'."""
    yolo_seg_model_path = YOLO_SEG_MODEL_PATH
    yolo_detect_model_path = YOLO_DETECT_MODEL_PATH
    yolo_conf_threshold = YOLO_CONF_THRESHOLD
    paddle_ocr_model_path = PADDLE_OCR_MODEL_PATH
    paddle_dict_path = PADDLE_DICT_PATH
    target_ocr_height = TARGET_OCR_HEIGHT
    max_ocr_width = MAX_OCR_WIDTH
    max_image_width = MAX_IMAGE_WIDTH
    min_image_height = MIN_IMAGE_HEIGHT

settings = _Settings()

# ── Neo4j ────────────────────────────────────────────────────────────────────
NEO4J_ENABLED = os.environ.get("NEO4J_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER") or os.environ.get("NEO4J_USERNAME") or "neo4j"
NEO4J_PASSWORD = get_secret("NEO4J_PASSWORD", default=os.environ.get("NEO4J_PASSWORD", ""))

# ── Redis (optional) ─────────────────────────────────────────────────────────
REDIS_URL = os.environ.get("REDIS_URL") or None

# ── LLM ──────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or None
GEMINI_API_KEY_2 = os.environ.get("GEMINI_API_KEY_2") or None
GEMINI_API_KEY_3 = os.environ.get("GEMINI_API_KEY_3") or None

# Danh sách model fallback (theo thứ tự ưu tiên)
LLM_MODELS = [
    "models/gemini-3.5-flash",
    "models/gemini-3.1-flash-lite",
    "models/gemini-2.5-flash",
]

# Danh sách API key fallback (theo thứ tự ưu tiên)
LLM_API_KEYS = [
    k for k in [GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_3] if k
]

LLM_MOCK = os.environ.get("LLM_MOCK", "0").strip().lower() in ("1", "true", "yes", "on")

# ── Log LLM config (dùng print vì logging chưa setup khi config được import) ──
_keys_log = []
for i, key in enumerate([GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_3], 1):
    if key:
        _keys_log.append(f"Key{i}=✅ {key[:8]}...{key[-4:]}")
    else:
        _keys_log.append(f"Key{i}=⬜ (trống)")
print(f"[CONFIG] 🔑 LLM keys: {' | '.join(_keys_log)}")
print(f"[CONFIG] 🤖 LLM models: {', '.join(LLM_MODELS)}")
print(f"[CONFIG] 🔧 LLM_MOCK={LLM_MOCK}")

# ── Observability ────────────────────────────────────────────────────────────
ENABLE_OTEL = os.environ.get("ENABLE_OTEL", "false").lower() in ("1", "true", "yes")
OTEL_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or os.environ.get("OTLP_ENDPOINT") or None
