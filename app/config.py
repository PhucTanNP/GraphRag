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


# ── Neo4j ────────────────────────────────────────────────────────────────────
NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER") or os.environ.get("NEO4J_USERNAME") or "neo4j"
NEO4J_PASSWORD = get_secret("NEO4J_PASSWORD", default=os.environ.get("NEO4J_PASSWORD", ""))

# ── Redis (optional) ─────────────────────────────────────────────────────────
REDIS_URL = os.environ.get("REDIS_URL") or None

# ── LLM ──────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or None
LLM_MOCK = os.environ.get("LLM_MOCK", "0").strip().lower() in ("1", "true", "yes", "on")

# ── Observability ────────────────────────────────────────────────────────────
ENABLE_OTEL = os.environ.get("ENABLE_OTEL", "false").lower() in ("1", "true", "yes")
OTEL_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or os.environ.get("OTLP_ENDPOINT") or None
