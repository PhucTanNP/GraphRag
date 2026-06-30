"""LLM Service — khởi tạo + quản lý LLM client (Gemini / Mock).

Cung cấp:
  - ``LLMClient`` class (full client, merged from ``app.llm_client``)
  - ``LLMService`` wrapper với singleton + lifecycle
  - ``GeminiLLM`` — ``LLMInterface`` wrapper cho ``neo4j-graphrag`` Text2Cypher
  - ``get_llm()`` singleton accessor
  - ``init_llm()`` factory

Usage:
    from app.services import LLMClient, LLMService

    client = LLMClient()
    answer = client.chat("lốp 120/70-17 tốc độ bao nhiêu?")

    # Dùng với neo4j-graphrag:
    from app.services.llm_service import GeminiLLM
    llm = GeminiLLM()
    response = llm.invoke("MATCH (t:Tire) RETURN t LIMIT 1")
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from google import genai

from app.config import GEMINI_API_KEY, LLM_MOCK

try:
    from neo4j_graphrag.llm import LLMInterface, LLMResponse
except ImportError:
    # Fallback nếu chưa cài neo4j-graphrag
    class LLMInterface:  # type: ignore
        pass

    class LLMResponse:  # type: ignore
        def __init__(self, content: str):
            self.content = content

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════
#  LLMClient
# ═══════════════════════════════════════════════════════════════════════════
class LLMClient:
    """Gemini LLM client với multi-key + multi-model fallback.

    Tự động luân phiên:
      1. Thử model chính với từng key (GEMINI_API_KEY_1 → 2 → 3)
      2. Nếu hết key → fallback sang model tiếp theo
      3. Log rõ model + key đang dùng
    """

    def __init__(
        self,
        model_name: str | None = None,
        models: list[str] | None = None,
        api_keys: list[str] | None = None,
        temperature: float = 0.0,
    ):
        self.temperature = temperature

        # Models
        from app.config import LLM_MODELS as FALLBACK_MODELS

        self._all_models = models or FALLBACK_MODELS
        # Nếu model_name được chỉ định, đặt lên đầu danh sách
        if model_name and model_name not in self._all_models:
            self._all_models = [model_name] + self._all_models
        elif model_name and model_name in self._all_models:
            self._all_models = [model_name] + [m for m in self._all_models if m != model_name]

        self.model = self._all_models[0]  # model chính (dùng cho log)

        # API keys
        from app.config import LLM_API_KEYS as FALLBACK_KEYS

        self._all_keys = api_keys or FALLBACK_KEYS

        self._mock = LLM_MOCK or not self._all_keys
        if not self._mock:
            self._client = genai.Client(api_key=self._all_keys[0])
        else:
            self._client = None  # type: ignore

    def _log_key_index(self, key: str) -> int:
        """Trả về index của key (1, 2, 3...) để log thân thiện."""
        try:
            return self._all_keys.index(key) + 1
        except ValueError:
            return 0

    def chat(self, prompt: str) -> str:
        if not prompt or not prompt.strip():
            return ""

        if self._mock:
            logger.info("[LLM] MOCK — no token counting")
            return "MOCK_RESPONSE"

        last_error = None

        # Strategy: thử model chính → model nhỏ hơn trên cùng key,
        # sau đó mới chuyển key tiếp theo. Tránh dồn cùng model quá tải.
        for key_idx, api_key in enumerate(self._all_keys):
            key_num = key_idx + 1

            # Khởi tạo client với key này
            try:
                self._client = genai.Client(api_key=api_key)
            except Exception as e:
                logger.warning("[LLM] Key %d init failed: %s", key_num, e)
                continue

            for model_idx, model_candidate in enumerate(self._all_models):
                # Log model + key
                if key_idx == 0 and model_idx == 0:
                    logger.info("[LLM] Using model=%s | key=%d", model_candidate, key_num)
                else:
                    logger.warning(
                        "[LLM] ⚠️ FALLBACK → model=%s key=%d (lý do: %s trước đó failed)",
                        model_candidate,
                        key_num,
                        self._all_models[model_idx - 1] if model_idx > 0 else f"key_{key_num - 1} hết",
                    )

                try:
                    response = self._client.models.generate_content(
                        model=model_candidate, contents=prompt
                    )
                    if response and hasattr(response, "text") and getattr(response, "text", "").strip():
                        resp_text = response.text.strip()

                        # ── Log token usage ────────────────────────────
                        usage = getattr(response, "usage_metadata", None)
                        if usage is not None:
                            logger.info(
                                "[LLM] model=%s | key=%d | tokens: prompt=%s candidate=%s total=%s",
                                model_candidate,
                                key_num,
                                getattr(usage, "prompt_token_count", "?"),
                                getattr(usage, "candidates_token_count", "?"),
                                getattr(usage, "total_token_count", "?"),
                            )
                        else:
                            logger.info(
                                "[LLM] model=%s | key=%d | tokens: N/A",
                                model_candidate,
                                key_num,
                            )

                        return resp_text
                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    if "429" in err_str or "quota" in err_str.lower():
                        logger.warning(
                            "[LLM] ⚠️ Key %d hết quota với model %s",
                            key_num,
                            model_candidate,
                        )
                    else:
                        logger.warning(
                            "[LLM] ⚠️ Lỗi model=%s key=%d: %s",
                            model_candidate,
                            key_num,
                            err_str[:100],
                        )
                    continue

        # Hết tất cả model + key
        logger.error("[LLM] ❌ Tất cả models và keys đều failed: %s", last_error)
        return ""


# ═══════════════════════════════════════════════════════════════════════════
#  GeminiLLM — LLMInterface for neo4j-graphrag
# ═══════════════════════════════════════════════════════════════════════════

class GeminiLLM(LLMInterface):
    """Wrapper Gemini → LLMInterface (dùng cho Text2CypherRetriever).

    Usage:
        llm = GeminiLLM()
        result = llm.invoke("CREATE (t:Tire {size: '120/70-17'})")
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "models/gemini-2.0-flash",
        temperature: float = 0.0,
    ):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name
        self.temperature = temperature
        self._mock = LLM_MOCK or not self.api_key

        if not self._mock:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    def invoke(self, prompt: str) -> LLMResponse:
        """Gọi Gemini và trả về LLMResponse (required bởi LLMInterface)."""
        if self._mock or self.client is None:
            logger.info("[GeminiLLM] MOCK — returning empty")
            return LLMResponse(content="")

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            text = response.text.strip() if response.text else ""
            return LLMResponse(content=text)
        except Exception as e:
            logger.error(f"[GeminiLLM] Error: {e}")
            return LLMResponse(content="")

    def is_healthy(self) -> bool:
        return self.client is not None or self._mock


# ── Module-level singleton ───────────────────────────────────────────────
_llm_instance: LLMClient | None = None


def get_llm() -> LLMClient:
    """Return singleton LLMClient (lazy init on first call)."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMClient()
    return _llm_instance


# ═══════════════════════════════════════════════════════════════════════════
#  LLMService
# ═══════════════════════════════════════════════════════════════════════════
class LLMService:
    """Wrapper service quanh ``LLMClient`` với singleton + lifecycle."""

    def __init__(self, client: LLMClient | None = None):
        self.client = client or get_llm()

    def chat(self, prompt: str) -> str:
        return self.client.chat(prompt)

    def chat_with_context(self, prompt: str, context: str) -> str:
        full_prompt = f"{context}\n\n---\n\n{prompt}"
        return self.client.chat(full_prompt)

    def is_healthy(self) -> bool:
        return self.client is not None and (
            self.client._mock or self.client._client is not None
        )


# ── Convenience factory ──────────────────────────────────────────────────
def init_llm() -> LLMService:
    logger.info("[SERVICES] Initialising LLMService...")
    service = LLMService()
    if service.is_healthy():
        if service.client._mock:
            logger.info("[SERVICES] LLMService running in MOCK mode")
        else:
            logger.info("[SERVICES] LLMService ready (model=%s)", service.client.model)
    else:
        logger.warning("[SERVICES] LLMService NOT available")
    return service
