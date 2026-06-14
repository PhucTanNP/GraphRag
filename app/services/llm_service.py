"""LLM Service — khởi tạo + quản lý LLM client (Gemini / Mock).

Cung cấp:
  - ``LLMClient`` class (full client, merged from ``app.llm_client``)
  - ``LLMService`` wrapper với singleton + lifecycle
  - ``get_llm()`` singleton accessor
  - ``init_llm()`` factory

Usage:
    from app.services import LLMClient, LLMService

    client = LLMClient()
    answer = client.chat("lốp 120/70-17 tốc độ bao nhiêu?")
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from google import genai

from app.config import GEMINI_API_KEY, LLM_MOCK

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════
#  LLMClient
# ═══════════════════════════════════════════════════════════════════════════
class LLMClient:
    """Gemini LLM client với mock fallback."""

    def __init__(self, model_name=None, temperature=0.0, max_retries=2):
        self.client = None
        self._mock = LLM_MOCK or not GEMINI_API_KEY

        if not self._mock:
            self.client = genai.Client(api_key=GEMINI_API_KEY)

        self.model = model_name or "models/gemini-3.1-flash-lite"
        self._fallback_models = [
            "models/gemini-3.5-flash",
            "models/gemini-3.1-flash-lite",
            "models/gemini-2.5-flash",
        ]
        self.temperature = temperature
        self.max_retries = max_retries

    def chat(self, prompt: str) -> str:
        if not prompt or not prompt.strip():
            return ""

        if self._mock:
            logger.info("[LLM] MOCK — no token counting")
            return "MOCK_RESPONSE"

        models_to_try = [self.model] + [m for m in self._fallback_models if m != self.model]

        for attempt in range(self.max_retries):
            for model_candidate in models_to_try:
                try:
                    response = self.client.models.generate_content(
                        model=model_candidate, contents=prompt
                    )
                    if response and hasattr(response, "text") and getattr(response, "text", "").strip():
                        resp_text = response.text.strip()

                        # ── Log token usage ────────────────────────────
                        usage = getattr(response, "usage_metadata", None)
                        if usage is not None:
                            logger.info(
                                "[LLM] model=%s | tokens: prompt=%s candidate=%s total=%s",
                                model_candidate,
                                getattr(usage, "prompt_token_count", "?"),
                                getattr(usage, "candidates_token_count", "?"),
                                getattr(usage, "total_token_count", "?"),
                            )
                        else:
                            logger.info("[LLM] model=%s | tokens: N/A (no usage_metadata)", model_candidate)

                        return resp_text
                except Exception as e:
                    if attempt == self.max_retries - 1 and model_candidate == models_to_try[-1]:
                        logger.error(f"Gemini error (final): {e}")
                    continue
            time.sleep(2 * (attempt + 1))

        logger.warning("[LLM] All models failed — returning empty")
        return ""


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
            self.client._mock or self.client.client is not None
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
