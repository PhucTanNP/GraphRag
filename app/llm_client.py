"""LLM client — standalone initialisation.

Uses Gemini API via ``google-genai`` when ``GEMINI_API_KEY`` is set,
otherwise falls back to a deterministic mock (useful for tests / dev).
"""
import os
import time
import logging
from dotenv import load_dotenv
from google import genai

from app.config import GEMINI_API_KEY, LLM_MOCK
from app import metrics

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace as ot_trace
except Exception:
    ot_trace = None

load_dotenv()


class LLMClient:
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

        start = time.time()

        if self._mock:
            resp = "MOCK_RESPONSE"
            self._record_metrics(True, start)
            return resp

        models_to_try = [self.model] + [m for m in self._fallback_models if m != self.model]

        for attempt in range(self.max_retries):
            for model_candidate in models_to_try:
                try:
                    response = self.client.models.generate_content(
                        model=model_candidate, contents=prompt
                    )
                    if response and hasattr(response, "text") and getattr(response, "text", "").strip():
                        resp_text = response.text.strip()
                        self._record_metrics(True, start)
                        return resp_text
                except Exception as e:
                    if attempt == self.max_retries - 1 and model_candidate == models_to_try[-1]:
                        logger.error(f"Gemini error (final): {e}")
                    continue

            time.sleep(2 * (attempt + 1))

        self._record_metrics(False, start)
        return ""

    # ── internal ──────────────────────────────────────────────────────────

    def _record_metrics(self, success: bool, start: float):
        try:
            if metrics.llm_call_counter is not None:
                metrics.llm_call_counter.labels(success=str(success).lower()).inc()
            if metrics.llm_latency is not None:
                metrics.llm_latency.observe(round(time.time() - start, 3))
        except Exception:
            pass
