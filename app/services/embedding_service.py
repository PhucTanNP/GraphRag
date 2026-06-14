"""Embedding Service — scaffold, chờ implement model sau.

TODO: User sẽ implement embedding model (vd: SentenceTransformer/all-MiniLM-L6-v2).
      Hiện tại tất cả methods đều trả về None — pipeline tự fallback sang LLM.
"""
from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)

# ── Module-level singleton (model) ──────────────────────────────────────
_model = None


def get_model():
    """TODO: implement — load embedding model singleton."""
    global _model
    if _model is None:
        logger.info("[Embedding] Model chưa được implement — pipeline sẽ dùng LLM fallback")
    return _model


# ═══════════════════════════════════════════════════════════════════════════
#  EmbeddingService
# ═══════════════════════════════════════════════════════════════════════════

class EmbeddingService:
    """Embed user query → vector.

    TODO: Implement ``get_model()`` với model embedding thật.
          - Input: câu hỏi tiếng Việt
          - Output: numpy vector (384-dim, normalized)
    """

    def __init__(self):
        self.model = get_model()

    def embed(self, query: str) -> np.ndarray | None:
        """TODO: implement — embed single query → normalized vector."""
        return None

    def embed_batch(self, queries: list[str]) -> np.ndarray | None:
        """TODO: implement — embed multiple queries → matrix (N, 384)."""
        return None

    def is_healthy(self) -> bool:
        return self.model is not None


# ── Convenience factory ──────────────────────────────────────────────────
def init_embedding() -> EmbeddingService:
    """Factory: create + return a ready-to-use EmbeddingService."""
    logger.info("[SERVICES] Initialising EmbeddingService...")
    service = EmbeddingService()
    if service.is_healthy():
        logger.info("[SERVICES] EmbeddingService ready (384-dim)")
    else:
        logger.warning("[SERVICES] EmbeddingService NOT available")
    return service
