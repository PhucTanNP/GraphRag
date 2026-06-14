"""Semantic Matcher — so khớp ngữ nghĩa câu hỏi user với Question Bank.

Step 2 trong pipeline:
  User Query vector → cosine similarity với Question Bank → best intent
"""

import numpy as np
import os
import logging

from app.services import EmbeddingService
from app.matching.question_bank import QuestionBank

logger = logging.getLogger(__name__)


class SemanticMatcher:
    """Match user query against question bank using cosine similarity.

    Usage:
        matcher = SemanticMatcher()
        result = matcher.match("lốp 120/70-17 tốc độ bao nhiêu")
        # → {"intent": "SPEED", "confidence": 0.85, "question": "..."}
    """

    def __init__(self, embedder: EmbeddingService | None = None, question_bank: QuestionBank | None = None):
        self.embedder = embedder or EmbeddingService()
        self.question_bank = question_bank or QuestionBank()
        self._built = False

    def build(self) -> None:
        """Build question bank embeddings."""
        if self._built:
            return
        self.question_bank.build(self.embedder)
        self._built = True

    def match(self, query: str, threshold: float = 0.40) -> dict | None:
        """Match query against question bank.

        Args:
            query: Raw user query.
            threshold: Minimum confidence threshold.

        Returns:
            {"intent": str, "confidence": float, "question": str} or None.
        """
        if not query or not query.strip():
            return None

        self.build()

        q_vec = self.embedder.embed(query)
        if q_vec is None:
            return None

        # Try grouped match first (more accurate for intent detection)
        result = self.question_bank.match_grouped(q_vec, threshold=threshold)
        if result is not None:
            return result

        # Fallback to single best match
        result = self.question_bank.match(q_vec, threshold=threshold)
        return result

    def match_with_threshold(self, query: str) -> tuple[dict | None, float]:
        """Match and return result with raw confidence.

        Returns:
            (result dict or None, raw confidence score)
        """
        if not query or not query.strip():
            return None, 0.0

        self.build()

        q_vec = self.embedder.embed(query)
        if q_vec is None:
            return None, 0.0

        result = self.question_bank.match_grouped(q_vec, threshold=0.0)
        if result is None:
            # Try single match
            result = self.question_bank.match(q_vec, threshold=0.0)

        if result is None:
            return None, 0.0

        return result, result["confidence"]

    def is_healthy(self) -> bool:
        return self.embedder.is_healthy() and self.question_bank.is_healthy()
