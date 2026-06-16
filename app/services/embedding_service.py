"""Embedding Service — dùng BM25 (Okapi BM25) để scoring.

BM25 là thuật toán xếp hạng cổ điển, siêu nhẹ, không cần GPU/PyTorch.
Fit trên 420+ câu hỏi mẫu của QuestionBank → user query được so khớp ngữ nghĩa.

Usage:
    svc = EmbeddingService()
    svc.fit()                          # train từ question_bank
    scores = svc.embed("lốp 120/70-17")  # → ndarray (420,) — BM25 scores
    batch  = svc.embed_batch([...])      # → ndarray (N, 420)
"""
from __future__ import annotations

import math
import os
import pickle
import logging
import re
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)

# ── Cache path ──────────────────────────────────────────────────────────
_CACHE_DIR = os.environ.get(
    "EMBEDDING_CACHE_DIR",
    os.path.join(os.path.dirname(__file__), "..", "Embeding_vector"),
)
_BM25_CACHE = os.path.join(_CACHE_DIR, "bm25_model.pkl")

# ── Module-level singleton ──────────────────────────────────────────────
_model = None  # Will hold dict: {"bm25": BM25Okapi, "questions": [...], "intents": [...]}


# ═══════════════════════════════════════════════════════════════════════════
#  BM25 Okapi implementation (no external dependencies)
# ═══════════════════════════════════════════════════════════════════════════

class BM25Okapi:
    """Okapi BM25 implementation — pure Python, zero dependencies."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.avgdl = sum(len(doc) for doc in corpus) / len(corpus) if corpus else 0
        self.nd = len(corpus)

        # IDF for each term
        self.idf: dict[str, float] = {}
        _df = Counter()  # document frequency
        for doc in corpus:
            for term in set(doc):
                _df[term] += 1
        for term, df in _df.items():
            self.idf[term] = math.log(1 + (self.nd - df + 0.5) / (df + 0.5))

    def get_scores(self, query: list[str]) -> list[float]:
        scores = []
        for doc in self.corpus:
            score = 0.0
            doc_len = len(doc)
            for q_term in query:
                if q_term not in self.idf:
                    continue
                tf = doc.count(q_term)
                numer = tf * (self.k1 + 1)
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += self.idf[q_term] * numer / denom if denom else 0.0
            scores.append(score)
        return scores


# ═══════════════════════════════════════════════════════════════════════════
#  Tokenizer
# ═══════════════════════════════════════════════════════════════════════════

def tokenize(text: str) -> list[str]:
    """Tokenize tiếng Việt đơn giản: lowercase + tách từ."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)  # bỏ dấu câu
    return [t for t in text.split() if len(t) > 1]  # bỏ từ 1 ký tự


# ═══════════════════════════════════════════════════════════════════════════
#  Model loader
# ═══════════════════════════════════════════════════════════════════════════

def get_model():
    """Load BM25 model singleton từ cache."""
    global _model
    if _model is not None:
        return _model

    if os.path.exists(_BM25_CACHE):
        try:
            with open(_BM25_CACHE, "rb") as f:
                _model = pickle.load(f)
            logger.info("[Embedding] Loaded BM25 model from cache (%s questions)", len(_model.get("questions", [])))
        except Exception as e:
            logger.warning("[Embedding] BM25 cache load failed: %s", e)
            _model = None
    else:
        logger.info("[Embedding] No BM25 cache found — call fit() first")

    return _model


# ═══════════════════════════════════════════════════════════════════════════
#  EmbeddingService
# ═══════════════════════════════════════════════════════════════════════════

class EmbeddingService:
    """BM25-based embedding: transform query → score vector."""

    def __init__(self):
        self._model_data = get_model()

    @property
    def bm25(self) -> BM25Okapi | None:
        return self._model_data.get("bm25") if self._model_data else None

    @property
    def questions(self) -> list[str]:
        return self._model_data.get("questions", []) if self._model_data else []

    # ── Fit / Train ─────────────────────────────────────────────────────

    def fit(self, questions: list[str], intents: list[str] | None = None):
        """Train BM25 trên danh sách câu hỏi và lưu cache."""
        if not questions:
            logger.error("[Embedding] fit() called with empty questions")
            return

        corpus = [tokenize(q) for q in questions]
        bm25 = BM25Okapi(corpus)

        self._model_data = {
            "bm25": bm25,
            "questions": questions,
            "intents": intents or [],
        }

        # Save cache
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(_BM25_CACHE, "wb") as f:
            pickle.dump(self._model_data, f)

        global _model
        _model = self._model_data

        logger.info("[Embedding] BM25 fit OK — %d questions, vocab=%d terms",
                     len(questions), len(bm25.idf))

    # ── Embed ───────────────────────────────────────────────────────────

    def embed(self, query: str) -> np.ndarray | None:
        """BM25 scores → vector (N_questions,)."""
        if self.bm25 is None or not query:
            return None
        try:
            tokens = tokenize(query)
            scores = self.bm25.get_scores(tokens)
            return np.array(scores, dtype=np.float32)
        except Exception as e:
            logger.error("[Embedding] BM25 score error: %s", e)
            return None

    def embed_batch(self, queries: list[str]) -> np.ndarray | None:
        """BM25 scores → matrix (N_queries, N_questions)."""
        if self.bm25 is None or not queries:
            return None
        try:
            rows = [self.bm25.get_scores(tokenize(q)) for q in queries]
            return np.array(rows, dtype=np.float32)
        except Exception as e:
            logger.error("[Embedding] BM25 batch error: %s", e)
            return None

    def is_healthy(self) -> bool:
        return self.bm25 is not None


# ── Convenience factory ──────────────────────────────────────────────────
def init_embedding() -> EmbeddingService:
    """Factory: create + return a ready-to-use EmbeddingService."""
    logger.info("[SERVICES] Initialising EmbeddingService (BM25)...")
    service = EmbeddingService()
    if service.is_healthy():
        logger.info("[SERVICES] EmbeddingService ready (%d questions)", len(service.questions))
    else:
        logger.warning("[SERVICES] EmbeddingService NOT available — run train_embeddings.ipynb first")
    return service
