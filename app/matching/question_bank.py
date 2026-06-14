"""Question Bank — quản lý câu hỏi mẫu và embedding của chúng.

Dùng để semantic matching:
  User query → embed → cosine similarity với tất cả câu mẫu → best intent
"""

import os
import pickle
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ─── Question-Intent Mapping ────────────────────────────────────────────
# Mỗi intent có nhiều câu hỏi mẫu (variants) để tăng độ chính xác matching

QUESTION_BANK = {
    "THONG_SO": [
        "lốp 120/70-17 thông số thế nào",
        "lốp 2.50-17 có những thông số gì",
        "cho mình biết thông tin lốp 100/90-18",
        "lốp 110/80-14 chiều rộng bao nhiêu",
        "thông số kỹ thuật lốp 2.75-17",
        "lốp 90/90-14 specs",
        "thông tin lốp 3.00-18",
    ],
    "SPEED": [
        "tốc độ tối đa của lốp 120/70-17",
        "lốp 100/80-14 chạy được bao nhiêu km/h",
        "tốc độ lốp 90/90-14",
        "lốp nào nhanh nhất",
        "lốp 110/70-14 tốc độ bao nhiêu",
        "tốc độ tối đa lốp này là bao nhiêu",
        "lốp này chạy được tối đa bao nhiêu km/h",
    ],
    "LOAD": [
        "tải trọng lốp 120/70-17",
        "lốp 2.50-17 chịu tải bao nhiêu kg",
        "lốp 100/90-18 tải trọng tối đa",
        "lốp nào chịu tải cao nhất",
        "lốp 110/80-14 chở được bao nhiêu kg",
        "tải trọng tối đa của lốp này",
        "lốp này chịu tải được bao nhiêu",
    ],
    "PRICE": [
        "giá lốp 120/70-17 bao nhiêu",
        "lốp 100/80-14 giá bao nhiêu tiền",
        "báo giá lốp 2.50-17",
        "lốp rẻ nhất",
        "lốp 3.00-18 giá",
        "lốp này giá bao nhiêu",
        "cho mình hỏi giá lốp 90/90-14",
    ],
    "COMPARE": [
        "so sánh lốp 120/70-17 và 110/70-17",
        "lốp 2.50-17 vs 2.75-17",
        "khác nhau giữa lốp 90/90-14 và 100/80-14",
        "nên mua lốp 120/70-17 hay 110/70-17",
        "so sánh 2.50 và 2.75",
        "so sánh lốp này với lốp 100/80-14",
    ],
    "PRESSURE": [
        "áp suất lốp 120/70-17",
        "bơm lốp 100/80-14 bao nhiêu kg",
        "áp suất tiêu chuẩn lốp 2.50-17",
        "lốp 90/90-14 bơm bao nhiêu psi",
        "áp suất lốp bao nhiêu",
    ],
    "BRAND": [
        "lốp DPLUS có tốt không",
        "thương hiệu lốp nào bền nhất",
        "lốp IRC giá bao nhiêu",
        "lốp MAXXIS chất lượng",
        "các hãng lốp xe máy",
        "lốp hãng nào tốt",
    ],
    "MAX_LOAD": [
        "lốp nào chịu tải cao nhất",
        "lốp chịu tải tốt nhất",
        "lốp nào tải trọng lớn nhất",
        "lốp chở nặng tốt nhất",
    ],
    "MAX_SPEED": [
        "lốp nào nhanh nhất",
        "lốp tốc độ cao nhất",
        "lốp nào chạy nhanh nhất",
    ],
    "DRAINAGE": [
        "lốp nào thoát nước tốt",
        "lốp đi mua tốt",
        "lốp chống trượt nước",
        "lốp phù hợp đường ướt",
    ],
    "DURABILITY": [
        "lốp nào bền nhất",
        "lốp độ bền cao",
        "lốp đi được nhiều km",
    ],
    "TUBE": [
        "lốp có săm không",
        "lốp không săm",
        "lốp tubeless",
        "lốp cần săm",
    ],
    "SERVICE": [
        "đặt lịch thay lốp",
        "dịch vụ thay lốp",
        "phi lắp đặt bao nhiêu",
        "lắp lốp tận nơi",
    ],
}

# ─── Question Bank Manager ──────────────────────────────────────────────

class QuestionBank:
    """Manage question bank: build embeddings, save/load cache.

    Usage:
        bank = QuestionBank()
        bank.build()  # Build embeddings from QUESTION_BANK
        result = bank.match(user_query_vector)
        # → {"intent": "SPEED", "confidence": 0.85, "question": "..."}
    """

    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), "..", "Embeding_vector"
        )
        self._questions: list[str] = []
        self._intents: list[str] = []
        self._embeddings: np.ndarray | None = None
        self._built = False

    # ── Public API ──────────────────────────────────────────────────────

    def build(self, embedder=None) -> None:
        """Build or load question embeddings.

        Args:
            embedder: QueryEmbedder instance. If None, lazy-import.
        """
        if self._built:
            return

        cache_path_npy = os.path.join(self.cache_dir, "question_embeddings.npy")
        cache_path_pkl = os.path.join(self.cache_dir, "question_bank.pkl")

        # Try loading from cache
        if os.path.exists(cache_path_npy) and os.path.exists(cache_path_pkl):
            try:
                self._embeddings = np.load(cache_path_npy)
                with open(cache_path_pkl, "rb") as f:
                    data = pickle.load(f)
                self._questions = data["questions"]
                self._intents = data["intents"]
                self._built = True
                logger.info(
                    f"[QuestionBank] Loaded {len(self._questions)} questions, "
                    f"{len(set(self._intents))} intents from cache"
                )
                return
            except Exception as e:
                logger.warning(f"[QuestionBank] Cache load failed: {e}")

        # Build from QUESTION_BANK
        self._questions = []
        self._intents = []
        for intent, qs in QUESTION_BANK.items():
            for q in qs:
                self._questions.append(q)
                self._intents.append(intent)

        if not self._questions:
            logger.error("[QuestionBank] No questions defined!")
            return

        # Embed all questions
        if embedder is None:
            from app.services import EmbeddingService
            embedder = EmbeddingService()

        vecs = embedder.embed_batch(self._questions)
        if vecs is None:
            logger.error("[QuestionBank] Embedding failed!")
            return

        self._embeddings = vecs

        # Persist cache
        os.makedirs(self.cache_dir, exist_ok=True)
        try:
            np.save(cache_path_npy, self._embeddings)
            with open(cache_path_pkl, "wb") as f:
                pickle.dump({
                    "questions": self._questions,
                    "intents": self._intents,
                }, f)
            logger.info(
                f"[QuestionBank] Built and cached {len(self._questions)} questions, "
                f"{len(set(self._intents))} intents"
            )
        except Exception as e:
            logger.warning(f"[QuestionBank] Cache persist failed: {e}")

        self._built = True

    def match(self, query_vector: np.ndarray, threshold: float = 0.40) -> dict | None:
        """Match query vector against question bank.

        Args:
            query_vector: Normalized query embedding (384,).
            threshold: Minimum cosine similarity threshold.

        Returns:
            Dict with keys: intent, confidence, question
            None if no match above threshold.
        """
        if self._embeddings is None or query_vector is None:
            return None

        # Cosine similarity = dot product (normalized vectors)
        scores = np.dot(self._embeddings, query_vector)

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score < threshold:
            return None

        return {
            "intent": self._intents[best_idx],
            "confidence": round(best_score, 4),
            "question": self._questions[best_idx],
        }

    def match_grouped(self, query_vector: np.ndarray, threshold: float = 0.30) -> dict | None:
        """Match with intent-level scoring (average across all questions per intent).

        Args:
            query_vector: Normalized query embedding (384,).
            threshold: Minimum average confidence threshold.

        Returns:
            Dict with keys: intent, confidence, top_question
        """
        if self._embeddings is None or query_vector is None:
            return None

        scores = np.dot(self._embeddings, query_vector)

        # Group by intent
        intent_scores: dict[str, list[tuple[float, str]]] = {}
        for i, intent in enumerate(self._intents):
            if intent not in intent_scores:
                intent_scores[intent] = []
            intent_scores[intent].append((float(scores[i]), self._questions[i]))

        # Average per intent
        intent_avg = {
            intent: (sum(s for s, _ in pairs) / len(pairs), pairs)
            for intent, pairs in intent_scores.items()
        }

        best_intent = max(intent_avg, key=lambda k: intent_avg[k][0])
        best_score, best_pairs = intent_avg[best_intent]

        if best_score < threshold:
            return None

        # Get top question within that intent
        best_question = max(best_pairs, key=lambda p: p[0])[1]

        return {
            "intent": best_intent,
            "confidence": round(best_score, 4),
            "question": best_question,
        }

    def get_all_questions(self) -> list[str]:
        return self._questions.copy()

    def get_all_intents(self) -> list[str]:
        return self._intents.copy()

    def is_healthy(self) -> bool:
        return self._built and self._embeddings is not None
