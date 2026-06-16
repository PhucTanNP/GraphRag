"""Pipeline Orchestrator V4 — flow xử lý câu hỏi.

Pipeline Flow (theo đúng thiết kế):
  User Query
  ↓
  Step 1: Embed câu hỏi user (EmbeddingService) — *scaffold, user sẽ implement model sau*
  ↓
  Step 2: Tìm câu hỏi tốt nhất từ QuestionBank (SemanticMatcher)
  ↓
  Step 3: Confidence Check
  ├─ ✅ Cao (≥ threshold) → dùng intent từ câu hỏi matched
  │                           + câu hỏi user gốc (extract size/brand = regex)
  └─ ❌ Thấp (< threshold) → LLM normalize → (intent + câu hỏi chuẩn hoá)
  ↓
  Step 4: Map → Cypher query (CypherMapper)
  ↓
  Step 5: Query Neo4j (Neo4jService)
  ↓
  Step 6: Fill template → Answer (TemplateManager)
  ↓
  Return answer
"""

import time
import logging
import os

from app.services import EmbeddingService, Neo4jService
from app.matching.semantic_matcher import SemanticMatcher
from app.cypher.cypher_mapper import CypherMapper
from app.response.builder import AnswerGenerator
from app.response.normalizer import normalize_data, dedup_data

logger = logging.getLogger(__name__)

# ─── Configuration ──────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 3.0))


class GraphRAGV4:
    """Pipeline V4.

    Usage:
        chatbot = GraphRAGV4()
        answer = chatbot.run("lốp 120/70-17 tốc độ bao nhiêu?")
    """

    def __init__(self):
        # Step 1: Embedding
        self.embedder = EmbeddingService()

        # Step 2: Semantic Matching (tìm câu hỏi tốt nhất)
        self.matcher = SemanticMatcher(embedder=self.embedder)

        # Step 4: Map intent + query → Cypher
        self.cypher_mapper = CypherMapper()

        # Step 5: Neo4j
        self.db = Neo4jService()

        # Step 6: Template-based answer
        self.answer_generator = AnswerGenerator()

        # LLM fallback (normalize + answer)
        from app.fallback.llm_fallback import LLMFallback
        self.llm_fallback = LLMFallback()

        self._built = False

    # ── Public API ──────────────────────────────────────────────────────

    def run(self, query: str) -> str:
        """Run pipeline: Embed → Tìm câu hỏi → Cypher → Neo4j → Template."""
        start = time.time()
        logger.info("=" * 50)
        logger.info("[V4] Query: %s", query)
        logger.info("=" * 50)

        if not query or not query.strip():
            return "Vui lòng nhập câu hỏi."

        # Build question bank embeddings (nếu chưa có)
        if not self._built:
            self.matcher.build()
            self._built = True

        try:
            # ═══════════════════════════════════════════════════════════
            # STEP 1 + 2: Embed → Tìm câu hỏi tốt nhất từ QuestionBank
            # ═══════════════════════════════════════════════════════════
            logger.info("[V4] Step 1+2: Embed → Find best question")
            matched, raw_conf = self.matcher.match_with_threshold(query)

            if matched:
                logger.info("[V4] Best: intent=%s conf=%.3f q=%s",
                            matched["intent"], matched["confidence"], matched["question"])
            else:
                logger.info("[V4] No match (conf=%.3f) — embedding model chưa có?", raw_conf)

            # ═══════════════════════════════════════════════════════════
            # STEP 3: Confidence Check
            #   HIGH → intent từ BM25 + extract entities từ matched question
            #   LOW  → LLM extract entities → map trực tiếp
            # ═══════════════════════════════════════════════════════════
            intent = matched["intent"] if matched else None

            if raw_conf >= CONFIDENCE_THRESHOLD:
                logger.info("[V4] ✅ Cao (%.3f) → intent=%s, matched_q=%s",
                             raw_conf, intent, matched["question"][:60])
                # Dùng matched question (đã chuẩn) để regex extract size/brand
                cypher, params = self.cypher_mapper.map(intent, matched["question"])
            else:
                logger.info("[V4] ❌ Thấp (%.3f) → LLM EXTRACT ENTITIES", raw_conf)
                entities = self.llm_fallback.extract_entities(query)
                if entities.get("intent"):
                    logger.info("[V4] LLM entities → %s", entities)
                    cypher, params = self.cypher_mapper.map_from_entities(
                        intent=entities["intent"],
                        size=entities.get("size"),
                        brand=entities.get("brand"),
                        compare_sizes=entities.get("compare_sizes"),
                    )
                else:
                    logger.info("[V4] LLM extract thất bại → fallback answer")
                    return self.llm_fallback.answer(query)

            if not cypher:
                logger.info("[V4] No Cypher → LLM fallback answer")
                return self.llm_fallback.answer_with_context(query, {"intent": intent})

            # ═══════════════════════════════════════════════════════════
            # STEP 5: Query Neo4j
            # ═══════════════════════════════════════════════════════════
            logger.info("[V4] Step 5: Neo4j\n%s", cypher)
            data = self.db.query(cypher, params=params)

            if not data:
                logger.info("[V4] Neo4j empty → LLM FALLBACK TRIGGERED")
                return self.llm_fallback.answer_with_context(query, {"intent": intent})

            data = dedup_data(data)
            data = normalize_data(data)

            # ═══════════════════════════════════════════════════════════
            # STEP 6: Fill response template → Answer
            # ═══════════════════════════════════════════════════════════
            logger.info("[V4] Step 6: Template → Answer")
            plan_type = intent if intent else "SINGLE"
            return self.answer_generator.generate(query, data, plan={"type": plan_type})

        except Exception as e:
            logger.exception("[V4] Pipeline error: %s", e)
            return self.llm_fallback.answer(query)

    def reset_context(self):
        pass

    # ── Health ──────────────────────────────────────────────────────────

    def is_healthy(self) -> bool:
        try:
            return (
                self.embedder.is_healthy()
                and self.matcher.is_healthy()
                and self.db.ping() is not None
            )
        except Exception:
            return False
