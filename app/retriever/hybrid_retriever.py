from app.retriever.mapper import Mapper
from app.retriever.embedding_matcher import EmbeddingMatcher
import logging
try:
    from opentelemetry import trace as ot_trace
except Exception:
    ot_trace = None

logger = logging.getLogger(__name__)

class HybridRetriever:

    def __init__(self):
        self.mapper = Mapper()
        self.embed = EmbeddingMatcher()
        # indicate whether EmbeddingMatcher loaded a FAISS index
        self.uses_faiss = getattr(self.embed, 'faiss_index', None) is not None
        if self.uses_faiss:
            logger.info('HybridRetriever: using FAISS index for semantic matching')

    def retrieve(self, query):
        # tracing span for retrieval
        if ot_trace is not None:
            tracer = ot_trace.get_tracer(__name__)
            span_ctx = tracer.start_as_current_span('retriever.retrieve', attributes={'query.length': len(query) if query else 0})
            span_ctx.__enter__()
        else:
            span_ctx = None

        try:
            mapped = self.mapper.map(query)
            semantic = self.embed.match(query)
        finally:
            if span_ctx is not None:
                try:
                    # set whether faiss used
                    span = ot_trace.get_tracer(__name__).get_current_span()
                    if span is not None:
                        span.set_attribute('retriever.uses_faiss', bool(self.uses_faiss))
                except Exception:
                    pass
                span_ctx.__exit__(None, None, None)

        # =========================
        # MERGE CONTEXT (QUAN TRỌNG)
        # =========================
        enriched = {
            "mapped": mapped,
            "semantic": semantic,
            "has_size": any(m.get("column") == "size" for m in mapped),
            "uses_faiss": self.uses_faiss
        }

        return enriched