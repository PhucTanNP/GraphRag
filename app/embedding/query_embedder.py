"""Backward-compat re-export. Prefer ``from app.services import EmbeddingService``."""
from app.services.embedding_service import EmbeddingService  # noqa: F401


class QueryEmbedder(EmbeddingService):
    """Legacy alias. Use ``EmbeddingService`` instead."""
    pass
