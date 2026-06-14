"""Services — khởi tạo tập trung các core service.

Tiện ích:
  - ``init_all()``: init toàn bộ service cùng lúc
  - Import trực tiếp từng service class để dùng riêng lẻ

Usage:
    from app.services import EmbeddingService, Neo4jService, LLMService
    from app.services import init_all

    services = init_all()
    services["embedding"].embed("...")
    services["neo4j"].query("...")
    services["llm"].chat("...")
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__all__ = [
    "EmbeddingService",
    "Neo4jService",
    "LLMService",
    "Neo4jClient",
    "LLMClient",
    "get_model",
    "get_db",
    "get_llm",
    "init_embedding",
    "init_neo4j",
    "init_llm",
    "init_all",
]


def __getattr__(name):
    """Lazy import — only load what's actually needed."""
    _imports = {
        # Embedding
        "EmbeddingService": ("app.services.embedding_service", "EmbeddingService"),
        "init_embedding": ("app.services.embedding_service", "init_embedding"),
        "get_model": ("app.services.embedding_service", "get_model"),
        # Neo4j
        "Neo4jClient": ("app.services.neo4j_service", "Neo4jClient"),
        "Neo4jService": ("app.services.neo4j_service", "Neo4jService"),
        "init_neo4j": ("app.services.neo4j_service", "init_neo4j"),
        "get_db": ("app.services.neo4j_service", "get_db"),
        # LLM
        "LLMService": ("app.services.llm_service", "LLMService"),
        "init_llm": ("app.services.llm_service", "init_llm"),
        "get_llm": ("app.services.llm_service", "get_llm"),
        "LLMClient": ("app.services.llm_service", "LLMClient"),
    }
    if name in _imports:
        mod_path, attr = _imports[name]
        from importlib import import_module
        mod = import_module(mod_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def init_all() -> dict:
    """Initialize all core services at once.

    Returns:
        Dictionary with keys ``embedding``, ``neo4j``, ``llm``.
    """
    logger.info("=" * 60)
    logger.info("[SERVICES] Initialising all services...")
    logger.info("=" * 60)

    services = {
        "embedding": init_embedding(),
        "neo4j": init_neo4j(),
        "llm": init_llm(),
    }

    logger.info("=" * 60)
    logger.info("[SERVICES] All services initialised")
    logger.info("=" * 60)

    return services


def init_all() -> Dict[str, object]:
    """Initialize all core services at once.

    Returns:
        Dictionary with keys ``embedding``, ``neo4j``, ``llm``.
    """
    logger.info("=" * 60)
    logger.info("[SERVICES] Initialising all services...")
    logger.info("=" * 60)

    services = {
        "embedding": init_embedding(),
        "neo4j": init_neo4j(),
        "llm": init_llm(),
    }

    logger.info("=" * 60)
    logger.info("[SERVICES] All services initialised")
    logger.info("=" * 60)

    return services
