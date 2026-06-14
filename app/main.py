import logging

from fastapi import FastAPI

from app.api.v1 import router as api_v1_router
from app.api.v1.main_routes import router as main_router
from app.pipeline.orchestrator_v4 import GraphRAGV4

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  App Init
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(title="GraphRAG Chatbot", version="4.0")

# ── Chatbot Instance ─────────────────────────────────────────────────────
chatbot = GraphRAGV4()
app.state.chatbot = chatbot

# ── Routers ──────────────────────────────────────────────────────────────
app.include_router(main_router)
app.include_router(api_v1_router)

# ═══════════════════════════════════════════════════════════════════════════
#  Startup Events
# ═══════════════════════════════════════════════════════════════════════════


@app.on_event("startup")
def check_system_on_startup():
    try:
        health_info = {
            "embedder": chatbot.embedder.is_healthy() if hasattr(chatbot, 'embedder') else False,
            "matcher": chatbot.matcher.is_healthy() if hasattr(chatbot, 'matcher') else False,
            "neo4j": chatbot.db.is_healthy() if hasattr(chatbot, 'db') else False,
        }
        try:
            chatbot.matcher.build()
            health_info["question_bank"] = True
        except Exception:
            health_info["question_bank"] = False
        app.state.health_info = health_info
        logger.info('System startup health: %s', health_info)
    except Exception:
        logger.exception('Failed to determine system health on startup')

