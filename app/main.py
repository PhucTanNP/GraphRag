import logging

from fastapi import FastAPI

from app.api.v1 import router as api_v1_router
from app.api.v1.main_routes import router as main_router
from app.pipeline.orchestrator_v5 import GraphRAGV5

# ── Logging config ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  App Init
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(title="GraphRAG Chatbot", version="5.0")

# ── Chatbot Instance ─────────────────────────────────────────────────────
chatbot = GraphRAGV5()
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
            "neo4j": chatbot.is_healthy(),
            "version": "5.0",
        }
        app.state.health_info = health_info
        logger.info('System startup health: %s', health_info)
    except Exception:
        logger.exception('Failed to determine system health on startup')

