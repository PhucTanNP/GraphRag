"""Main routes — endpoints trực tiếp cho frontend và monitoring.

Tách riêng khỏi ``main.py`` để giữ file đó gọn nhẹ, chỉ làm app factory.
"""
import re
import logging

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from app.templates import get_template_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Main"])
templates = get_template_manager()


def markdown_to_html(text: str) -> str:
    """Convert simple Markdown to HTML for chat display."""
    if not text:
        return text

    # Bold **text**
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Italic *text*
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)

    lines = text.split('\n')
    in_list = False
    result: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('- '):
            if not in_list:
                result.append('<ul>')
                in_list = True
            result.append(f'<li>{stripped[2:]}</li>')
        else:
            if in_list:
                result.append('</ul>')
                in_list = False
            if stripped:
                result.append(f'<p>{stripped}</p>')

    if in_list:
        result.append('</ul>')

    return '\n'.join(result)


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
def home():
    """Trang chat HTML."""
    return templates.load_chat_html()


@router.get("/query")
def query(q: str, request: Request):
    """Truy vấn GraphRAG pipeline (GET, đơn giản)."""
    chatbot = request.app.state.chatbot
    try:
        result = chatbot.run(q)
        html = markdown_to_html(result)
        return {"result": html}
    except Exception as e:
        logger.exception("Query failed: %s", q)
        return {"error": str(e)}


@router.get("/health")
def health(request: Request):
    """Health check — trạng thái các component."""
    health_info = getattr(request.app.state, "health_info", None)
    if health_info is None:
        chatbot = request.app.state.chatbot
        try:
            health_info = {
                "embedder": chatbot.embedder.is_healthy(),
                "matcher": chatbot.matcher.is_healthy(),
                "neo4j": chatbot.db.is_healthy(),
            }
        except Exception:
            health_info = {"error": "Health check failed"}
    return JSONResponse(content={
        "status": "ok",
        "pipeline": "v4",
        "components": health_info,
    })


@router.post("/reset")
def reset_context(request: Request):
    """Reset conversation (no-op trong V4 pipeline mới)."""
    chatbot = request.app.state.chatbot
    chatbot.reset_context()
    return {"status": "OK"}
