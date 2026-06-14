"""Template Manager — load & render answer templates theo topic.

Các template .md được lưu trong ``app/templates/answers/``,
hỗ trợ placeholder ``{variable}`` style (format string).

Usage:
    from app.templates import TemplateManager

    tm = TemplateManager()
    html = tm.render("speed", size="120/70-17", brand="DPLUS", speed=120)
    chat_page = tm.load_chat_html()
"""

from __future__ import annotations

import os
import logging
from typing import Any

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = os.path.dirname(os.path.abspath(__file__))
_ANSWERS_DIR = os.path.join(_TEMPLATES_DIR, "answers")
_CHAT_HTML_PATH = os.path.join(_TEMPLATES_DIR, "chat.html")


class TemplateManager:
    """Load và render answer templates từ file .md theo topic.

    Templates được cache sau lần load đầu tiên.
    """

    def __init__(self):
        self._cache: dict[str, str] = {}

    # ── Public API ──────────────────────────────────────────────────────

    def render(self, topic: str, **kwargs: Any) -> str | None:
        """Render template cho topic với các biến ``**kwargs``.

        Args:
            topic: Tên topic (speed, load, price, pressure, max, no_match, list, ...).
            **kwargs: Biến để format vào template.

        Returns:
            String đã render, hoặc None nếu template không tồn tại.
        """
        template = self._load(topic)
        if template is None:
            logger.warning("[Templates] No template for topic=%s", topic)
            return None
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.warning("[Templates] Missing placeholder %s for topic=%s", e, topic)
            return template
        except Exception as e:
            logger.error("[Templates] Render error for topic=%s: %s", topic, e)
            return template

    def load_chat_html(self) -> str:
        """Load file ``chat.html`` — giao diện web chat."""
        try:
            with open(_CHAT_HTML_PATH, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.warning("[Templates] chat.html not found at %s", _CHAT_HTML_PATH)
            return "<h1>Chat interface not found</h1>"
        except Exception as e:
            logger.error("[Templates] Failed to load chat.html: %s", e)
            return "<h1>Error loading chat interface</h1>"

    def list_topics(self) -> list[str]:
        """Liệt kê tất cả topics có template."""
        topics = set(self._cache.keys())
        try:
            topics |= {f.replace(".md", "") for f in os.listdir(_ANSWERS_DIR) if f.endswith(".md")}
        except FileNotFoundError:
            pass
        return sorted(topics)

    # ── Internal ────────────────────────────────────────────────────────

    def _load(self, topic: str) -> str | None:
        """Load template từ cache hoặc đọc từ file."""
        # Check cache
        if topic in self._cache:
            return self._cache[topic]

        # Load from file
        path = os.path.join(_ANSWERS_DIR, f"{topic}.md")
        if not os.path.exists(path):
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            self._cache[topic] = content
            return content
        except Exception as e:
            logger.error("[Templates] Failed to load %s: %s", path, e)
            return None


# ── Singleton ────────────────────────────────────────────────────────────
_template_manager: TemplateManager | None = None


def get_template_manager() -> TemplateManager:
    """Return singleton TemplateManager."""
    global _template_manager
    if _template_manager is None:
        _template_manager = TemplateManager()
    return _template_manager
