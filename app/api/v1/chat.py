"""Chat API — endpoint cho hội thoại với GraphRAG pipeline.

Backend (Node.js) gọi:
  POST /api/v1/chat
  { message: "...", history: [...] }

Trả về:
  { result: "..." }
"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from app.pipeline.orchestrator import GraphRAGv3
from app.metrics import request_counter

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])


# ── Request / Response models ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="Nội dung tin nhắn")
    history: list = Field(default=[], description="Lịch sử hội thoại (chưa dùng, để tương thích)")


class ChatResponse(BaseModel):
    result: str
    # Có thể mở rộng: actions, sources, ...


# ── Dependency: lấy instance chatbot (singleton) ────────────────────────────

_chatbot_instance = None


def get_chatbot():
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = GraphRAGv3()
    return _chatbot_instance


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, chatbot: GraphRAGv3 = Depends(get_chatbot)):
    """Gửi tin nhắn chat và nhận phản hồi từ GraphRAG pipeline.

    - Dùng GET /query?q=... cho truy vấn đơn giản (hỗ trợ từ main.py)
    - Dùng POST /api/v1/chat cho tích hợp backend chính thức
    """
    try:
        # Track metrics
        if request_counter is not None:
            try:
                request_counter.labels(endpoint='/api/v1/chat').inc()
            except Exception:
                pass

        message = req.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        # Gọi pipeline GraphRAG
        result = chatbot.run(message)

        return ChatResponse(result=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chat API error")
        raise HTTPException(status_code=500, detail=str(e))
