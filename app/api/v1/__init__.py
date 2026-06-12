"""API v1 router — tổng hợp tất cả endpoints của GraphRag."""
from fastapi import APIRouter
from app.api.v1.chat import router as chat_router
from app.api.v1.detect import router as detect_router

router = APIRouter(prefix="/api/v1")

# Chat
router.include_router(chat_router)

# Inspect / Detect
router.include_router(detect_router)

__all__ = ["router"]
