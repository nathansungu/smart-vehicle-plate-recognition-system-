from fastapi import APIRouter

from app.api.routes.recognition import router as recognition_router


api_router = APIRouter(
    prefix="/api",
)

api_router.include_router(
    recognition_router,
)