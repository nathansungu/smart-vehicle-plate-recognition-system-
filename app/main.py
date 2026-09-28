from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import settings
from app.models.model_manager import model_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown lifecycle.
    """

    print("Starting Vehicle Recognition API...")

    try:
        model_manager.load_plate_detector()
    except RuntimeError as exc:
        print(
            "WARNING: Plate detector could not be loaded. "
            f"Recognition will not be available. ({exc})"
        )

    yield

    # Release model resources during shutdown.
    model_manager.unload_plate_detector()

    print("Vehicle Recognition API stopped.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "AI service for vehicle plate and "
        "serial number recognition."
    ),
    lifespan=lifespan,
)


app.include_router(api_router)


@app.get("/")
async def root():
    return {
        "service": settings.APP_NAME,
        "status": "running",
        "version": settings.APP_VERSION,
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "plate_detector_loaded": model_manager.is_plate_detector_loaded(),
    }