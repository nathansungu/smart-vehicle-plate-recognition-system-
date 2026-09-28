from pydantic import BaseModel, Field

from app.schemas.recognition import (
    DetectionResult,
    RecognitionResult,
)


class QualityResult(BaseModel):
    quality_ok: bool
    width: int
    height: int
    brightness: float
    sharpness: float
    reasons: list[str]


class RecognitionResponse(BaseModel):
    success: bool
    message: str

    quality: QualityResult

    detections: list[DetectionResult] = Field(
        default_factory=list
    )

    recognitions: list[RecognitionResult] = Field(
        default_factory=list
    )