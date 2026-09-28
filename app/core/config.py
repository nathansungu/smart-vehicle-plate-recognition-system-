from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root
BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Vehicle Recognition API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Plate detector
    PLATE_DETECTOR_MODEL: str = "yolo-v9-t-384-license-plate-end2end"
    DETECTION_CONFIDENCE: float = 0.40

    # Image processing
    MAX_IMAGE_SIZE_MB: int = 10
    ALLOWED_IMAGE_TYPES: str = "image/jpeg,image/png,image/webp"

    # OCR
    PLATE_OCR_MODEL: str = "cct-xs-v1-global-model"
    OCR_CONFIDENCE: float = 0.50

    # Security
    API_KEY: str = ""

    # Storage
    INPUT_DIR: str = str(BASE_DIR / "storage" / "input")
    OUTPUT_DIR: str = str(BASE_DIR / "storage" / "output")
    FAILED_DIR: str = str(BASE_DIR / "storage" / "failed")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


settings = Settings()