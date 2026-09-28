import re
from typing import Any

import numpy as np
from fast_plate_ocr import LicensePlateRecognizer

from app.core.config import settings


class OCRService:
    """
    Handles optical character recognition for
    vehicle plates and serial numbers.
    """

    def __init__(self) -> None:
        """
        Initialize the plate-recognition model once.

        NOTE: This model is trained specifically on cropped license plates.
        """

        self.reader = LicensePlateRecognizer(settings.PLATE_OCR_MODEL)

    def extract_text(
        self,
        image: np.ndarray,
    ) -> dict[str, Any]:
        """
        Extract text from a cropped plate image.
        """

        if image is None or image.size == 0:
            raise ValueError("Invalid image supplied to OCR.")

        predictions = self.reader.run(image, return_confidence=True)

        if not predictions:
            return {"text": "", "confidence": 0.0}

        prediction = predictions[0]

        if prediction.char_probs is not None and len(prediction.char_probs) > 0:
            confidence = float(np.mean(prediction.char_probs))
        else:
            confidence = 0.0

        text = self.clean_text(prediction.plate)

        # Ignore low-confidence reads instead of returning noise.
        if confidence < settings.OCR_CONFIDENCE:
            return {"text": "", "confidence": confidence}

        return {"text": text, "confidence": confidence}

    def clean_text(
        self,
        text: str,
    ) -> str:
        """
        Clean OCR output by removing characters
        that are not valid for our recognition use case.
        """

        text = text.upper()

        text = re.sub(
            r"[^A-Z0-9]",
            "",
            text,
        )

        return text.strip()

    def extract(
        self,
        image: np.ndarray,
    ) -> dict[str, Any]:
        """
        Perform OCR and return a structured result.
        """

        result = self.extract_text(image)

        return {
            "text": result["text"],
            "detected": bool(result["text"]),
        }


ocr_service = OCRService()
