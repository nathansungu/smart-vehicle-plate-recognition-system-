from typing import Any

from app.processing.crop import image_cropper
from app.processing.image_preprocessor import image_preprocessor
from app.processing.image_quality import image_quality_checker
from app.services.detection_service import detection_service
from app.services.ocr_service import ocr_service


class RecognitionService:
    """
    Coordinates the complete vehicle plate/serial recognition pipeline.
    """

    def recognize(self, image_bytes: bytes) -> dict[str, Any]:
      
        # --------------------------------------------------
        # 1. Preprocess image
        # --------------------------------------------------

        image = image_preprocessor.preprocess(
            image_bytes
        )

        # --------------------------------------------------
        # 2. Check image quality
        # --------------------------------------------------

        quality = image_quality_checker.check(
            image
        )

        if not quality["quality_ok"]:
            return {
                "success": False,
                "message": "Image quality is insufficient.",
                "quality": quality,
                "detections": [],
                "recognitions": [],
            }

        # --------------------------------------------------
        # 3. Detect plate / serial region
        # --------------------------------------------------

        detections = detection_service.detect(
            image
        )

        if not detections:
            return {
                "success": False,
                "message": "No plate or serial region detected.",
                "quality": quality,
                "detections": [],
                "recognitions": [],
            }

        # --------------------------------------------------
        # 4. Crop detected regions
        # --------------------------------------------------

        recognitions = []

        for detection in detections:

            bounding_box = detection["bounding_box"]

            cropped = image_cropper.crop(
                image,
                bounding_box,
            )

            # --------------------------------------------------
            # 5. OCR
            # --------------------------------------------------

            ocr_result = ocr_service.extract(
                cropped
            )

            recognitions.append(
                {
                    "detection": detection,
                    "ocr": ocr_result,
                }
            )

        # --------------------------------------------------
        # 6. Return complete result
        # --------------------------------------------------

        successful_recognitions = [
            recognition
            for recognition in recognitions
            if recognition["ocr"]["detected"]
        ]

        return {
            "success": bool(successful_recognitions),
            "message": (
                "Recognition completed."
                if successful_recognitions
                else "Region detected but no text was recognized."
            ),
            "quality": quality,
            "detections": detections,
            "recognitions": recognitions,
        }


recognition_service = RecognitionService()