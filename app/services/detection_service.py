from typing import Any

import numpy as np

from app.models.model_manager import model_manager


class DetectionService:
    """
    Handles license-plate detection using the loaded plate-detector model.
    """

    def __init__(self) -> None:
        self.model_manager = model_manager

    def detect(
        self,
        image: np.ndarray,
    ) -> list[dict[str, Any]]:
        """
        Detect license-plate regions in an image.

        Returns a list of detections containing:
        - class_id
        - class_name
        - confidence
        - bounding_box
        """

        if image is None or image.size == 0:
            raise ValueError("Invalid image supplied for detection.")

        detector = self.model_manager.get_plate_detector()

        results = detector.predict(image)

        detections: list[dict[str, Any]] = []

        for class_id, result in enumerate(results):
            box = result.bounding_box

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": result.label,
                    "confidence": float(result.confidence),
                    "bounding_box": {
                        "x1": float(box.x1),
                        "y1": float(box.y1),
                        "x2": float(box.x2),
                        "y2": float(box.y2),
                    },
                }
            )

        return detections


detection_service = DetectionService()
