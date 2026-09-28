from typing import Any

import numpy as np


class ImageCropper:
    """
    Extracts detected regions from images using YOLO bounding boxes.
    """

    def crop(
        self,
        image: np.ndarray,
        bounding_box: dict[str, Any],
    ) -> np.ndarray:
        """
        Crop an image using a bounding box.

        Expected bounding box format:

        {
            "x1": 100,
            "y1": 50,
            "x2": 500,
            "y2": 200
        }
        """

        if image is None or image.size == 0:
            raise ValueError("Invalid image supplied for cropping.")

        required_keys = {"x1", "y1", "x2", "y2"}

        if not required_keys.issubset(bounding_box):
            raise ValueError(
                "Bounding box must contain x1, y1, x2 and y2."
            )

        height, width = image.shape[:2]

        x1 = max(0, int(bounding_box["x1"]))
        y1 = max(0, int(bounding_box["y1"]))
        x2 = min(width, int(bounding_box["x2"]))
        y2 = min(height, int(bounding_box["y2"]))

        if x1 >= x2 or y1 >= y2:
            raise ValueError("Invalid bounding box coordinates.")

        cropped = image[y1:y2, x1:x2]

        if cropped.size == 0:
            raise ValueError("Cropping produced an empty image.")

        return cropped


image_cropper = ImageCropper()