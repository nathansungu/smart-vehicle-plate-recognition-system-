from io import BytesIO

import cv2
import numpy as np
from PIL import Image


class ImagePreprocessor:
    """
    Handles validation and preprocessing of images before
    they are passed to the detection and OCR services.
    """

    ALLOWED_FORMATS = {
        "JPEG",
        "PNG",
        "WEBP",
    }

    def validate_image(self, image_bytes: bytes) -> bool:
        """
        Validate that the supplied bytes represent a supported image.
        """

        if not image_bytes:
            raise ValueError("Image data is empty.")

        try:
            image = Image.open(BytesIO(image_bytes))
            image.verify()

            return True

        except Exception as exc:
            raise ValueError(
                f"Invalid image: {exc}"
            ) from exc

    def bytes_to_array(self, image_bytes: bytes) -> np.ndarray:
        """
        Convert image bytes into an OpenCV/Numpy image.
        """

        self.validate_image(image_bytes)

        buffer = np.frombuffer(image_bytes, dtype=np.uint8)

        image = cv2.imdecode(
            buffer,
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise ValueError(
                "Unable to decode image."
            )

        return image

    def resize(
        self,
        image: np.ndarray,
        max_width: int = 1920,
        max_height: int = 1080,
    ) -> np.ndarray:
        """
        Resize an image while maintaining its aspect ratio.
        """

        height, width = image.shape[:2]

        if width <= max_width and height <= max_height:
            return image

        scale = min(
            max_width / width,
            max_height / height,
        )

        new_width = int(width * scale)
        new_height = int(height * scale)

        return cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

    def preprocess(
        self,
        image_bytes: bytes,
    ) -> np.ndarray:
        """
        Complete preprocessing pipeline.
        """

        image = self.bytes_to_array(image_bytes)

        image = self.resize(image)

        return image


image_preprocessor = ImagePreprocessor()