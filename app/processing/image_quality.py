import cv2
import numpy as np


class ImageQualityChecker:
    """
    Evaluates whether an image is suitable for recognition.
    """

    def __init__(
        self,
        min_width: int = 320,
        min_height: int = 240,
        min_brightness: float = 40.0,
        max_brightness: float = 220.0,
        min_sharpness: float = 100.0,
    ) -> None:
        self.min_width = min_width
        self.min_height = min_height
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_sharpness = min_sharpness

    def check(self, image: np.ndarray) -> dict:
        """
        Check the overall quality of an image.

        Returns information about:
        - resolution
        - brightness
        - sharpness
        - whether the image is suitable
        """

        if image is None or image.size == 0:
            raise ValueError("Invalid image supplied for quality check.")

        height, width = image.shape[:2]

        resolution_ok = (
            width >= self.min_width
            and height >= self.min_height
        )

        grayscale = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        brightness = float(np.mean(grayscale))

        brightness_ok = (
            self.min_brightness
            <= brightness
            <= self.max_brightness
        )

        sharpness = float(
            cv2.Laplacian(
                grayscale,
                cv2.CV_64F,
            ).var()
        )

        sharpness_ok = sharpness >= self.min_sharpness

        quality_ok = (
            resolution_ok
            and brightness_ok
            and sharpness_ok
        )

        reasons = []

        if not resolution_ok:
            reasons.append(
                "Image resolution is too low."
            )

        if not brightness_ok:
            if brightness < self.min_brightness:
                reasons.append(
                    "Image is too dark."
                )
            else:
                reasons.append(
                    "Image is too bright."
                )

        if not sharpness_ok:
            reasons.append(
                "Image is too blurry."
            )

        return {
            "quality_ok": quality_ok,
            "width": width,
            "height": height,
            "brightness": round(brightness, 2),
            "sharpness": round(sharpness, 2),
            "reasons": reasons,
        }


image_quality_checker = ImageQualityChecker()