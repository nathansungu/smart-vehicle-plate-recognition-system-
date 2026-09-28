from typing import Optional

from open_image_models import LicensePlateDetector

from app.core.config import settings


class ModelManager:
    """
    Manages AI models used by the recognition service.
    """

    def __init__(self) -> None:
        self._plate_detector: Optional[LicensePlateDetector] = None

    def load_plate_detector(self) -> LicensePlateDetector:
        """
        Load the license-plate detector once and keep it in memory.
        """

        if self._plate_detector is not None:
            return self._plate_detector

        model_name = settings.PLATE_DETECTOR_MODEL

        # print(f"Loading license-plate detector: {model_name}")

        try:
            self._plate_detector = LicensePlateDetector(
                detection_model=model_name,
                conf_thresh=settings.DETECTION_CONFIDENCE,
            )

            print("License-plate detector loaded successfully.")

            return self._plate_detector

        except Exception as exc:
            self._plate_detector = None

            raise RuntimeError(
                f"Failed to load license-plate detector '{model_name}': {exc}"
            ) from exc

    def get_plate_detector(self) -> LicensePlateDetector:
        """
        Return the loaded license-plate detector.
        """

        if self._plate_detector is None:
            return self.load_plate_detector()

        return self._plate_detector

    def is_plate_detector_loaded(self) -> bool:
        """
        Check whether the license-plate detector is currently loaded.
        """

        return self._plate_detector is not None

    def unload_plate_detector(self) -> None:
        """
        Release the license-plate detector.
        """

        self._plate_detector = None


model_manager = ModelManager()
