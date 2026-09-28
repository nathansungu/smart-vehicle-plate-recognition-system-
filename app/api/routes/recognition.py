from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.dependencies import require_api_key
from app.schemas.response import RecognitionResponse
from app.services.recognition_service import recognition_service


router = APIRouter(
    prefix="/recognition",
    tags=["Recognition"],
    dependencies=[Depends(require_api_key)],
)


@router.post(
    "/",
    response_model=RecognitionResponse,
    status_code=status.HTTP_200_OK,
)
async def recognize(
    file: UploadFile = File(...),
) -> RecognitionResponse:
    """
    Receive an image and attempt to recognize
    a vehicle plate or serial number.
    """

    if not file.content_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to determine file type.",
        )

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                "Unsupported image format. "
                "Use JPEG, PNG, or WEBP."
            ),
        )

    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded image is empty.",
            )

        result = recognition_service.recognize(
            image_bytes
        )

        return RecognitionResponse(**result)

    except HTTPException:
        raise

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recognition failed: {exc}",
        ) from exc