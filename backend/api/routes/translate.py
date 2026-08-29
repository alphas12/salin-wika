from fastapi import APIRouter, HTTPException

from ..config import settings
from ..schemas import TranslateRequest, TranslateResponse

router = APIRouter()


@router.post("/translate", response_model=TranslateResponse)
def post_translate(payload: TranslateRequest):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    try:
        from ..services.translation_service import ModelNotFoundError, translate
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Translation dependencies are not installed on this backend.",
        ) from exc

    try:
        translated = translate(
            model_name=payload.model_name,
            text=text,
            results_dir=settings.RESULTS_DIR,
            base_config_path=settings.BASE_CONFIG_PATH,
        )
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # model loading / decoding failures
        raise HTTPException(status_code=500, detail=f"Translation failed: {exc}") from exc

    return TranslateResponse(
        model_name=payload.model_name,
        source_text=text,
        translated_text=translated,
    )
