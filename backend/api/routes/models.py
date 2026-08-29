from typing import List

from fastapi import APIRouter

from ..config import settings
from ..schemas import ModelInfo
from ..services.model_registry import list_models

router = APIRouter()


@router.get("/models", response_model=List[ModelInfo])
def get_models():
    """List runs under results/ that have enough saved artifacts to
    translate with. Used by the frontend to populate the model picker."""
    return list_models(settings.RESULTS_DIR)
