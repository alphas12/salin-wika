from typing import Optional

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    name: str
    bidirectional: bool
    peeky: bool
    bleu: Optional[float] = None


class TranslateRequest(BaseModel):
    model_name: str = Field(..., description="Run name under results/, e.g. 'peeky-v3'.")
    text: str = Field(..., min_length=1, description="Cebuano source text to translate.")


class TranslateResponse(BaseModel):
    model_name: str
    source_text: str
    translated_text: str


class CorpusSample(BaseModel):
    source_text: str
    target_text: str
