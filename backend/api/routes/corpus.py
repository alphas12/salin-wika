import csv

from fastapi import APIRouter

from ..config import settings
from ..schemas import CorpusSample

router = APIRouter()


@router.get("/corpus/samples", response_model=list[CorpusSample])
def get_corpus_samples(limit: int = 6):
    if limit < 1:
        limit = 1

    corpus_path = settings.CORPUS_PATH
    if not corpus_path.is_file():
        return []

    samples = []
    with open(corpus_path, encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            source_text = (
                row.get("cebuano")
                or row.get("source_text")
                or row.get("source_sentence")
                or ""
            ).strip()
            target_text = (
                row.get("tagalog")
                or row.get("target_text")
                or row.get("target_sentence")
                or ""
            ).strip()
            if not source_text or not target_text:
                continue
            samples.append({"source_text": source_text, "target_text": target_text})
            if len(samples) >= limit:
                break

    return samples