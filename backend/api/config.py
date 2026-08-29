# AI Amended: Central settings for the API layer, overridable via environment
# variables so dev/staging/prod can point at different results/ dirs and origins.
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings:
    # Same results/ directory main.py already writes to.
    RESULTS_DIR = Path(os.getenv("SALINWIKA_RESULTS_DIR", str(PROJECT_ROOT / "results")))

    # Corpus CSV used as the input dataset for the web UI and training flows.
    _corpus_override = os.getenv("SALINWIKA_CORPUS_PATH")
    if _corpus_override:
        CORPUS_PATH = Path(_corpus_override)
    else:
        _corpus_candidates = [
            PROJECT_ROOT / "corpus" / "cebtag_bible_31k.csv",
            PROJECT_ROOT / "corpus" / "cebtag_105k.csv",
        ]
        CORPUS_PATH = next((path for path in _corpus_candidates if path.is_file()), _corpus_candidates[0])

    # Base config.yaml used for device/decoding settings that aren't
    # overridden per-request (e.g. translation.max_length, lowercasing).
    BASE_CONFIG_PATH = Path(os.getenv("SALINWIKA_CONFIG_PATH", str(PROJECT_ROOT / "config.yaml")))

    # Comma-separated list of allowed frontend origins.
    CORS_ORIGINS = os.getenv(
        "SALINWIKA_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")

    # Allow any local development port for Vite preview/dev.
    CORS_ORIGIN_REGEX = os.getenv(
        "SALINWIKA_CORS_ORIGIN_REGEX",
        r"^https?://(localhost|127\.0\.0\.1):(\d+)$",
    )


settings = Settings()
