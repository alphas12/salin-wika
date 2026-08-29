# AI Amended: Small FastAPI app that exposes the existing translation
# pipeline over HTTP. Run from the project root (same level as main.py) so
# the `utils` and `models` imports it relies on resolve correctly:
#
#   uvicorn api.server:app --reload --port 8000
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routes import corpus, models, translate

app = FastAPI(title="SalinWika API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_origin_regex=settings.CORS_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(models.router, tags=["models"])
app.include_router(corpus.router, tags=["corpus"])
app.include_router(translate.router, tags=["translate"])


@app.get("/health")
def health():
    return {"status": "ok"}
