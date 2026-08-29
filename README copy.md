# SalinWika Web

A web UI for the existing Cebuano→Tagalog seq2seq translator: a FastAPI
layer over the existing training/inference code, and a SvelteKit frontend.
Deployed and run as two separate services (CORS between them), not bundled
into one process.

## Layout

```
salinwika-web/
├── backend/
│   ├── api/
│   │   ├── server.py              FastAPI app + CORS
│   │   ├── config.py              env-driven settings
│   │   ├── schemas.py             request/response models
│   │   ├── routes/
│   │   │   ├── models.py          GET /models
│   │   │   └── translate.py       POST /translate
│   │   └── services/
│   │       ├── model_registry.py  scans results/ for translatable runs
│   │       └── translation_service.py  wraps utils.inference
│   └── requirements-api.txt
└── frontend/                      SvelteKit app
    └── src/...
```

`backend/api/` is meant to be **copied into your existing project root**,
alongside `main.py`, `utils/`, and `models/` — it imports
`from utils.inference import translate_from_results`, so it needs to sit
where that import resolves.

## How it wires into the existing code

Nothing in `utils/` or `models/` needs to change. `translate_from_results()`
already does exactly what a translate request needs: given a config with
`translation.model_name` and `translation.text` set, it validates the run,
loads that run's saved `configs.json`, vocabularies, and `best_model.pt`,
reconstructs `Seq2Seq`, and returns one translated string.

`translation_service.py` builds that config per request: it loads your
project's `config.yaml` once (for `device`, `translation.max_length`, etc.)
and overrides just `model_name` and `text` with what the API request sent.
The CLI path (`python main.py translation`) is untouched.

`model_registry.py` lists translatable runs by reading `configs.json` and
`test_results.json` under each `results/<name>/` directory — no PyTorch
import needed just to populate a dropdown.

## Running locally

**Backend** (from your project root, with `api/` copied in):

```bash
pip install -r requirements.txt -r backend/requirements-api.txt  # adjust paths as needed
uvicorn api.server:app --reload --port 8000
```

**Frontend**:

```bash
cd frontend
npm install
cp .env.example .env   # PUBLIC_API_BASE_URL=http://localhost:8000
npm run dev
```

Open the printed `localhost:5173` URL. The model dropdown is empty until at
least one run exists under `results/` with both `configs.json` and
`best_model.pt` (i.e. you've run `python main.py training` at least once).

## Configuration

- `SALINWIKA_RESULTS_DIR` (backend) — defaults to `results`.
- `SALINWIKA_CONFIG_PATH` (backend) — defaults to `config.yaml`.
- `SALINWIKA_CORS_ORIGINS` (backend) — comma-separated allowed origins,
  defaults to the SvelteKit dev server.
- `PUBLIC_API_BASE_URL` (frontend) — where the backend is running.

For production, put both behind your normal reverse proxy / hosting setup;
nothing here assumes localhost beyond the defaults above.

## Deliberately left out (for now)

- **Training over HTTP.** Training is long-running and GPU-bound; exposing
  it needs a job queue and progress streaming, which is a separate piece of
  work. The web UI is translate-only; training stays on the CLI/Docker path.
- **Model caching.** `translate_from_results()` reloads weights and
  vocabularies on every request. Fine for getting this working; if latency
  matters later, the next step is splitting `utils/inference.py` into a
  `load_run(model_name)` step (cacheable per model) and a
  `translate(loaded_run, text)` step, and caching the former in
  `translation_service.py`.
- **`bahdanau.py`.** Not wired into `main.py`/`pipelines.py`/`inference.py`
  today (per ARCHITECTURE.md, the current runtime is attention-free), so
  it's not reachable from this API either. If/when it's connected the same
  way `seq2seq.py` is, the model registry and translate route won't need
  changes — the run's `configs.json` already records which architecture
  built it.
