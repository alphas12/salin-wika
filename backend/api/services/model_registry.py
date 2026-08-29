# AI Amended: List translation-ready runs by inspecting results/<name>/ the
# same way ARCHITECTURE.md describes each run's artifacts, without importing
# torch or touching model weights — just the JSON metadata.
import json
from pathlib import Path


def _read_json(path: Path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def list_models(results_dir: Path):
    """Return metadata for every run under results_dir that has enough
    artifacts to translate with (configs.json + best_model.pt)."""
    if not results_dir.exists():
        return []

    models = []
    for run_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        configs_path = run_dir / "configs.json"
        weights_path = run_dir / "best_model.pt"

        if not (configs_path.exists() and weights_path.exists()):
            continue

        run_config = _read_json(configs_path)
        model_config = run_config.get("model", {})

        bleu = None
        test_results_path = run_dir / "test_results.json"
        if test_results_path.exists():
            test_results = _read_json(test_results_path)
            # CODE_GUIDE.md: evaluate_bleu() writes corpus BLEU into
            # test_results.json alongside loss/perplexity/hypotheses.
            bleu = test_results.get("bleu")

        models.append(
            {
                "name": run_dir.name,
                "bidirectional": bool(model_config.get("bidirectional", False)),
                "peeky": bool(model_config.get("peeky", False)),
                "bleu": bleu,
            }
        )

    return models
