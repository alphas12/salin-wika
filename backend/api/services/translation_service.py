# AI Amended: Thin adapter between the HTTP layer and the existing CLI
# contract in utils/inference.py, so inference code stays single-sourced.
#
# translate_from_results() (see CODE_GUIDE.md) already does everything a
# request needs: validate the run name, load its saved configs.json,
# vocabularies, and best_model.pt, reconstruct Seq2Seq, and translate one
# string taken from config["translation"]["text"] for the run named in
# config["translation"]["model_name"]. Rather than re-implement any of that,
# this module builds a per-request config that overrides just those two
# keys on top of the project's own config.yaml (so device, lowercasing, and
# translation.max_length still come from one place).
#
# NOTE: this calls translate_from_results() fresh on every request, which
# means it re-loads weights + vocabs each time. That's fine for getting a
# working web UI up. If translation latency becomes a problem, the next
# step is splitting utils/inference.py into a load_run(model_name) step
# that can be cached per model_name, and a translate(model, text) step that
# reuses it — see the "Suggested follow-up" note in the project README.
import copy
from pathlib import Path

import yaml

from utils.inference import translate_from_results


class ModelNotFoundError(Exception):
    pass


def _load_base_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"'{config_path}' must contain a YAML mapping.")

    return config


def translate(model_name: str, text: str, results_dir: Path, base_config_path: Path) -> str:
    run_dir = results_dir / model_name
    if not run_dir.exists():
        raise ModelNotFoundError(f"No run named '{model_name}' under {results_dir}.")

    config = copy.deepcopy(_load_base_config(base_config_path))
    config.setdefault("translation", {})
    config["translation"]["model_name"] = model_name
    config["translation"]["text"] = text

    return translate_from_results(config=config, results_dir=results_dir)
