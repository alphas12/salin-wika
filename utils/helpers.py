import json

import torch


# AI Amended: Centralize JSON loading and validated automatic device selection for training and inference.
def load_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def resolve_device(requested_device="auto"):
    requested_device = requested_device.lower()

    if requested_device == "auto":
        if torch.cuda.is_available():
            requested_device = "cuda"
        elif torch.backends.mps.is_available():
            requested_device = "mps"
        else:
            requested_device = "cpu"

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested in config.yaml but is unavailable.")
    if requested_device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested in config.yaml but is unavailable.")
    if requested_device not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps.")

    return torch.device(requested_device)


def save_vocabs(
    results_dir,
    src_vocab,
    tgt_vocab,
    src_id_to_token,
    tgt_id_to_token,
):
    vocabularies = {
        "src_token_to_id.json": src_vocab,
        "tgt_token_to_id.json": tgt_vocab,
        "src_id_to_token.json": [
            src_id_to_token[index] for index in range(len(src_id_to_token))
        ],
        "tgt_id_to_token.json": [
            tgt_id_to_token[index] for index in range(len(tgt_id_to_token))
        ],
    }

    # AI Amended: Write vocabulary text as UTF-8 on every operating system.
    for filename, vocabulary in vocabularies.items():
        with open(results_dir / filename, "w", encoding="utf-8") as file:
            json.dump(vocabulary, file, ensure_ascii=False)

    print(f"Saved Vocabulary at '{results_dir}'...")


def save_results(
    training_dir,
    training_data,
    test_results,
    config,
):
    # AI Amended: Keep all saved run metadata portable when it contains Unicode text.
    with open(training_dir / "training_data.json", "w", encoding="utf-8") as file:
        json.dump(training_data, file, indent=4)

    with open(training_dir / "test_results.json", "w", encoding="utf-8") as file:
        json.dump(test_results, file, indent=4)

    with open(training_dir / "configs.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

    print(f"Saved results for: {training_dir.name}")
