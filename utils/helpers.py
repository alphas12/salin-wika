import json
import numpy as np


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

    for filename, vocabulary in vocabularies.items():
        with open(results_dir / filename, "w") as file:
            json.dump(vocabulary, file, ensure_ascii=False)

    print(f"Saved Vocabulary at '{results_dir}'...")


def save_results(
    training_dir,
    training_data,
    test_results,
    config,
):
    def to_builtin(value): # helper function for the serialization of numpy values
        if isinstance(value, dict):
            return {key: to_builtin(item) for key, item in value.items()}
        if isinstance(value, list):
            return [to_builtin(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    with open(training_dir / "training_data.json", "w") as file:
        json.dump(to_builtin(training_data), file, indent=4)

    with open(training_dir / "test_results.json", "w") as file:
        json.dump(to_builtin(test_results), file, indent=4)

    with open(training_dir / "configs.json", "w") as file:
        json.dump(to_builtin(config), file, indent=4)

    print(f"Saved results for: {training_dir.name}")
