# SalinWika

SalinWika is a configuration-driven CLI for translation, full fine-tuning, and new-language adaptation with Meta's `facebook/nllb-200-distilled-600M` checkpoint.

> NLLB is released under CC BY-NC 4.0 for research use. Its model card says it is not intended for production, certified, medical, legal, or document translation.

## Setup

Use Python 3.12 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

All settings live in `config.yaml`. `device` accepts `auto`, `cuda`, `mps`, or `cpu`; an explicitly unavailable device fails instead of silently falling back.

## CLI

```bash
python main.py preload
python main.py translate
python main.py finetune
python main.py adapt
```

- `preload` loads the configured tokenizer and model, applies maximum generation length and beam search, reports readiness, and exits.
- `translate` loads once, uses the YAML source/target pair, and repeatedly prompts for text until `/exit`.
- `finetune` updates all model weights for one configured, already-supported language pair.
- `adapt` adds one new language token, resizes the embeddings, and updates all weights using both new→pivot and pivot→new examples.

The default available languages are Tagalog, Cebuano, Ilocano, Waray, and Pangasinense. To use an adapted model, set `model.name_or_path` to its result directory and add the new name/code to `languages`.

## Corpus format

Each run accepts one parallel pair in a two-column UTF-8 CSV:

```csv
cebuano,tagalog
Maayong buntag.,Magandang umaga.
```

Set the path, column names, and language names in `fine_tuning` or `adaptation`. Rows with null or blank sentences are rejected. A CSV containing several language columns is not mixed automatically; run each pair separately or first convert it into separate two-column corpora.

Splits are deterministic. Adaptation splits the original pairs before reversing them, so a sentence pair and its reverse stay in the same split.

## Training and outputs

Training uses Hugging Face `Seq2SeqTrainer`, dynamic padding, epoch validation/checkpointing, early stopping, and full-weight updates without gradient accumulation. Held-out results include loss, FLORES-200 spBLEU, and chrF++.

Each run is saved in standard Hugging Face format below `results/<run_name>/`:

- model and tokenizer files loadable with `from_pretrained()`;
- best/limited checkpoints and trainer state;
- `effective_config.yaml`;
- train, validation, and test metric JSON files.

The 600M model should be trained on a capable GPU. CPU is suitable for CLI inference and tests but is impractical for the provided 104K-pair corpus. The first spBLEU evaluation downloads SacreBLEU's FLORES-200 tokenizer.

Run tests with:

```bash
python -m pytest -q
```

## Docker

```bash
docker build -t salinwika .
docker run --rm salinwika --help
docker run --rm -it -v "$PWD/config.yaml:/app/config.yaml:ro" salinwika translate
docker run --rm --gpus all \
  -v "$PWD/config.yaml:/app/config.yaml:ro" \
  -v "$PWD/corpus:/app/corpus:ro" \
  -v "$PWD/results:/app/results" \
  salinwika finetune
```

Apple MPS is available only to local macOS Python, not inside the Linux container.

See [Architecture](docs/ARCHITECTURE.md), [Code Guide](docs/CODE_GUIDE.md), the [NLLB documentation](https://huggingface.co/docs/transformers/model_doc/nllb), and the [official model card](https://huggingface.co/facebook/nllb-200-distilled-600M).
