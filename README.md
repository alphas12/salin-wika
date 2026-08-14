# SalinWika!

> **NOTE: PLEASE CLEAN THE CORPUS BEFORE TRAINING, AND IT MUST BE IN .CSV FORMAT WITH TWO (2) COLUMNS (FOR THE TWO LANGUAGES)**

This project trains and runs a Cebuano-to-Tagalog LSTM encoder-decoder without attention. Configuration lives only in `config.yaml`, and generated artifacts always live in `results/`. More language support will be provided soon (I'm still testing with the models themselves). Contributions for this project are more than welcome!.

### **PROJECT GOAL: Support for various Filipino languages**
**Current Goal: Train on Eng-Fil to find the best translation model**


## Local setup

Use Python 3.12, then run `python -m pip install -r requirements.txt`. The PyTorch dependency prefers the CUDA 12.8 wheel on supported Linux/Windows computers, uses the native macOS wheel on Mac, and automatically falls back to CPU at runtime. Set `device` in `config.yaml` to `auto`, `cuda`, `mps`, or `cpu`.

The CLI has exactly two commands:

- `python main.py training`
- `python main.py translation`

Before either command, edit `config.yaml`. Translation reads `translation.text` and loads `results/<translation.model_name>/best_model.pt` plus that run's saved configuration and vocabularies.

## Docker (Recommended setup)

Build with `docker build -t encoder-decoder .`.

Docker builds use CPU PyTorch on ARM64 hosts such as Apple Silicon and the CUDA 12.8 wheel on AMD64 NVIDIA training computers. The Linux container cannot use Apple MPS.

Train on CPU with `docker run --rm -v "$PWD/results:/app/results" encoder-decoder training`.

Train on an NVIDIA GPU by adding `--gpus all`: `docker run --rm --gpus all -v "$PWD/results:/app/results" encoder-decoder training`.

Translate with `docker run --rm -v "$PWD/config.yaml:/app/config.yaml:ro" -v "$PWD/results:/app/results" encoder-decoder translation`.

The Docker host must have Docker's NVIDIA GPU support configured for `--gpus all`. Apple MPS remains available for local Python runs.

## Outputs

Each training run writes checkpoints, the best model, vocabularies, the effective configuration, loss/perplexity history, and BLEU results below `results/<training.name>/`.

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) and [CODE_GUIDE.md](docs/CODE_GUIDE.md) for the full project map.
