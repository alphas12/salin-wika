# Architecture

<!-- AI Amended: Explain the connected model and application architecture for maintainers. -->

_Written August 15, 2026 at 5:18 AM (Asia/Manila)._

## Scope

The current system is a Cebuano-to-Tagalog sequence-to-sequence translator without attention. It supports the existing vanilla, bidirectional-encoder, peeky-decoder, and combined modes through `config.yaml`. Weights & Biases, `.env.example`, and `testing/` are intentionally not part of the runtime.

## End-to-end flow

Training follows this path:

`config.yaml` → `main.py training` → `TrainingPipeline` → CSV split → vocabularies → datasets and length-bucketed loaders → `Seq2Seq` → epoch training → validation and early stopping → test loss/perplexity/BLEU → `results/<training.name>/`

Translation follows this path:

`config.yaml` → `main.py translation` → saved run configuration and vocabularies → `Seq2Seq` reconstruction → `best_model.pt` → greedy decoding → translated text

## Model

The encoder embeds the source token IDs, applies dropout, packs padded batches using their true lengths, and processes them with an LSTM. Its final hidden and cell states summarize the source sentence.

The decoder embeds the target input and runs a second LSTM. During training it receives the shifted ground-truth target sequence (teacher forcing). During inference it receives its previous highest-probability token (greedy decoding) until `<eos>` or `translation.max_length`.

When `model.bidirectional` is true, the encoder's forward and backward final states are concatenated and projected through learned hidden/cell bridges into the decoder dimension. When `model.peeky` is true, the encoder context is concatenated to every decoder input and output step. No attention matrix or per-source-position context is computed.

## Data and vocabulary

`TrainingPipeline` reads the configured CSV with Polars and deterministically shuffles it using `training.seed`. It creates train, validation, and test slices from the configured ratios. Only the training slice builds vocabularies, preventing validation/test vocabulary leakage.

The multilingual blank spaCy tokenizer lowercases when configured and preserves token boundaries. Source sequences end with `<eos>`. Target sequences begin with `<bos>` and end with `<eos>`. The collator pads each batch dynamically, while `LengthBucketBatchSampler` groups similarly sized sequences to reduce padding.

## Training and evaluation

Cross-entropy ignores target padding and is normalized by the number of real target tokens. Gradients are clipped using `training.max_grad_norm`. The configured optimizer can be `adamw`, `adam`, or `sgd`.

Every epoch writes `checkpoint_<epoch>.pt`. A lower validation loss replaces `best_model.pt`; training stops after `training.patience` epochs without improvement. The best model is evaluated once on the held-out test set for loss, perplexity, and corpus BLEU.

## Device and container behavior

With `device: auto`, runtime selection is CUDA first, then Apple MPS, then CPU. Explicit unavailable devices fail immediately instead of silently training elsewhere.

The Docker image uses Python 3.12 and the same `requirements.txt` as local setup. ARM64 builds preinstall the CPU PyTorch wheel because Apple MPS is unavailable inside the Linux VM; AMD64 builds install the configured CUDA 12.8 wheel for the NVIDIA training target. Passing Docker's `--gpus all` exposes a supported NVIDIA GPU; without it, PyTorch runs on CPU. The image entry point is `python main.py`, so the container accepts only `training` or `translation` as its application command.

## Result layout

Each run is self-contained below `results/<training.name>/`:

- `best_model.pt`: weights selected by validation loss.
- `checkpoint_<epoch>.pt`: per-epoch weights.
- `configs.json`: the exact configuration used for training.
- `src_token_to_id.json` and `tgt_token_to_id.json`: token lookup maps.
- `src_id_to_token.json` and `tgt_id_to_token.json`: reverse lookup lists.
- `training_data.json`: epoch loss and perplexity history.
- `test_results.json`: held-out metrics, hypotheses, and references.

Translation reconstructs the model from these artifacts, so changing model dimensions in the current `config.yaml` does not corrupt an older run. Only `device` and the `translation` section are taken from the current config during inference.
