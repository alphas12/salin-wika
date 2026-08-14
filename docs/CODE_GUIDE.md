# Code Guide

<!-- AI Amended: Catalog the maintained functions and show how modules call one another. -->

_Written August 15, 2026 at 5:18 AM (Asia/Manila)._

## Entry point: `main.py`

- `load_config()` parses `config.yaml` with safe YAML loading and requires a top-level mapping.
- `run_training()` creates `results/` and hands the configuration to `TrainingPipeline`.
- `run_translation()` loads the configured run and prints one translation.
- `create_parser()` defines the only two CLI commands: `training` and `translation`.
- `main()` dispatches the selected command.

## Model: `models/seq2seq.py`

- `Encoder.__init__()` creates the source embedding, LSTM, dropout, and hidden-state normalization.
- `Encoder.forward()` embeds and packs a padded source batch, then returns final hidden and cell states.
- `Decoder.__init__()` creates the target embedding, LSTM, dropout, output normalization, and vocabulary projection. Peeky mode expands its input/output sizes.
- `Decoder.forward()` decodes one or more target positions and returns vocabulary logits plus recurrent states.
- `Seq2Seq.__init__()` connects the encoder and decoder and creates bridge projections only for a bidirectional encoder.
- `Seq2Seq.bridge_states()` converts bidirectional final states into one decoder layer; vanilla states pass through unchanged.
- `Seq2Seq.encode()` returns decoder initial states and the fixed encoder context.
- `Seq2Seq.forward()` encodes a source batch and teacher-forces the supplied decoder input.

## Preprocessing: `utils/preprocessing.py`

- `spacy_tokenizer()` normalizes a value to text, optionally lowercases it, and tokenizes it.
- `detokenize()` joins generated tokens and removes spaces before common punctuation.
- `build_vocab()` counts training tokens, reserves the four special tokens, applies `min_freq`, and returns forward/reverse maps.
- `TranslationDataset` converts Polars source and target columns into reusable ID sequences. `__getitem__()` returns tensors; `_encode_src()` and `_encode_tgt()` add the required boundary tokens.
- `collate_fn()` dynamically pads a batch and returns source lengths for LSTM packing.
- `LengthBucketBatchSampler` shuffles deterministically per epoch, sorts within temporary buckets, and yields compact batches.

## Pipeline: `utils/pipelines.py`

- `TrainingPipeline.__init__()` reads the CSV and runs split, vocabulary, dataset, and loader construction in order.
- `_create_splits_()` validates ratios, shuffles once with the seed, and creates non-overlapping train/validation/test frames.
- `_create_dataset_()` builds the three encoded datasets with shared training vocabularies.
- `_create_loaders_()` connects padding and length bucketing to PyTorch `DataLoader` instances.
- `train()` selects the device, builds the configured model/optimizer/loss, trains epochs, checkpoints, early-stops, reloads the best weights, evaluates the test set, and saves all artifacts.

## Training and evaluation

- `utils/training.py: train_one_epoch()` shifts target tokens into decoder inputs/targets, runs backpropagation, clips gradients, updates weights, and returns token-normalized loss and perplexity.
- `utils/evaluation.py: evaluate_loss()` performs the same loss calculation without gradients.
- `utils/evaluation.py: evaluate_bleu()` greedily translates every test batch, reconstructs reference strings, and computes corpus BLEU with SacreBLEU.

## Inference: `utils/inference.py`

- `translate_from_results()` validates the run name, loads its saved configuration/vocabularies/best weights, reconstructs `Seq2Seq`, and calls sentence translation.
- `translate_sentence()` tokenizes one configured string and performs greedy autoregressive decoding.
- `translate_batch()` performs the same greedy process for evaluation batches while tracking completed rows.

## Artifact helpers: `utils/helpers.py`

- `load_json()` reads saved run metadata or vocabularies.
- `resolve_device()` implements validated `auto`, `cuda`, `mps`, and `cpu` selection.
- `save_vocabs()` writes both directions of the source and target vocabularies.
- `save_results()` writes training history, test metrics, and the effective configuration.

## Configuration reference

- `device`: `auto`, `cuda`, `mps`, or `cpu`.
- `data`: CSV path/column names, split ratios, vocabulary minimum frequency, sequence length, and lowercasing.
- `model`: embedding/hidden dimensions, bidirectional and peeky flags, and dropout.
- `training`: result run name, loader settings, epochs, optimizer, learning rate, patience, gradient clipping, and seed.
- `translation`: saved run name, input text, lowercasing, and decoding limit.

`training.name` chooses the output directory. `translation.model_name` chooses the run to load; set both to the same value to translate with the latest configured run.

## Maintenance path

For a new model family, add its model module first, then connect model construction in the training and inference paths together so saved configurations remain sufficient to reconstruct it. Do not change preprocessing, result ownership, or the two-command CLI unless the new architecture requires different data.
