import torch

from pathlib import Path

from models.seq2seq import Seq2Seq
from models.bahdanau import BahdanauSeq2Seq
from utils.helpers import load_json, resolve_device
from utils.preprocessing import detokenize, spacy_tokenizer


# AI Amended: Load a complete trained run from results so the CLI needs only config.yaml.
def translate_from_results(config, results_dir):
    translation_config = config["translation"]
    model_name = translation_config["model_name"]

    if not isinstance(model_name, str) or model_name != Path(model_name).name:
        raise ValueError("translation.model_name must be a directory name, not a path.")

    training_dir = results_dir / model_name
    saved_config = load_json(training_dir / "configs.json")
    src_vocab = load_json(training_dir / "src_token_to_id.json")
    tgt_vocab = load_json(training_dir / "tgt_token_to_id.json")
    tgt_id_to_token = load_json(training_dir / "tgt_id_to_token.json")
    model_choice = saved_config["model"]
    if type(model_choice) is not int or model_choice not in {1, 2}:
        raise ValueError("saved model must be 1 (Seq2Seq) or 2 (Bahdanau).")
    model_config = saved_config[f"model_{model_choice}"]
    device = resolve_device(config.get("device", "auto"))

    if model_choice == 1:
        model = Seq2Seq(
            input_dim=len(src_vocab),
            output_dim=len(tgt_vocab),
            embedding_dim=model_config["embedding_dim"],
            hidden_dim=model_config["hidden_dim"],
            src_pad_idx=src_vocab["<pad>"],
            tgt_pad_idx=tgt_vocab["<pad>"],
            bidirectional=model_config["bidirectional"],
            peeky=model_config["peeky"],
            dropout=model_config["dropout"],
            num_layers=model_config["num_layers"],
        ).to(device)
    else:
        model = BahdanauSeq2Seq(
            input_dim=len(src_vocab),
            output_dim=len(tgt_vocab),
            embedding_dim=model_config["embedding_dim"],
            encoder_dim=model_config["encoder_dim"],
            decoder_dim=model_config["decoder_dim"],
            attention_dim=model_config["attention_dim"],
            dropout=model_config["dropout"],
        ).to(device)
    model.load_state_dict(
        torch.load(
            training_dir / "best_model.pt",
            map_location=device,
            weights_only=True,
        )
    )

    return translate_sentence(
        model=model,
        sentence=translation_config["text"],
        tokenizer=spacy_tokenizer,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        tgt_id_to_token=tgt_id_to_token,
        device=device,
        lowercase=translation_config.get("lowercase", True),
        max_length=translation_config.get("max_length", 100),
    )


def _encode(model, src, src_lengths):
    if isinstance(model, BahdanauSeq2Seq):
        encoder_outputs, states = model.encoder(src, src_lengths)
        positions = torch.arange(src.size(1), device=src.device).unsqueeze(0)
        src_mask = positions < src_lengths.to(src.device).unsqueeze(1)
        return states, (encoder_outputs, src_mask)

    return model.encode(src, src_lengths)


def _decode_step(model, current_token, states, context):
    if isinstance(model, BahdanauSeq2Seq):
        encoder_outputs, src_mask = context
        attention = model.attention(encoder_outputs, states, src_mask)
        return model.decoder(current_token.squeeze(1), states, attention)

    logits, states = model.decoder(
        x=current_token,
        states=states,
        context=context,
    )
    return logits[:, -1, :], states


@torch.inference_mode()
def translate_sentence(
    model,
    sentence,
    tokenizer,
    src_vocab,
    tgt_vocab,
    tgt_id_to_token,
    device,
    lowercase=True,
    max_length=100,
):
    model.eval()

    query_tokens = tokenizer(sentence, lowercase=lowercase)

    src_unk_idx = src_vocab["<unk>"]
    src_ids = [src_vocab.get(token, src_unk_idx) for token in query_tokens]
    src_ids.append(src_vocab["<eos>"])

    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_lengths = torch.tensor([len(src_ids)], dtype=torch.long)

    states, context = _encode(model, src, src_lengths)
    current_token = torch.tensor(
        [[tgt_vocab["<bos>"]]], dtype=torch.long, device=device
    )

    generated_ids = []
    for _ in range(max_length):
        logits, states = _decode_step(model, current_token, states, context)

        next_id = logits.argmax(dim=-1)
        next_id_value = next_id.item()

        # End the translation if it ends with the ending sentence marker
        if next_id_value == tgt_vocab["<eos>"]:
            break

        # Don't include padding and beginning of sentence markers in the generated sentences
        if next_id_value not in {tgt_vocab["<pad>"], tgt_vocab["<bos>"]}:
            generated_ids.append(next_id_value)

        current_token = next_id.unsqueeze(1)

    generated_tokens = [tgt_id_to_token[token_id] for token_id in generated_ids]

    return detokenize(tokens=generated_tokens)


@torch.inference_mode()
def translate_batch(
    model,
    src,
    src_lengths,
    tgt_vocab,
    tgt_id_to_token,
    device,
    max_length=100,
):
    model.eval()

    src = src.to(device)

    batch_size = src.size(0)

    states, context = _encode(model, src, src_lengths)

    current_token = torch.full(
        (batch_size, 1),
        tgt_vocab["<bos>"],
        dtype=torch.long,
        device=device,
    )

    generated = [[] for _ in range(batch_size)]

    # tracks which sentences already generated <eos>
    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=device,
    )

    for _ in range(max_length):

        logits, states = _decode_step(model, current_token, states, context)

        next_ids = logits.argmax(dim=-1)

        for i, token_id in enumerate(next_ids.tolist()):
            if finished[i]:
                continue

            if token_id == tgt_vocab["<eos>"]:
                finished[i] = True
                continue

            if token_id not in {
                tgt_vocab["<pad>"],
                tgt_vocab["<bos>"],
            }:
                generated[i].append(token_id)

        # everyone finished -> stop decoding
        if finished.all():
            break

        current_token = next_ids.unsqueeze(1)

    translations = []

    for ids in generated:
        tokens = [tgt_id_to_token[token_id] for token_id in ids]

        translations.append(detokenize(tokens))

    return translations
