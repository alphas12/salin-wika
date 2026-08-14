import torch

from preprocessing import detokenize


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

    states, context = model.encode(src, src_lengths)
    current_token = torch.tensor(
        [[tgt_vocab["<bos>"]]], dtype=torch.long, device=device
    )

    generated_ids = []
    for _ in range(max_length):
        logits, states = model.decoder(x=current_token, states=states, context=context)

        next_id = logits[:, -1, :].argmax(dim=-1)
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

    states, context = model.encode(src, src_lengths)

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

        logits, states = model.decoder(
            x=current_token,
            states=states,
            context=context,
        )

        next_ids = logits[:, -1, :].argmax(dim=-1)

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
