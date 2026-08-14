import torch
import numpy as np
import sacrebleu

from inference import translate_batch
from preprocessing import detokenize


@torch.inference_mode()
def evaluate_loss(model, loader, criterion, tgt_pad_idx, device):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for src, src_lengths, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        decoder_input = tgt[:, :-1]
        decoder_target = tgt[:, 1:]

        logits = model(
            src=src, 
            src_lengths=src_lengths, 
            decoder_input=decoder_input
        )

        loss_sum = criterion(
            input=logits.reshape(-1, logits.size(-1)), 
            target=decoder_target.reshape(-1)
        )

        # DO NOT COUNT PADDINGS AS TOKENS
        target_token_count = (decoder_target != tgt_pad_idx).sum()

        total_loss += loss_sum.item()
        total_tokens += target_token_count.item()

    average_loss = total_loss / total_tokens
    perplexity = float(np.exp(average_loss))

    return average_loss, perplexity


@torch.inference_mode()
def evaluate_bleu(
    model,
    loader,
    tgt_vocab,
    tgt_id_to_token,
    device,
    max_length=100,
):
    model.eval()

    hypotheses = []
    references = []

    for src, src_lengths, tgt in loader:

        translations = translate_batch(
            model=model,
            src=src,
            src_lengths=src_lengths,
            tgt_vocab=tgt_vocab,
            tgt_id_to_token=tgt_id_to_token,
            device=device,
            max_length=max_length,
        )

        hypotheses.extend(translations)

        # convert target tensors back to strings
        for target_ids in tgt:
            tokens = []

            for token_id in target_ids.tolist():

                if token_id == tgt_vocab["<eos>"]:
                    break

                if token_id not in {
                    tgt_vocab["<pad>"],
                    tgt_vocab["<bos>"],
                }:
                    tokens.append(
                        tgt_id_to_token[token_id]
                    )

            references.append(detokenize(tokens))

    bleu = sacrebleu.corpus_bleu(
        hypotheses,
        [references],
        tokenize="13a",
    )

    return bleu.score, hypotheses, references
