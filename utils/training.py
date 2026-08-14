import math

from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_


# AI Amended: Call Seq2Seq with its real parameter names and keep token-normalized training metrics.
def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    tgt_pad_idx,
    device,
    max_grad_norm=1.0,
):
    model.train()

    total_loss = 0.0
    total_tokens = 0

    tqdm_loader = tqdm(loader, desc="Training...", leave=True)
    for src, src_lengths, tgt in tqdm_loader:
        src = src.to(device)
        tgt = tgt.to(device)

        decoder_input = tgt[:, :-1]
        decoder_target = tgt[:, 1:]

        optimizer.zero_grad()

        logits = model(
            src=src,
            src_lengths=src_lengths,
            decoder_input=decoder_input,
        )

        loss_sum = criterion(
            input=logits.reshape(-1, logits.size(-1)),
            target=decoder_target.reshape(-1),
        )

        target_token_count = (decoder_target != tgt_pad_idx).sum()

        loss = loss_sum / target_token_count

        loss.backward()

        clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        total_loss += loss_sum.item()
        total_tokens += target_token_count.item()

    average_loss = total_loss / total_tokens
    perplexity = math.exp(average_loss)

    return average_loss, perplexity
