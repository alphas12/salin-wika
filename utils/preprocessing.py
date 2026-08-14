import torch
import spacy
import math
import random

from collections import Counter
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence

# AI Amended: Keep the existing spaCy tokenizer while making batching and Polars-backed datasets executable.
nlp = spacy.blank("xx")


def spacy_tokenizer(text, lowercase=True):
    text = str(text).strip()

    if lowercase:
        text = text.lower()

    return [token.text for token in nlp.tokenizer(text)]


def detokenize(tokens):
    text = " ".join(tokens)

    # punctuation that should not have a preceding space
    for p in [".", ",", "!", "?", ":", ";"]:
        text = text.replace(f" {p}", p)

    return text


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def build_vocab(texts, tokenizer, min_freq=2):
    counter = Counter()

    for text in texts:
        counter.update(tokenizer(text))

    token_to_id = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}

    for token, frequency in counter.items():
        if frequency >= min_freq and token not in token_to_id:
            token_to_id[token] = len(token_to_id)

    id_to_token = {idx: token for token, idx in token_to_id.items()}

    return token_to_id, id_to_token


class TranslationDataset(Dataset):
    def __init__(
        self,
        df,
        src_col,
        tgt_col,
        tokenizer,
        src_vocab,
        tgt_vocab,
        lowercase=True,
        max_length=None,
    ):
        self.tokenizer = tokenizer
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.lowercase = lowercase
        self.max_length = max_length
        self.tgt_texts = [str(text) for text in df[tgt_col].to_list()]

        self.src = [self._encode_src(text) for text in df[src_col].to_list()]
        self.tgt = [self._encode_tgt(text) for text in df[tgt_col].to_list()]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = torch.tensor(self.src[idx], dtype=torch.long)
        tgt = torch.tensor(self.tgt[idx], dtype=torch.long)

        return src, tgt

    def _tokenize(self, text):
        return self.tokenizer(
            text,
            lowercase=self.lowercase,
        )

    def _tokens_to_ids(self, tokens, vocab):
        unk_id = vocab["<unk>"]

        return [vocab.get(token, unk_id) for token in tokens]

    def _encode_src(self, text):
        tokens = self._tokenize(text)
        if self.max_length is not None:
            tokens = tokens[: self.max_length - 1]
        token_ids = self._tokens_to_ids(tokens, self.src_vocab)

        return token_ids + [self.src_vocab["<eos>"]]

    def _encode_tgt(self, text):
        tokens = self._tokenize(text)
        if self.max_length is not None:
            tokens = tokens[: self.max_length - 2]
        token_ids = self._tokens_to_ids(tokens, self.tgt_vocab)

        return [self.tgt_vocab["<bos>"]] + token_ids + [self.tgt_vocab["<eos>"]]


def collate_fn(batch, src_pad_idx, tgt_pad_idx):
    src_batch, tgt_batch = zip(*batch)

    src_lengths = torch.tensor([len(src) for src in src_batch], dtype=torch.long)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=src_pad_idx)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_pad_idx)

    return src_batch, src_lengths, tgt_batch


class LengthBucketBatchSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=True,
        seed=42,
        drop_last=False,
        bucket_size_multiplier=20,
    ):
        """Sampler to batch data with similar lengths to reduce padding"""
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.bucket_size = batch_size * bucket_size_multiplier
        self.lengths = [
            max(len(src), len(tgt)) for src, tgt in zip(dataset.src, dataset.tgt)
        ]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))

        if self.shuffle:
            rng.shuffle(indices)

        batches = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=self.lengths.__getitem__)
            for batch_start in range(0, len(bucket), self.batch_size):
                batch = bucket[batch_start : batch_start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)

        self.epoch += 1
        yield from batches

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return math.ceil(len(self.lengths) / self.batch_size)
