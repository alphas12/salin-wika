# From the Paper: Bahdanau et al. 2014
# Link: https://arxiv.org/abs/1409.0473

# Code loosely from: "Transformers for Machine Learning: A Deep Dive" by Kamath et al. 2022
# Good info link: https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Dimension Legend:
# - Batch Size (B)
# - Token Size (T)
# - Embedding Dim (E)
# - Encoder Dim (Enc)
# - Decoder Dim (Dec)
# - Attention Dim (A)
# - Target Vocabulary Dim (V_tgt)


class BahdanauEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, encoder_dim, decoder_dim, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=input_dim, 
            embedding_dim=embedding_dim,
        )

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=encoder_dim,
            bidirectional=True,
            batch_first=True,
        )

        self.linear = nn.Linear(
            in_features=encoder_dim * 2,
            out_features=decoder_dim,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):

        # Input: [B, T]

        # [B, T, E]
        embeddings = self.dropout(self.embedding(x))

        # Outputs: [B, T, 2 * Enc]
        # Hidden: [2, B, Enc]
        # AI Amended: Ignore padded time steps when producing encoder states.
        packed = pack_padded_sequence(
            embeddings,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, hidden = self.gru(packed)
        outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=x.size(1),
        )

        # [B, 2 * Enc]
        concat_hidden = torch.cat((hidden[-2], hidden[-1]), dim=-1)

        # [B, Dec]
        hidden = torch.tanh(self.linear(concat_hidden))

        return outputs, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, query_size, key_size):
        super().__init__()

        self.query_layer = nn.Linear(query_size, hidden_size)
        self.key_layer = nn.Linear(key_size, hidden_size)
        self.energy_layer = nn.Linear(hidden_size, 1)

    def forward(self, encoder_outputs, prev_hidden_decoder, src_mask):

        # Prev Hidden Decoder: [B, Dec]
        # Encoder Dims: [B, T, 2 * Enc]

        # q_i = W_a s_{i-1}
        # [B, A]
        query_out = self.query_layer(prev_hidden_decoder)

        # [B, 1, A]
        query_out = query_out.unsqueeze(1)

        # k_j = U_a h_j
        # [B, T, A]
        key_out = self.key_layer(encoder_outputs)

        # tanh(q_i + k_j)
        # [B, T, A]
        energy_input = torch.tanh(query_out + key_out)

        # e_ij = v_a^T tanh(q_i + k_j)
        # [B, T, 1]
        energies = self.energy_layer(energy_input)

        # [B, T]
        energies = energies.squeeze(-1)
        energies = energies.masked_fill(src_mask == 0, float("-inf"))

        # softmax(e_ij)
        weights = F.softmax(energies, dim=-1)

        # [B, 1, T]
        weights = weights.unsqueeze(1)

        # attention-scores = softmax(e_ij) @ U_a
        # [B, 1, 2 * Enc]
        attention = torch.bmm(weights, encoder_outputs)

        return attention


class BahdanauDecoder(nn.Module):
    def __init__(
        self, output_dim, embedding_dim, encoder_dim, decoder_dim, dropout=0.3
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=output_dim, 
            embedding_dim=embedding_dim,
        )

        self.gru = nn.GRU(
            input_size=embedding_dim + encoder_dim * 2,
            hidden_size=decoder_dim,
            batch_first=True,
        )

        self.output = nn.Linear(
            in_features=decoder_dim + encoder_dim * 2 + embedding_dim, 
            out_features=output_dim,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, prev_hidden, attention):
        # Input: [B,]

        # [B,] -> [B, E]
        embedding = self.dropout(self.embedding(x))

        # [B, E] -> [B, 1, E]
        embedding = embedding.unsqueeze(1)

        # [B, 1, E + 2 * Enc] = [B, 1, E] + [B, 1, 2 * Enc]
        rnn_input = torch.cat((embedding, attention), dim=-1)

        # [B, Dec] -> [1, B, Dec], since GRU expects [num_layers, B, Dec]
        hidden = prev_hidden.unsqueeze(0)

        # Output: [B, 1, Dec]
        # Hidden: [1, B, Dec]
        output, hidden = self.gru(rnn_input, hidden)

        # [B, 1, Dec] -> [B, Dec]
        output = output.squeeze(1)

        # [B, Dec] -> [B, V_tgt]
        logits = self.output(
            torch.cat(
                [output, attention.squeeze(1), embedding.squeeze(1)],
                dim=-1,
            )
        )

        # [1, B, Dec] -> [B, Dec]
        hidden = hidden.squeeze(0)

        return logits, hidden


class BahdanauSeq2Seq(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        embedding_dim,
        encoder_dim,
        decoder_dim,
        attention_dim,
        dropout=0.3,
    ):
        super().__init__()

        self.output_dim = output_dim

        self.encoder = BahdanauEncoder(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            dropout=dropout,
        )

        self.attention = BahdanauAttention(
            hidden_size=attention_dim,
            query_size=decoder_dim,
            key_size=encoder_dim * 2,
        )

        self.decoder = BahdanauDecoder(
            output_dim=output_dim,
            embedding_dim=embedding_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            dropout=dropout,
        )

    def forward(
        self,
        src,
        src_lengths,
        tgt=None,
        src_mask=None,
        decoder_input=None,
    ):
        if decoder_input is None:
            if tgt is None:
                raise ValueError("decoder_input or tgt is required.")
            decoder_input = tgt[:, :-1]

        if src_mask is None:
            positions = torch.arange(src.size(1), device=src.device).unsqueeze(0)
            src_mask = positions < src_lengths.to(src.device).unsqueeze(1)

        encoder_outputs, hidden = self.encoder(src, src_lengths)

        outputs = []

        for t in range(decoder_input.size(1)):

            # AI Amended: Pass encoder keys and decoder query to attention correctly.
            attention = self.attention(
                encoder_outputs=encoder_outputs,
                prev_hidden_decoder=hidden,
                src_mask=src_mask,
            )

            logits, hidden = self.decoder(
                x=decoder_input[:, t],
                prev_hidden=hidden,
                attention=attention,
            )

            outputs.append(logits)

        outputs = torch.stack(outputs, dim=1)
        return outputs
