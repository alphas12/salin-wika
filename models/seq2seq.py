import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence


# RNN-based Encoder-Decoder model
class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        pad_idx,
        bidirectional=False,
        dropout=0.3,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.embed = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)

        self.hidden_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, lengths):
        embeddings = self.dropout(self.embed(x))
        packed_embeddings = pack_padded_sequence(
            embeddings,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden_state, cell_state) = self.lstm(packed_embeddings)
        hidden_state = self.hidden_norm(hidden_state)
        return hidden_state, cell_state


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim,
        embedding_dim,
        hidden_dim,
        pad_idx,
        peeky=False,
        dropout=0.3,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.peeky = peeky

        self.embed = nn.Embedding(
            num_embeddings=output_dim, embedding_dim=embedding_dim, padding_idx=pad_idx
        )

        lstm_input_dim = embedding_dim
        fc_input_dim = hidden_dim

        if peeky:
            lstm_input_dim += hidden_dim
            fc_input_dim += hidden_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim, hidden_size=hidden_dim, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(fc_input_dim, output_dim)

    def forward(self, x, states, context):
        embeddings = self.dropout(self.embed(x))

        if self.peeky:
            context_expanded = context.unsqueeze(1).expand(-1, embeddings.size(1), -1)
            lstm_input = torch.cat([embeddings, context_expanded], dim=-1)
        else:
            lstm_input = embeddings

        output, states = self.lstm(lstm_input, states)
        output = self.output_norm(output)

        if self.peeky:
            output = torch.cat([output, context_expanded], dim=-1)  # type: ignore

        logits = self.fc(output)

        return logits, states


class Seq2Seq(nn.Module):
    """
    This variant includes the following modes:

    - Vanilla Seq2Seq
    - BiLSTM Encoder (Note: Set bidirectional to 'True')
    - Peeky Decoder (Note: Set peeky to 'True')
    - BiLSTM Encoder + Peeky Decoder (Note: Set bidirectional and peeky to 'True')
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        embedding_dim,
        hidden_dim,
        src_pad_idx,
        tgt_pad_idx,
        bidirectional=False,
        peeky=False,
        dropout=0.3,
    ):
        super().__init__()

        self.bidirectional = bidirectional
        self.peeky = peeky

        self.encoder = Encoder(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            pad_idx=src_pad_idx,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.decoder = Decoder(
            output_dim=output_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            pad_idx=tgt_pad_idx,
            peeky=peeky,
            dropout=dropout,
        )

        if bidirectional:
            self.hidden_bridge = nn.Linear(
                in_features=hidden_dim * 2, out_features=hidden_dim
            )
            self.cell_bridge = nn.Linear(
                in_features=hidden_dim * 2, out_features=hidden_dim
            )

    def bridge_states(self, hidden_state, cell_state):
        if not self.bidirectional:
            return hidden_state, cell_state

        forward_hidden = hidden_state[-2]
        backward_hidden = hidden_state[-1]

        forward_cell = cell_state[-2]
        backward_cell = cell_state[-1]

        combined_hidden = torch.cat([forward_hidden, backward_hidden], dim=-1)
        combined_cell = torch.cat([forward_cell, backward_cell], dim=-1)

        decoder_hidden = torch.tanh(self.hidden_bridge(combined_hidden)).unsqueeze(0)
        decoder_cell = torch.tanh(self.cell_bridge(combined_cell)).unsqueeze(0)

        return decoder_hidden, decoder_cell

    def encode(self, src, src_lengths):
        encoder_hidden, encoder_cell = self.encoder(src, src_lengths)

        decoder_hidden, decoder_cell = self.bridge_states(
            hidden_state=encoder_hidden,
            cell_state=encoder_cell,
        )

        states = (
            decoder_hidden,
            decoder_cell,
        )

        context = decoder_hidden.squeeze(0)

        return states, context

    def forward(self, src, src_lengths, decoder_input):
        states, context = self.encode(src=src, src_lengths=src_lengths)
        logits, _ = self.decoder(x=decoder_input, states=states, context=context)
        return logits
