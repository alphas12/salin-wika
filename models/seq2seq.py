import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        pad_idx,
        bidirectional=False,
        dropout=0.3,
        num_layers=2,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_layers = num_layers

        self.embed = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )

        self.gru = nn.GRU(
            input_size=embedding_dim,  # FIXED
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
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

        _, hidden_state = self.gru(packed_embeddings)

        hidden_state = self.hidden_norm(hidden_state)

        return hidden_state


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim,
        embedding_dim,
        hidden_dim,
        pad_idx,
        peeky=False,
        dropout=0.3,
        num_layers=2,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.peeky = peeky
        self.num_layers = num_layers

        self.embed = nn.Embedding(
            num_embeddings=output_dim,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )

        gru_input_dim = embedding_dim
        fc_input_dim = hidden_dim

        if peeky:
            gru_input_dim += hidden_dim
            fc_input_dim += hidden_dim

        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)

        self.fc = nn.Linear(
            fc_input_dim,
            output_dim,
        )

    def forward(self, x, states, context):
        embeddings = self.dropout(self.embed(x))

        if self.peeky:
            context_expanded = context.unsqueeze(1).expand(-1, embeddings.size(1), -1)
            gru_input = torch.cat([embeddings, context_expanded], dim=-1)
        else:
            gru_input = embeddings

        output, states = self.gru(
            gru_input,
            states,
        )

        output = self.output_norm(output)

        if self.peeky:
            output = torch.cat([output, context_expanded], dim=-1)

        logits = self.fc(output)

        return logits, states


class Seq2Seq(nn.Module):
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
        num_layers=2,
    ):
        super().__init__()

        self.bidirectional = bidirectional
        self.peeky = peeky
        self.num_layers = num_layers

        self.encoder = Encoder(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            pad_idx=src_pad_idx,
            bidirectional=bidirectional,
            dropout=dropout,
            num_layers=num_layers,
        )

        self.decoder = Decoder(
            output_dim=output_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            pad_idx=tgt_pad_idx,
            peeky=peeky,
            dropout=dropout,
            num_layers=num_layers,
        )

        if bidirectional:
            self.hidden_bridge = nn.Linear(
                hidden_dim * 2,
                hidden_dim,
            )

    def bridge_states(self, hidden_state):
        if not self.bidirectional:
            return hidden_state

        batch_size = hidden_state.size(1)

        hidden_state = hidden_state.view(self.num_layers, 2, batch_size, self.encoder.hidden_dim)

        forward_hidden = hidden_state[:, 0]
        backward_hidden = hidden_state[:, 1]

        combined_hidden = torch.cat([forward_hidden, backward_hidden], dim=-1)
        decoder_hidden = torch.tanh(self.hidden_bridge(combined_hidden))

        return decoder_hidden

    def encode(self, src, src_lengths):
        encoder_hidden = self.encoder(src, src_lengths)

        decoder_hidden = self.bridge_states(encoder_hidden)
        context = decoder_hidden[-1]

        return decoder_hidden, context

    def forward(self, src, src_lengths, decoder_input):
        state, context = self.encode(src, src_lengths)

        logits, _ = self.decoder(
            x=decoder_input,
            states=state,
            context=context,
        )

        return logits