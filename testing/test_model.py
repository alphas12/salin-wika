import torch

from models.bahdanau import BahdanauSeq2Seq
from models.seq2seq import Seq2Seq
from utils.inference import translate_batch


def test_both_models_support_training_and_inference():
    src = torch.tensor([[4, 3, 0], [4, 4, 3]])
    src_lengths = torch.tensor([2, 3])
    decoder_input = torch.tensor([[2, 4], [2, 4]])
    tgt_vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, "word": 4}
    tgt_id_to_token = {index: token for token, index in tgt_vocab.items()}

    models = [
        Seq2Seq(
            input_dim=5,
            output_dim=5,
            embedding_dim=4,
            hidden_dim=6,
            src_pad_idx=0,
            tgt_pad_idx=0,
            num_layers=1,
        ),
        BahdanauSeq2Seq(
            input_dim=5,
            output_dim=5,
            embedding_dim=4,
            encoder_dim=6,
            decoder_dim=6,
            attention_dim=4,
        ),
    ]

    for model in models:
        logits = model(
            src=src,
            src_lengths=src_lengths,
            decoder_input=decoder_input,
        )
        assert logits.shape == (2, 2, 5)
        assert len(
            translate_batch(
                model=model,
                src=src,
                src_lengths=src_lengths,
                tgt_vocab=tgt_vocab,
                tgt_id_to_token=tgt_id_to_token,
                device=torch.device("cpu"),
                max_length=2,
            )
        ) == 2


if __name__ == "__main__":
    test_both_models_support_training_and_inference()
