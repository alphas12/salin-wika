import torch
import torch.nn as nn
import torch.optim as optim
import polars as pl

from pathlib import Path
from torch.utils.data import DataLoader
from functools import partial
from sklearn.model_selection import train_test_split
from preprocessing import (
    build_vocab,
    spacy_tokenizer,
    TranslationDataset,
    LengthBucketBatchSampler,
    collate_fn,
)
from models.seq2seq import Seq2Seq
from training import train_one_epoch
from evaluation import evaluate_bleu, evaluate_loss
from helpers import save_results, save_vocabs


class TrainingPipeline:
    def __init__(
        self,
        df_path: Path | str,
        src_col: str,
        tgt_col: str,
        config: dict,
    ):
        self.df = pl.read_csv(df_path)
        self.src_col = src_col
        self.tgt_col = tgt_col
        self.config = config

        # Pre-processing Pipeline
        # ---------------------------------

        # Step 1: Split the dataset
        self.train_df, self.valid_df, self.test_df = self._create_splits_()

        # Step 2: Vocabulary creation and token mapping
        self.src_vocab, self.src_id_to_token = build_vocab(
            self.train_df[src_col], spacy_tokenizer, min_freq=config["min_freq"]
        )
        self.tgt_vocab, self.tgt_id_to_token = build_vocab(
            self.train_df[tgt_col], spacy_tokenizer, min_freq=config["min_freq"]
        )

        # Step 3: Initialize the datasets
        self.train_set, self.valid_set, self.test_set = self._create_dataset_()

        # Step 4: Initialize the dataloaders
        self.train_loader, self.valid_loader, self.test_loader = self._create_loaders_()

    def _create_splits_(self):

        df = self.df
        train_size = self.config["train_size"]
        valid_size = self.config["valid_size"]
        test_size = self.config["test_size"]
        seed = self.config["seed"]

        # Ensure sizes sums up to 1.0
        assert (
            train_size + valid_size + test_size
        ) == 1.0, "Split sizes must sum up to 1.0!"

        train_df, temp_df = train_test_split(
            df,
            train_size=train_size,
            random_state=seed,
        )

        # Computing the relative sizes for exact threeway split
        remaining_size = valid_size + test_size
        valid_relative_size = valid_size / remaining_size
        test_relative_size = test_size / remaining_size

        valid_df, test_df = train_test_split(
            temp_df,
            train_size=valid_relative_size,
            test_size=test_relative_size,
            random_state=seed,
        )

        return train_df, valid_df, test_df

    def _create_dataset_(self):

        train_df = self.train_df
        valid_df = self.valid_df
        test_df = self.test_df
        src_col = self.src_col
        tgt_col = self.tgt_col
        tokenizer = spacy_tokenizer
        src_vocab = self.src_vocab
        tgt_vocab = self.tgt_vocab
        max_length = self.config["max_length"]

        train_set = TranslationDataset(
            df=train_df,
            src_col=src_col,
            tgt_col=tgt_col,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            tokenizer=tokenizer,
            max_length=max_length,
        )

        valid_set = TranslationDataset(
            df=valid_df,
            src_col=src_col,
            tgt_col=tgt_col,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            tokenizer=tokenizer,
            max_length=max_length,
        )

        test_set = TranslationDataset(
            df=test_df,
            src_col=src_col,
            tgt_col=tgt_col,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            tokenizer=tokenizer,
            max_length=max_length,
        )

        return train_set, valid_set, test_set

    def _create_loaders_(self):

        src_pad_idx = self.src_vocab["<pad>"]
        tgt_pad_idx = self.tgt_vocab["<pad>"]

        train_set = self.train_set
        valid_set = self.valid_set
        test_set = self.test_set

        batch_size = self.config["batch_size"]
        num_workers = self.config["num_workers"]
        pin_memory = self.config["pin_memory"]

        seed = self.config["seed"]

        collate = partial(
            collate_fn,
            src_pad_idx=src_pad_idx,
            tgt_pad_idx=tgt_pad_idx,
        )

        train_batch_sampler = LengthBucketBatchSampler(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
        )

        train_loader = DataLoader(
            dataset=train_set,
            batch_sampler=train_batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate,
        )

        valid_loader = DataLoader(
            dataset=valid_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate,
        )

        test_loader = DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate,
        )

        return train_loader, valid_loader, test_loader

    def train(self, results_dir: Path):

        model_name = self.config["name"]
        src_dim = len(self.src_vocab)
        tgt_dim = len(self.tgt_vocab)
        src_pad_idx = self.src_vocab["<pad>"]
        tgt_pad_idx = self.tgt_vocab["<pad>"]
        embedding_dim = self.config["embedding_dim"]
        hidden_dim = self.config["hidden_dim"]
        is_bidirectional = self.config["bidirectional"]
        is_peeky = self.config["peeky"]
        dropout = self.config["dropout"]
        epochs = self.config["epochs"]
        chosen_optimizer = self.config["optimizer"].lower()
        learning_rate = self.config["lr"]
        patience = self.config["patience"]
        train_loader = self.train_loader
        valid_loader = self.valid_loader
        test_loader = self.test_loader
        config = self.config
        src_vocab = self.src_vocab
        tgt_vocab = self.tgt_vocab
        src_id_to_token = self.src_id_to_token
        tgt_id_to_token = self.tgt_id_to_token
        max_length = self.config["max_length"]

        # Training Initialization
        # ---------------------------------
        save_vocabs(
            results_dir=results_dir, 
            src_vocab=src_vocab, 
            tgt_vocab=tgt_vocab,
            src_id_to_token=src_id_to_token,
            tgt_id_to_token=tgt_id_to_token,
        )

        req_device = "cpu"
        if torch.cuda.is_available():
            req_device = "cuda"
        elif torch.backends.mps.is_available():
            req_device = "mps"
        device = torch.device(req_device)

        model = Seq2Seq(
            input_dim=src_dim,
            output_dim=tgt_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            src_pad_idx=src_pad_idx,
            tgt_pad_idx=tgt_pad_idx,
            bidirectional=is_bidirectional,
            peeky=is_peeky,
            dropout=dropout,
        ).to(device)

        optimizer_classes = {
            "adamw": optim.AdamW,
            "adam": optim.Adam,
            "sgd": optim.SGD,
        }
        optimizer = optimizer_classes[chosen_optimizer](
            params=model.parameters(), lr=learning_rate
        )
        criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx, reduction="sum")

        # Training Loop
        # ---------------------------------
        training_dir = results_dir / model_name
        training_dir.mkdir(exist_ok=True)

        training_data = {
            "model_name": model_name,
            "train_loss": [],
            "train_perplexity": [],
            "valid_loss": [],
            "valid_perplexity": [],
        }

        best_loss = float("inf")
        epochs_wout_improvement = 0
        for epoch in range(epochs):

            train_loss, train_perplexity = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                tgt_pad_idx=tgt_pad_idx,
                device=device,
            )

            valid_loss, valid_perplexity = evaluate_loss(
                model=model,
                loader=valid_loader,
                criterion=criterion,
                tgt_pad_idx=tgt_pad_idx,
                device=device,
            )

            training_data["train_loss"].append(train_loss)
            training_data["valid_loss"].append(valid_loss)
            training_data["train_perplexity"].append(train_perplexity)
            training_data["valid_perplexity"].append(valid_perplexity)

            # Checkpoint
            torch.save(
                obj=model.state_dict(), f=(training_dir / f"checkpoint_{epoch+1}.pt")
            )

            # Save best model and early stopping check
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(obj=model.state_dict(), f=(training_dir / "best_model.pt"))
                epochs_wout_improvement = 0
            else:
                epochs_wout_improvement += 1

            if epochs_wout_improvement >= patience:
                print(f"Early Stopping at epoch {epoch+1} for '{model_name}'...")
                break

        # Evaluation
        # ---------------------------------
        model.load_state_dict(
            state_dict=torch.load(
                f=(training_dir / "best_model.pt"), map_location=device
            )
        )

        test_loss, test_perplexity = evaluate_loss(
            model=model,
            loader=test_loader,
            criterion=criterion,
            tgt_pad_idx=tgt_pad_idx,
            device=device,
        )

        bleu_score, hypotheses, references = evaluate_bleu(
            model=model,
            loader=test_loader,
            tgt_vocab=tgt_vocab,
            tgt_id_to_token=tgt_id_to_token,
            device=device,
            max_length=max_length,
        )

        test_results = {
            "test_loss": test_loss,
            "test_perplexity": test_perplexity,
            "bleu": bleu_score,
            "bleu_results": {
                "hypotheses": hypotheses,
                "references": references,
            },
        }

        save_results(
            training_dir=training_dir,
            training_data=training_data,
            test_results=test_results,
            config=config,
        )

        print(f"Model '{model_name}' has been trained successfully...")
