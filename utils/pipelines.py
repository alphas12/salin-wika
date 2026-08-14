import torch
import torch.nn as nn
import torch.optim as optim
import polars as pl

from pathlib import Path
from torch.utils.data import DataLoader
from functools import partial
from utils.preprocessing import (
    build_vocab,
    spacy_tokenizer,
    TranslationDataset,
    LengthBucketBatchSampler,
    collate_fn,
)
from models.seq2seq import Seq2Seq
from utils.training import train_one_epoch
from utils.evaluation import evaluate_bleu, evaluate_loss
from utils.helpers import resolve_device, save_results, save_vocabs


# AI Amended: Make the existing pipeline consume the shared config and connect every training stage end to end.
class TrainingPipeline:
    def __init__(self, config: dict):
        data_config = config["data"]

        self.df = pl.read_csv(data_config["corpus_path"])
        self.src_col = data_config["src_col"]
        self.tgt_col = data_config["tgt_col"]
        self.config = config
        self.data_config = data_config
        self.model_config = config["model"]
        self.training_config = config["training"]

        # Pre-processing Pipeline
        # ---------------------------------

        # Step 1: Split the dataset
        self.train_df, self.valid_df, self.test_df = self._create_splits_()

        # Step 2: Vocabulary creation and token mapping
        self.src_vocab, self.src_id_to_token = build_vocab(
            self.train_df[self.src_col],
            spacy_tokenizer,
            min_freq=data_config["min_freq"],
        )
        self.tgt_vocab, self.tgt_id_to_token = build_vocab(
            self.train_df[self.tgt_col],
            spacy_tokenizer,
            min_freq=data_config["min_freq"],
        )

        # Step 3: Initialize the datasets
        self.train_set, self.valid_set, self.test_set = self._create_dataset_()

        # Step 4: Initialize the dataloaders
        self.train_loader, self.valid_loader, self.test_loader = self._create_loaders_()

    def _create_splits_(self):

        train_size = self.data_config["train_size"]
        valid_size = self.data_config["valid_size"]
        test_size = self.data_config["test_size"]
        seed = self.training_config["seed"]

        # Ensure sizes sums up to 1.0
        if abs(train_size + valid_size + test_size - 1.0) > 1e-9:
            raise ValueError("Data split sizes must sum to 1.0.")
        if min(train_size, valid_size, test_size) <= 0:
            raise ValueError("Data split sizes must all be greater than zero.")

        shuffled_df = self.df.sample(fraction=1.0, shuffle=True, seed=seed)
        train_count = int(len(shuffled_df) * train_size)
        valid_count = int(len(shuffled_df) * valid_size)
        test_count = len(shuffled_df) - train_count - valid_count

        if min(train_count, valid_count, test_count) == 0:
            raise ValueError("The configured data splits must each contain at least one row.")

        train_df = shuffled_df.slice(0, train_count)
        valid_df = shuffled_df.slice(train_count, valid_count)
        test_df = shuffled_df.slice(train_count + valid_count)

        print("Created dataset splits...")

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
        max_length = self.data_config["max_length"]
        lowercase = self.data_config.get("lowercase", True)

        train_set = TranslationDataset(
            df=train_df,
            src_col=src_col,
            tgt_col=tgt_col,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            tokenizer=tokenizer,
            lowercase=lowercase,
            max_length=max_length,
        )

        valid_set = TranslationDataset(
            df=valid_df,
            src_col=src_col,
            tgt_col=tgt_col,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            tokenizer=tokenizer,
            lowercase=lowercase,
            max_length=max_length,
        )

        test_set = TranslationDataset(
            df=test_df,
            src_col=src_col,
            tgt_col=tgt_col,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            tokenizer=tokenizer,
            lowercase=lowercase,
            max_length=max_length,
        )

        print("Created datasets...")

        return train_set, valid_set, test_set

    def _create_loaders_(self):

        src_pad_idx = self.src_vocab["<pad>"]
        tgt_pad_idx = self.tgt_vocab["<pad>"]

        train_set = self.train_set
        valid_set = self.valid_set
        test_set = self.test_set

        batch_size = self.training_config["batch_size"]
        num_workers = self.training_config["num_workers"]
        pin_memory = self.training_config["pin_memory"] and torch.cuda.is_available()

        seed = self.training_config["seed"]

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

        print("Created data loaders...")

        return train_loader, valid_loader, test_loader

    def train(self, results_dir: Path):

        model_name = self.training_config["name"]
        if not isinstance(model_name, str) or model_name != Path(model_name).name:
            raise ValueError("training.name must be a directory name, not a path.")

        training_dir = results_dir / model_name
        training_dir.mkdir(parents=True, exist_ok=True)

        src_dim = len(self.src_vocab)
        tgt_dim = len(self.tgt_vocab)
        src_pad_idx = self.src_vocab["<pad>"]
        tgt_pad_idx = self.tgt_vocab["<pad>"]

        embedding_dim = self.model_config["embedding_dim"]
        hidden_dim = self.model_config["hidden_dim"]
        is_bidirectional = self.model_config["bidirectional"]
        is_peeky = self.model_config["peeky"]
        dropout = self.model_config["dropout"]

        epochs = self.training_config["epochs"]
        chosen_optimizer = self.training_config["optimizer"].lower()
        learning_rate = self.training_config["lr"]
        patience = self.training_config["patience"]
        max_grad_norm = self.training_config.get("max_grad_norm", 1.0)
        
        train_loader = self.train_loader
        valid_loader = self.valid_loader
        test_loader = self.test_loader
        config = self.config
        src_vocab = self.src_vocab
        tgt_vocab = self.tgt_vocab
        src_id_to_token = self.src_id_to_token
        tgt_id_to_token = self.tgt_id_to_token
        max_length = self.data_config["max_length"]

        # Training Initialization
        # ---------------------------------
        save_vocabs(
            results_dir=training_dir,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            src_id_to_token=src_id_to_token,
            tgt_id_to_token=tgt_id_to_token,
        )

        device = resolve_device(self.config.get("device", "auto"))
        torch.manual_seed(self.training_config["seed"])
        print(f"Training '{model_name}' on {device}...")

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
        if chosen_optimizer not in optimizer_classes:
            choices = ", ".join(optimizer_classes)
            raise ValueError(f"optimizer must be one of: {choices}.")

        optimizer = optimizer_classes[chosen_optimizer](
            params=model.parameters(), lr=learning_rate
        )
        criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx, reduction="sum")

        # Training Loop
        # ---------------------------------
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
                max_grad_norm=max_grad_norm,
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

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train loss: {train_loss:.4f} | valid loss: {valid_loss:.4f}"
            )

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
                f=(training_dir / "best_model.pt"),
                map_location=device,
                weights_only=True,
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
