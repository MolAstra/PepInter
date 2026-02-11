import pandas as pd
import lightning as L
from pathlib import Path
from torch.utils.data import DataLoader
import torch
from typing import Any, Dict, List, Optional
from loguru import logger

from torchmetrics import MeanMetric, Accuracy, AUROC, R2Score, MeanAbsoluteError
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding

from .pepinter import ESMSeqTokenizer
from .data.csv import CSVDataset
from .pepinter import (
    PepInterConfig,
    PepInterModel,
    PepInterModelForClassification,
    PepInterModelForMaskedLM,
    PepInterModelForEnergy,
    PepInterModelForAffinity,
)
from .esmc.modeling_esmc import load_weights_from_esm, ESMCOutput

from .base import BaseLightningModule, MetricCollection


class PepInter(BaseLightningModule):
    def __init__(
        self,
        model_args: Optional[dict] = None,
        tokenizer: Optional[ESMSeqTokenizer] = None,
        load_from_esmc: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.model_args = model_args or {}
        self.tokenizer = tokenizer
        self.model: PepInterModel = PepInterModel(PepInterConfig(**self.model_args))

        if tokenizer is not None:
            self.tokenizer = tokenizer

        if load_from_esmc:
            load_weights_from_esm(self.model)

        self._init_metrics()

    def _init_metrics(self):
        self.metrics: MetricCollection = MetricCollection.from_metrics(
            {
                "train_loss": MeanMetric(sync_on_compute=True),
                "val_loss": MeanMetric(sync_on_compute=True),
                "val_acc": Accuracy(sync_on_compute=True, task="binary"),
                "val_auc": AUROC(sync_on_compute=True, task="binary"),
            }
        )

    def forward(
        self, input_ids, attention_mask, token_type_ids, labels, return_loss=True
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            return_loss=return_loss,
        )

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            labels=batch["labels"],
            return_loss=True,
        )
        self.log_step(
            "train_loss_step",
            outputs.loss,
        )
        self.metrics.update("train_loss", outputs.loss)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs: ESMCOutput = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            labels=batch["labels"],
            return_loss=True,
        )

        logits = outputs.logits.detach()

        self.metrics.update("val_loss", outputs.loss.detach())
        self.metrics.update("val_acc", logits, labels=batch["labels"].flatten())
        self.metrics.update("val_auc", logits, labels=batch["labels"].flatten())

    def on_train_epoch_end(self):
        self.log_epoch(
            "train_loss",
            self.metrics.compute("train_loss"),
        )
        self.metrics.reset_all(prefix="train")

    def on_validation_epoch_end(self):
        self.log_epoch(
            "val_loss",
            self.metrics.compute("val_loss"),
        )
        self.log_epoch(
            "val_acc",
            self.metrics.compute("val_acc"),
        )
        self.log_epoch(
            "val_auc",
            self.metrics.compute("val_auc"),
        )
        self.metrics.reset_all(prefix="val")


class PepInterForClassification(BaseLightningModule):
    def __init__(
        self,
        model_args: Optional[dict] = None,
        tokenizer: Optional[ESMSeqTokenizer] = None,
        load_from_esmc: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.model_args = model_args or {}
        self.tokenizer = tokenizer
        self.model: PepInterModelForClassification = PepInterModelForClassification(
            PepInterConfig(**self.model_args)
        )

        if tokenizer is not None:
            self.tokenizer = tokenizer

        if load_from_esmc:
            load_weights_from_esm(self.model)

        self._init_metrics()

    def _init_metrics(self):
        if self.model_args["num_labels"] == 1:
            acc_metrics = Accuracy(sync_on_compute=True, task="binary")
            auc_metrics = AUROC(sync_on_compute=True, task="binary")
        else:
            acc_metrics = Accuracy(
                sync_on_compute=True,
                task="multiclass",
                num_classes=self.model_args["num_labels"],
            )
            auc_metrics = AUROC(
                sync_on_compute=True,
                task="multiclass",
                num_classes=self.model_args["num_labels"],
            )

        self.metrics: MetricCollection = MetricCollection.from_metrics(
            {
                "train_loss": MeanMetric(sync_on_compute=True),
                "val_loss": MeanMetric(sync_on_compute=True),
                "val_acc": acc_metrics,
                "val_auc": auc_metrics,
            }
        )

    def forward(
        self, input_ids, attention_mask, token_type_ids, labels, return_loss=True
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            return_loss=return_loss,
        )

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            labels=batch["labels"],
            return_loss=True,
        )
        self.log_step(
            "train_loss_step",
            outputs.loss,
        )
        self.metrics.update("train_loss", outputs.loss)
        return outputs.loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        outputs: ESMCOutput = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            labels=batch["labels"],
            return_loss=True,
        )

        logits = outputs.logits.detach()

        self.metrics.update("val_loss", outputs.loss.detach())
        self.metrics.update("val_acc", logits, labels=batch["labels"].unsqueeze(-1))
        self.metrics.update("val_auc", logits, labels=batch["labels"].unsqueeze(-1))

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx):
        outputs: ESMCOutput = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            labels=None,
            return_loss=False,
        )

        logits = outputs.logits.detach()
        if self.model_args["num_labels"] == 1:
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
        else:
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return {
            "logits": logits.detach().cpu(),
            "probs": probs.detach().cpu().squeeze(-1),
            "preds": preds.detach().cpu().squeeze(-1),
        }

    def on_train_epoch_end(self):
        self.log_epoch(
            "train_loss",
            self.metrics.compute("train_loss"),
        )
        self.metrics.reset_all(prefix="train")

    def on_validation_epoch_end(self):
        self.log_epoch(
            "val_loss",
            self.metrics.compute("val_loss"),
        )
        self.log_epoch(
            "val_acc",
            self.metrics.compute("val_acc"),
        )

        self.log_epoch(
            "val_auc",
            self.metrics.compute("val_auc"),
        )
        self.metrics.reset_all(prefix="val")


class PepInterForEnergy(BaseLightningModule):
    def __init__(
        self,
        model_args: Optional[dict] = None,
        tokenizer: Optional[ESMSeqTokenizer] = None,
        load_from_esmc: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.model_args = model_args or {}
        self.tokenizer = tokenizer
        self.model: PepInterModelForEnergy = PepInterModelForEnergy(
            PepInterConfig(**self.model_args)
        )

        if tokenizer is not None:
            self.tokenizer = tokenizer

        if load_from_esmc:
            load_weights_from_esm(self.model)

        self._init_metrics()

    def _init_metrics(self):
        self.metrics: MetricCollection = MetricCollection.from_metrics(
            {
                "train_loss": MeanMetric(sync_on_compute=True),
                "val_loss": MeanMetric(sync_on_compute=True),
                "val_r2": R2Score(sync_on_compute=True),
                "val_mae": MeanAbsoluteError(sync_on_compute=True),
            }
        )

    def forward(
        self, input_ids, attention_mask, token_type_ids, labels, return_loss=True
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            return_loss=return_loss,
        )

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            labels=batch["labels"],
            return_loss=True,
        )
        self.log_step(
            "train_loss_step",
            outputs.loss,
        )
        self.metrics.update("train_loss", outputs.loss)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs: ESMCOutput = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            labels=batch["labels"],
            return_loss=True,
        )  # (B, 1)

        self.metrics.update("val_loss", outputs.loss.detach())
        self.metrics.update("val_r2", outputs.logits, labels=batch["labels"])
        self.metrics.update("val_mae", outputs.logits, labels=batch["labels"])

    def on_train_epoch_end(self):
        self.log_epoch(
            "train_loss",
            self.metrics.compute("train_loss"),
        )
        self.metrics.reset_all(prefix="train")

    def on_validation_epoch_end(self):
        self.log_epoch(
            "val_loss",
            self.metrics.compute("val_loss"),
        )
        self.log_epoch(
            "val_r2",
            self.metrics.compute("val_r2"),
        )
        self.log_epoch(
            "val_mae",
            self.metrics.compute("val_mae"),
        )
        self.metrics.reset_all(prefix="val")


class PepInterForAffinity(BaseLightningModule):
    def __init__(
        self,
        model_args: Optional[dict] = None,
        tokenizer: Optional[ESMSeqTokenizer] = None,
        load_from_esmc: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.model_args = model_args or {}
        self.tokenizer = tokenizer
        # self.model: PepInterModelForAffinity = PepInterModelForAffinity(
        #     PepInterConfig(**self.model_args)
        # )
        self.model: PepInterModelForEnergy = PepInterModelForEnergy(
            PepInterConfig(**self.model_args)
        )

        if tokenizer is not None:
            self.tokenizer = tokenizer

        if load_from_esmc:
            load_weights_from_esm(self.model)

        self._init_metrics()

    def _init_metrics(self):
        self.metrics: MetricCollection = MetricCollection.from_metrics(
            {
                "train_loss": MeanMetric(sync_on_compute=True),
                "val_loss": MeanMetric(sync_on_compute=True),
                "val_r2": R2Score(sync_on_compute=True),
                "val_mae": MeanAbsoluteError(sync_on_compute=True),
            }
        )

    def forward(
        self, input_ids, attention_mask, token_type_ids, labels, return_loss=True
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            return_loss=return_loss,
        )

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            labels=batch["labels"],
            return_loss=True,
        )
        self.log_step(
            "train_loss_step",
            outputs.loss,
        )
        self.metrics.update("train_loss", outputs.loss)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs: ESMCOutput = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            labels=batch["labels"],
            return_loss=True,
        )  # (B, 1)

        self.metrics.update("val_loss", outputs.loss.detach())
        self.metrics.update("val_r2", outputs.logits, labels=batch["labels"])
        self.metrics.update("val_mae", outputs.logits, labels=batch["labels"])

    def on_train_epoch_end(self):
        self.log_epoch(
            "train_loss",
            self.metrics.compute("train_loss"),
        )
        self.metrics.reset_all(prefix="train")

    def on_validation_epoch_end(self):
        self.log_epoch(
            "val_loss",
            self.metrics.compute("val_loss"),
        )
        self.log_epoch(
            "val_r2",
            self.metrics.compute("val_r2"),
        )
        self.log_epoch(
            "val_mae",
            self.metrics.compute("val_mae"),
        )
        self.metrics.reset_all(prefix="val")

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx):
        outputs: ESMCOutput = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            labels=None,
            return_loss=False,
        )

        logits = outputs.logits.detach()
        return {
            "preds": logits.detach().cpu().unsqueeze(-1),
        }


class PepInterForMaskedLM(BaseLightningModule):
    def __init__(
        self,
        model_args: Optional[dict] = None,
        tokenizer: Optional[ESMSeqTokenizer] = None,
        load_from_esmc: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.model_args = model_args or {}
        self.tokenizer = tokenizer
        self.model: PepInterModelForMaskedLM = PepInterModelForMaskedLM(
            PepInterConfig(**self.model_args)
        )

        if tokenizer is not None:
            self.tokenizer = tokenizer

        if load_from_esmc:
            load_weights_from_esm(self.model)

        self._init_metrics()

    def _init_metrics(self):
        self.metrics: MetricCollection = MetricCollection.from_metrics(
            {
                "train_loss": MeanMetric(sync_on_compute=True),
                "val_loss": MeanMetric(sync_on_compute=True),
            }
        )

    def forward(
        self, input_ids, attention_mask, token_type_ids, labels, return_loss=True
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            return_loss=return_loss,
        )

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            labels=batch["labels"],
            return_loss=True,
        )
        self.log_step(
            "train_loss_step",
            outputs.loss,
        )
        self.metrics.update("train_loss", outputs.loss)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs: ESMCOutput = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            labels=batch["labels"],
            return_loss=True,
        )

        self.metrics.update("val_loss", outputs.loss.detach())

    def on_train_epoch_end(self):
        self.log_epoch(
            "train_loss",
            self.metrics.compute("train_loss"),
        )
        self.metrics.reset_all(prefix="train")

    def on_validation_epoch_end(self):
        self.log_epoch(
            "val_loss",
            self.metrics.compute("val_loss"),
        )
        self.metrics.reset_all(prefix="val")


class PepInterDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_data_path: Path,
        val_data_path: Path,
        tokenizer: ESMSeqTokenizer,
        batch_size: int = 128,
        num_workers: int = 8,
        max_len: int = 1024,
    ):
        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_len = max_len
        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding="longest",
            pad_to_multiple_of=8,
        )

    def setup(self, stage=None):
        if stage == "fit":
            self.train_set = CSVDataset(
                self.train_data_path,
                select_columns=["target_seq", "mobile_seq", "label"],
                transform=self._preprocess,
            )
            self.val_set = CSVDataset(
                self.val_data_path,
                select_columns=["target_seq", "mobile_seq", "label"],
                transform=self._preprocess,
            )
            logger.info(f"train_set: {len(self.train_set)}")
            logger.info(f"val_set: {len(self.val_set)}")
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def _preprocess(self, sample: dict, max_prot_len: int = 960, max_pep_len: int = 64):
        prot_seq = sample["target_seq"].strip()[:max_prot_len]
        pep_seq = sample["mobile_seq"].strip()[:max_pep_len]

        label = sample["label"]

        tokenized = self.tokenizer(
            prot_seq,
            pep_seq,
            padding=False,
            return_token_type_ids=True,
        )
        tokenized["label"] = label

        return tokenized

    def collate_fn(self, batch: list[dict]):
        tokenized = self.data_collator(batch)
        labels = torch.tensor(
            [item["label"] for item in batch],
            dtype=torch.float,
        )  # (B,)

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "token_type_ids": tokenized["token_type_ids"],
            "labels": labels,
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )


class PepInterDataModuleForClassification(PepInterDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PepInterDataModuleForEnergy(PepInterDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, stage=None):
        if stage == "fit":
            self.train_set = CSVDataset(
                self.train_data_path,
                select_columns=["target_sequence", "fragments", "affinity"],
                transform=self._preprocess,
            )
            self.val_set = CSVDataset(
                self.val_data_path,
                select_columns=["target_sequence", "fragments", "affinity"],
                transform=self._preprocess,
            )
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def _preprocess(self, sample: dict, max_prot_len: int = 960, max_pep_len: int = 64):
        prot_seq = sample["target_sequence"].strip()[:max_prot_len]
        pep_seq = sample["fragments"].strip()[:max_pep_len]

        label = sample["affinity"]

        tokenized = self.tokenizer(
            prot_seq,
            pep_seq,
            padding=False,
            return_token_type_ids=True,
        )
        tokenized["label"] = label

        return tokenized

    def collate_fn(self, batch: list[dict]):
        tokenized = self.data_collator(batch)
        labels = torch.tensor(
            [item["label"] for item in batch],
            dtype=torch.float,
        ).unsqueeze(
            -1
        )  # (B) -> (B, 1)

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "token_type_ids": tokenized["token_type_ids"],
            "labels": labels,
        }


class PepInterDataModuleForMaskedLM(L.LightningDataModule):
    def __init__(
        self,
        train_data_path: Path | None = None,
        val_data_path: Path | None = None,
        tokenizer: ESMSeqTokenizer | None = None,
        batch_size: int = 128,
        num_workers: int = 8,
        pin_memory: bool = True,
        max_prot_len: int = 1024,
        max_pep_len: int = 64,
        make_toy: bool = False,
    ):
        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_prot_len = max_prot_len
        self.max_pep_len = max_pep_len
        self.max_len = max_prot_len + max_pep_len
        self.make_toy = make_toy

        self.data_collator: DataCollatorForLanguageModeling = (
            DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=True,
                pad_to_multiple_of=8,
            )
        )

    def transform(self, example: Dict[str, Any]) -> Dict[str, Any]:
        target_seq = example["target_seq"].strip()[: self.max_prot_len]
        mobile_seq = example["mobile_seq"].strip()[: self.max_pep_len]

        tok = self.tokenizer(
            target_seq,
            mobile_seq,
            padding=False,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=True,
        )

        return {
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "token_type_ids": tok["token_type_ids"],
            "target_seq": target_seq,
            "mobile_seq": mobile_seq,
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_tensor = self.data_collator(
            [
                {
                    "input_ids": example["input_ids"],
                    "attention_mask": example["attention_mask"],
                    "token_type_ids": example["token_type_ids"],
                }
                for example in batch
            ]
        )

        target_seqs = [example["target_seq"] for example in batch]
        mobile_seqs = [example["mobile_seq"] for example in batch]

        return {
            "input_ids": batch_tensor["input_ids"],
            "attention_mask": batch_tensor["attention_mask"],
            "token_type_ids": batch_tensor["token_type_ids"],
            "labels": batch_tensor["labels"],
            "target_seq": target_seqs,
            "mobile_seq": mobile_seqs,
        }

    def setup(self, stage=None):
        if stage == "fit":
            self.train_set = CSVDataset(
                self.train_data_path,
                select_columns=["target_seq", "mobile_seq"],
                transform=self.transform,
                make_toy=self.make_toy,
            )
            self.val_set = CSVDataset(
                self.val_data_path,
                select_columns=["target_seq", "mobile_seq"],
                transform=self.transform,
                make_toy=self.make_toy,
            )
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )


class PepInterInferDataModuleForClassification(L.LightningDataModule):
    def __init__(
        self,
        data_path: Path,
        tokenizer: ESMSeqTokenizer | None = None,
        batch_size: int = 128,
        num_workers: int = 8,
        max_len: int = 1024,
    ):
        super().__init__()
        self.data_path = data_path

        if tokenizer is None:
            self.tokenizer = ESMSeqTokenizer()
        else:
            self.tokenizer = tokenizer

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_len = max_len

        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding="longest",
            pad_to_multiple_of=8,
        )

    def setup(self, stage=None):
        if stage == "predict":
            self.dataset = CSVDataset(
                self.data_path,
                # select_columns=["target_seq", "mobile_seq"],
                transform=self._preprocess,
            )
            assert (
                "target_seq" in self.dataset.df.columns
            ), "target_seq column not found"
            assert (
                "mobile_seq" in self.dataset.df.columns
            ), "mobile_seq column not found"
            logger.info(f"dataset: {len(self.dataset)}")
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def _preprocess(self, sample: dict, max_prot_len: int = 960, max_pep_len: int = 64):
        prot_seq = sample["target_seq"].strip()[:max_prot_len]
        pep_seq = sample["mobile_seq"].strip()[:max_pep_len]

        tokenized = self.tokenizer(
            prot_seq,
            pep_seq,
            padding=False,
            return_token_type_ids=True,
        )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "token_type_ids": tokenized["token_type_ids"],
            **sample,
        }

    def collate_fn(self, batch: list[dict]):
        input_batch = {
            "input_ids": [item["input_ids"] for item in batch],
            "attention_mask": [item["attention_mask"] for item in batch],
            "token_type_ids": [item["token_type_ids"] for item in batch],
        }
        other_batch = {
            k: [item[k] for item in batch]
            for k in batch[0].keys()
            if k not in ["input_ids", "attention_mask", "token_type_ids"]
        }
        tokenized = self.data_collator(input_batch)
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "token_type_ids": tokenized["token_type_ids"],
            **other_batch,
        }

    def predict_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )


class PepInterInferDataModuleForAffinity(PepInterInferDataModuleForClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CSVWriterCallback(L.Callback):
    def __init__(
        self,
        save_dir: str = "./outputs",
        filename: str = "predictions.csv",
    ):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.filename = filename
        self._records: List[Dict[str, Any]] = []

    def _collect_batch_outputs(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]
    ):
        if isinstance(outputs, list):
            outputs = outputs[0]
        if outputs is None:
            return

        preds = outputs.get("preds").tolist()
        probs = outputs.get("probs").tolist()

        # 取除输入外的 batch 字段
        other_keys = [
            k
            for k in batch.keys()
            if k not in ["input_ids", "attention_mask", "token_type_ids"]
        ]
        batch_size = len(preds)

        for i in range(batch_size):
            rec = {k: batch[k][i] for k in other_keys}
            rec["pred"] = preds[i]
            rec["prob"] = probs[i] if probs is not None else None
            self._records.append(rec)

    def _write_to_csv(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / self.filename

        df = pd.DataFrame(self._records)
        if len(df) == 0:
            logger.warning("No prediction records to write.")
            return
        else:
            df.to_csv(path, index=False)
            logger.info(f"Saved {len(df)} rows to {path}")

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._collect_batch_outputs(outputs, batch)

    def on_predict_epoch_end(self, trainer, pl_module):
        self._write_to_csv()
        self._records.clear()


class CSVWriterCallbackForAffinity(CSVWriterCallback):
    def _collect_batch_outputs(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]
    ):
        if isinstance(outputs, list):
            outputs = outputs[0]
        if outputs is None:
            return

        preds = outputs.get("preds").tolist()
        # 取除输入外的 batch 字段
        other_keys = [
            k
            for k in batch.keys()
            if k not in ["input_ids", "attention_mask", "token_type_ids"]
        ]
        batch_size = len(preds)

        for i in range(batch_size):
            rec = {k: batch[k][i] for k in other_keys}
            rec["pred"] = preds[i]

            self._records.append(rec)
