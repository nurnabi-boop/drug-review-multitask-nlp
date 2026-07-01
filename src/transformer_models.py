"""Fine-tune a transformer for Task A (regression) or Task B (3-class).

Defaults to DistilBERT for fast iteration; pass `--model
microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` to test the
biomedical-pretrained-advantage hypothesis.

This script keeps things deliberately simple: HF `Trainer`, no DeepSpeed, no
mixed-precision toggles beyond `fp16=True` on CUDA. The multi-task variant
lives in `multitask_model.py` to keep the dual-head logic separate.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from .evaluate import classification_metrics, regression_metrics

logger = logging.getLogger(__name__)

Task = Literal["A", "B"]
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {lab: i for i, lab in enumerate(SENTIMENT_LABELS)}
ID2LABEL = {i: lab for lab, i in LABEL2ID.items()}


# --- data ------------------------------------------------------------------


def _to_hf(df: pd.DataFrame, task: Task) -> Dataset:
    cols = {"review": df["review"].astype(str).to_list()}
    if task == "B":
        cols["labels"] = df["sentiment_3"].astype(str).map(LABEL2ID).astype("int64").to_list()
    else:  # Task A
        # Normalize 1-10 -> [0, 1] for stable training; we'll un-normalize at predict time.
        cols["labels"] = ((df["rating"].astype(float) - 1.0) / 9.0).to_list()
    return Dataset.from_dict(cols)


def _tokenize(ds: Dataset, tokenizer, max_length: int = 256) -> Dataset:
    def _fn(batch):
        return tokenizer(batch["review"], truncation=True, max_length=max_length)

    return ds.map(_fn, batched=True, remove_columns=["review"])


# --- metrics callbacks -----------------------------------------------------


def _compute_metrics_b(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    y_true = [ID2LABEL[int(i)] for i in labels]
    y_pred = [ID2LABEL[int(i)] for i in preds]
    m = classification_metrics(np.array(y_true), np.array(y_pred))
    return {"accuracy": m.accuracy, "macro_f1": m.macro_f1}


def _compute_metrics_a(eval_pred):
    logits, labels = eval_pred
    # un-normalize
    pred_ratings = np.clip(logits.squeeze(-1) * 9.0 + 1.0, 1.0, 10.0)
    true_ratings = np.asarray(labels).squeeze() * 9.0 + 1.0
    m = regression_metrics(true_ratings, pred_ratings)
    return {"mae": m.mae, "rmse": m.rmse, "acc_within_1": m.acc_within_1}


# --- main fit --------------------------------------------------------------


def run(
    data_path: Path,
    model_name: str,
    task: Task,
    out_dir: Path,
    *,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 256,
    max_train_rows: int | None = None,
    seed: int = 42,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(data_path)
    logger.info("Loaded %d rows from %s", len(df), data_path)

    train = df.loc[df["split"] == "train"]
    val = df.loc[df["split"] == "val"]
    test = df.loc[df["split"] == "test"]
    if max_train_rows is not None:
        train = train.sample(min(len(train), max_train_rows), random_state=seed)
        logger.info("Subsampled train to %d rows", len(train))

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if task == "B":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        compute_metrics = _compute_metrics_b
        metric_for_best = "macro_f1"
        greater_is_better = True
    elif task == "A":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1, problem_type="regression"
        )
        compute_metrics = _compute_metrics_a
        metric_for_best = "mae"
        greater_is_better = False
    else:
        raise ValueError(f"unsupported task: {task!r}")

    train_ds = _tokenize(_to_hf(train, task), tokenizer, max_length=max_length)
    val_ds = _tokenize(_to_hf(val, task), tokenizer, max_length=max_length)
    test_ds = _tokenize(_to_hf(test, task), tokenizer, max_length=max_length)

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best,
        greater_is_better=greater_is_better,
        fp16=torch.cuda.is_available(),
        report_to=["none"],
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    val_metrics = trainer.evaluate(val_ds)
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")

    final_dir = out_dir / "best"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    payload = {
        "model": model_name,
        "task": task,
        "val": {k.replace("eval_", ""): v for k, v in val_metrics.items()},
        "test": {k.replace("test_", ""): v for k, v in test_metrics.items()},
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, default=float))
    logger.info("Saved best model + metrics to %s", out_dir)
    return payload


# --- entrypoint -------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/clean.parquet"))
    p.add_argument("--model", default="distilbert-base-uncased")
    p.add_argument("--task", choices=["A", "B"], default="B")
    p.add_argument("--out", type=Path, default=None,
                   help="defaults to models/{model_slug}_{task}")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--max_train_rows", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    if args.out is None:
        slug = args.model.replace("/", "__")
        args.out = Path("models") / f"{slug}_{args.task}"
    payload = run(
        data_path=args.data,
        model_name=args.model,
        task=args.task,
        out_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        max_train_rows=args.max_train_rows,
        seed=args.seed,
    )
    print(json.dumps(payload, indent=2, default=float))


if __name__ == "__main__":
    main()
