"""Multi-task model: shared encoder with rating regression + ADR token-classification heads.

Training data:
- Rating head: every train-split review (label = 1-10 rating, normalized to [0, 1]).
- ADR head: only reviews that have a non-empty span set, supervised at the
  *token* level using BIO tags derived from scispaCy `DISEASE` mentions
  treated as silver ADR labels. The 200 hand-labeled reviews are held back
  for evaluation only.

This silver-supervision approach is intentional: hand-labeling 215k reviews
is infeasible, but scispaCy-as-teacher gives a noisy-but-useful ADR signal
that the multi-task model can refine through joint training with the
rating signal.

Loss = lambda_rating * MSE(rating) + lambda_adr * CE(BIO tags), with
lambda_adr = 0 on rows that have no scispaCy spans (so they don't bias the
ADR head toward all-O).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, DataCollatorForTokenClassification, get_linear_schedule_with_warmup

from .adr_extraction import extract_corpus, load_scispacy
from .evaluate import regression_metrics, span_metrics

logger = logging.getLogger(__name__)


# BIO scheme for a single ADR class (we collapse DISEASE -> ADR for silver labels)
BIO_LABELS = ["O", "B-ADR", "I-ADR"]
BIO2ID = {lab: i for i, lab in enumerate(BIO_LABELS)}
ID2BIO = {i: lab for lab, i in BIO2ID.items()}


# --- model -----------------------------------------------------------------


@dataclass
class MultitaskOutput:
    loss: torch.Tensor | None
    rating_pred: torch.Tensor  # [B] in [0, 1]
    bio_logits: torch.Tensor   # [B, T, 3]


class MultitaskHead(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.rating_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )
        self.bio_head = nn.Linear(hidden, len(BIO_LABELS))

    def forward(
        self,
        input_ids,
        attention_mask,
        rating_label: torch.Tensor | None = None,
        bio_labels: torch.Tensor | None = None,
        adr_weight: torch.Tensor | None = None,
        lambda_rating: float = 1.0,
        lambda_adr: float = 1.0,
    ) -> MultitaskOutput:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq = self.dropout(out.last_hidden_state)        # [B, T, H]
        cls = seq[:, 0]                                  # [B, H]

        rating_pred = self.rating_head(cls).squeeze(-1)  # [B] in [0, 1]
        bio_logits = self.bio_head(seq)                  # [B, T, 3]

        loss = None
        if rating_label is not None or bio_labels is not None:
            loss = torch.zeros((), device=seq.device)
            if rating_label is not None:
                rating_loss = nn.functional.mse_loss(rating_pred, rating_label)
                loss = loss + lambda_rating * rating_loss
            if bio_labels is not None:
                bio_loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
                # bio_logits: [B, T, C]; targets: [B, T]
                losses = bio_loss_fct(
                    bio_logits.view(-1, len(BIO_LABELS)),
                    bio_labels.view(-1),
                ).view(bio_labels.size())
                # mask: per-row weight (0 if review had no silver spans)
                if adr_weight is None:
                    adr_weight = torch.ones(bio_labels.size(0), device=seq.device)
                token_mask = (bio_labels != -100).float()
                per_row_denom = token_mask.sum(dim=1).clamp(min=1.0)
                per_row_loss = (losses * token_mask).sum(dim=1) / per_row_denom
                bio_loss = (per_row_loss * adr_weight).sum() / adr_weight.sum().clamp(min=1.0)
                loss = loss + lambda_adr * bio_loss

        return MultitaskOutput(loss=loss, rating_pred=rating_pred, bio_logits=bio_logits)


# --- silver-label construction --------------------------------------------


def build_silver_spans(reviews: pd.DataFrame, model_name: str = "en_ner_bc5cdr_md") -> pd.DataFrame:
    """Run scispaCy over the train split and return a per-review list of
    DISEASE spans, treated as silver ADR labels."""
    nlp = load_scispacy(model_name)
    spans = extract_corpus(reviews, nlp=nlp, keep_labels=("DISEASE",))
    grouped = (
        spans.groupby("review_id")
        .apply(lambda g: [(int(r["start"]), int(r["end"])) for _, r in g.iterrows()])
        .rename("silver_spans")
        .reset_index()
    )
    return grouped


def _spans_to_bio(
    text: str,
    spans: list[tuple[int, int]],
    tokenizer,
    max_length: int = 256,
) -> dict:
    """Tokenize a review and produce BIO labels aligned to subword tokens.

    Returns input_ids, attention_mask, bio_labels (-100 on special tokens),
    and the binary `has_adr` flag.
    """
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        padding=False,
    )
    offsets = enc.pop("offset_mapping")
    bio = []
    span_intervals = sorted(spans)
    for tok_start, tok_end in offsets:
        if tok_start == 0 and tok_end == 0:
            bio.append(-100)
            continue
        # find any silver span this token is inside of
        found = None
        for s_start, s_end in span_intervals:
            if tok_start >= s_end:
                continue
            if tok_end <= s_start:
                break
            if max(tok_start, s_start) < min(tok_end, s_end):
                found = (s_start, s_end)
                break
        if found is None:
            bio.append(BIO2ID["O"])
        else:
            # B if this is the first token overlapping the span, else I
            if tok_start <= found[0]:
                bio.append(BIO2ID["B-ADR"])
            else:
                bio.append(BIO2ID["I-ADR"])
    enc["bio_labels"] = bio
    enc["has_adr"] = int(len(spans) > 0)
    return enc


# --- dataset construction --------------------------------------------------


def build_multitask_dataset(
    df: pd.DataFrame,
    tokenizer,
    silver_map: dict[int, list[tuple[int, int]]],
    max_length: int = 256,
) -> Dataset:
    rows = []
    for _, r in df.iterrows():
        spans = silver_map.get(int(r["review_id"]), [])
        enc = _spans_to_bio(r["review"], spans, tokenizer, max_length=max_length)
        rating_norm = (float(r["rating"]) - 1.0) / 9.0
        rows.append(
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "bio_labels": enc["bio_labels"],
                "rating_label": rating_norm,
                "adr_weight": float(enc["has_adr"]),
            }
        )
    return Dataset.from_list(rows)


# --- train loop ------------------------------------------------------------


def _collate(batch, pad_token_id: int = 0):
    max_len = max(len(x["input_ids"]) for x in batch)

    def pad(seq, val):
        return seq + [val] * (max_len - len(seq))

    input_ids = torch.tensor([pad(x["input_ids"], pad_token_id) for x in batch])
    attention_mask = torch.tensor([pad(x["attention_mask"], 0) for x in batch])
    bio_labels = torch.tensor([pad(x["bio_labels"], -100) for x in batch])
    rating_label = torch.tensor([x["rating_label"] for x in batch], dtype=torch.float32)
    adr_weight = torch.tensor([x["adr_weight"] for x in batch], dtype=torch.float32)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "bio_labels": bio_labels,
        "rating_label": rating_label,
        "adr_weight": adr_weight,
    }


def train_multitask(
    df: pd.DataFrame,
    silver_map: dict[int, list[tuple[int, int]]],
    *,
    model_name: str,
    out_dir: Path,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 256,
    lambda_rating: float = 1.0,
    lambda_adr: float = 1.0,
    seed: int = 42,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MultitaskHead(model_name).to(device)

    train = df.loc[df["split"] == "train"]
    val = df.loc[df["split"] == "val"]
    train_ds = build_multitask_dataset(train, tokenizer, silver_map, max_length)
    val_ds = build_multitask_dataset(val, tokenizer, silver_map, max_length)

    pad = tokenizer.pad_token_id or 0
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: _collate(b, pad_token_id=pad),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=lambda b: _collate(b, pad_token_id=pad),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_dl) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps
    )

    best_val_mae = float("inf")
    history = []
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                rating_label=batch["rating_label"],
                bio_labels=batch["bio_labels"],
                adr_weight=batch["adr_weight"],
                lambda_rating=lambda_rating,
                lambda_adr=lambda_adr,
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += out.loss.item()
            if (step + 1) % 100 == 0:
                logger.info(
                    "epoch %d step %d loss=%.4f", epoch, step + 1, running / (step + 1)
                )

        # val
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                preds.append(out.rating_pred.cpu().numpy())
                trues.append(batch["rating_label"].cpu().numpy())
        pred_ratings = np.clip(np.concatenate(preds) * 9.0 + 1.0, 1.0, 10.0)
        true_ratings = np.concatenate(trues) * 9.0 + 1.0
        m = regression_metrics(true_ratings, pred_ratings)
        history.append({"epoch": epoch, "val": m.as_dict()})
        logger.info("epoch %d val MAE=%.3f acc_within_1=%.3f", epoch, m.mae, m.acc_within_1)
        if m.mae < best_val_mae:
            best_val_mae = m.mae
            torch.save(model.state_dict(), out_dir / "multitask_best.pt")
            tokenizer.save_pretrained(out_dir)

    payload = {
        "model": model_name,
        "best_val_mae": float(best_val_mae),
        "history": history,
        "lambda_rating": lambda_rating,
        "lambda_adr": lambda_adr,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, default=float))
    return payload


# --- prediction ------------------------------------------------------------


def predict(
    texts: list[str],
    model: MultitaskHead,
    tokenizer,
    *,
    device: torch.device,
    max_length: int = 256,
) -> list[dict]:
    """Return per-text dicts with rating, BIO tags, and decoded ADR spans."""
    model.eval()
    out = []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = enc.pop("offset_mapping")[0].tolist()
            enc = {k: v.to(device) for k, v in enc.items()}
            o = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            rating = float(o.rating_pred.item()) * 9.0 + 1.0
            tag_ids = o.bio_logits.argmax(dim=-1)[0].cpu().tolist()
            tags = [ID2BIO[i] for i in tag_ids]
            spans = _decode_bio(offsets, tags)
            out.append({"rating": rating, "bio_tags": tags, "spans": spans})
    return out


def _decode_bio(offsets: list[list[int]], tags: list[str]) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    cur_start: int | None = None
    cur_end: int | None = None
    for (s, e), tag in zip(offsets, tags):
        if s == 0 and e == 0:
            continue
        if tag == "B-ADR":
            if cur_start is not None:
                spans.append((cur_start, cur_end, "ADR"))
            cur_start, cur_end = s, e
        elif tag == "I-ADR" and cur_start is not None:
            cur_end = e
        else:
            if cur_start is not None:
                spans.append((cur_start, cur_end, "ADR"))
                cur_start = cur_end = None
    if cur_start is not None:
        spans.append((cur_start, cur_end, "ADR"))
    return spans


# --- entrypoint ------------------------------------------------------------


def run(
    data_path: Path,
    silver_path: Path | None,
    model_name: str,
    out_dir: Path,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
) -> dict:
    df = pd.read_parquet(data_path)
    if silver_path is None or not silver_path.exists():
        logger.info("Building silver ADR spans with scispaCy on the train split...")
        train = df.loc[df["split"] == "train"]
        silver = build_silver_spans(train)
        out_dir.mkdir(parents=True, exist_ok=True)
        silver.to_parquet(out_dir / "silver_spans.parquet", index=False)
    else:
        silver = pd.read_parquet(silver_path)

    silver_map: dict[int, list[tuple[int, int]]] = {}
    for _, r in silver.iterrows():
        silver_map[int(r["review_id"])] = [tuple(s) for s in r["silver_spans"]]

    return train_multitask(
        df=df,
        silver_map=silver_map,
        model_name=model_name,
        out_dir=out_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_length=max_length,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/clean.parquet"))
    p.add_argument("--silver", type=Path, default=None,
                   help="Pre-computed silver spans parquet (skips scispaCy pass)")
    p.add_argument("--model", default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    p.add_argument("--out", type=Path, default=Path("models/multitask"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=256)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    payload = run(
        data_path=args.data,
        silver_path=args.silver,
        model_name=args.model,
        out_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
    )
    print(json.dumps(payload, indent=2, default=float))


if __name__ == "__main__":
    main()
