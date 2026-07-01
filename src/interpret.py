"""SHAP word-level attributions on the fine-tuned transformer.

Wraps `shap.Explainer` with HF tokenizer + classification head. Designed to
be called from a notebook OR from the Streamlit demo, so the heavy
explainer construction is cached.
"""

from __future__ import annotations

import argparse
import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .transformer_models import ID2LABEL

logger = logging.getLogger(__name__)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=4)
def _load(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(_device())
    model.eval()
    return tokenizer, model


def _build_predict_fn(tokenizer, model, max_length: int = 256):
    device = _device()

    def predict(texts):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        enc = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        # SHAP wants probabilities for classification or scalar for regression
        if logits.shape[-1] == 1:
            return logits.squeeze(-1).cpu().numpy()
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    return predict


def build_explainer(model_dir: str | Path, max_length: int = 256):
    tokenizer, model = _load(str(model_dir))
    predict = _build_predict_fn(tokenizer, model, max_length=max_length)
    masker = shap.maskers.Text(tokenizer)
    is_regression = model.config.num_labels == 1
    output_names = None if is_regression else [ID2LABEL[i] for i in range(model.config.num_labels)]
    explainer = shap.Explainer(predict, masker, output_names=output_names)
    return explainer


def explain_texts(model_dir: str | Path, texts: list[str], max_length: int = 256):
    explainer = build_explainer(model_dir, max_length=max_length)
    return explainer(texts)


def explain_to_html(
    model_dir: str | Path, texts: list[str], out_html: Path, max_length: int = 256
) -> Path:
    sv = explain_texts(model_dir, texts, max_length=max_length)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    html = shap.plots.text(sv, display=False)
    out_html.write_text(html, encoding="utf-8")
    logger.info("Wrote SHAP HTML to %s", out_html)
    return out_html


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_dir", type=Path, required=True,
                   help="HF model directory (e.g. models/distilbert-base-uncased_B/best)")
    p.add_argument("--data", type=Path, default=Path("data/processed/clean.parquet"))
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--bucket", choices=["negative", "neutral", "positive"], default="negative")
    p.add_argument("--out_html", type=Path, default=Path("results/shap_negative_samples.html"))
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    df = pd.read_parquet(args.data)
    pool = df.loc[
        (df["split"] == "test") & (df["sentiment_3"].astype(str) == args.bucket),
        "review",
    ]
    if pool.empty:
        raise SystemExit(f"No test reviews in bucket {args.bucket}")
    texts = pool.sample(min(args.n, len(pool)), random_state=42).astype(str).to_list()
    explain_to_html(args.model_dir, texts, args.out_html)
    print(json.dumps({"out_html": str(args.out_html), "n": len(texts)}, indent=2))


if __name__ == "__main__":
    main()
