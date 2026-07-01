"""TF-IDF + Logistic Regression baselines for Tasks A and B.

Run as a module to fit, evaluate on val/test, and dump:
  models/baseline_<task>.joblib       — fitted Pipeline
  results/baseline_<task>_metrics.json — metrics
  results/baseline_<task>_predictions.parquet — per-row preds on val/test

The user explicitly asked for this to be the FIRST runnable thing in the
project: a clean reference number for the transformer fine-tunes to beat.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from .evaluate import (
    classification_metrics,
    pretty_classification_report,
    regression_metrics,
)

logger = logging.getLogger(__name__)

Task = Literal["A", "B"]


# --- pipelines --------------------------------------------------------------


def build_classification_pipeline(
    *, max_features: int = 200_000, ngram_max: int = 2, C: float = 4.0
) -> Pipeline:
    """TF-IDF (1-2 gram, sublinear, English stopwords) + multinomial LogReg.

    `class_weight='balanced'` because the 3-class bucket is roughly 35/15/50 —
    not catastrophic but worth correcting given the user-facing macro-F1 metric.
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, ngram_max),
                    max_features=max_features,
                    min_df=3,
                    sublinear_tf=True,
                    stop_words="english",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    C=C,
                    class_weight="balanced",
                    max_iter=2000,
                    n_jobs=None,
                    random_state=42,
                ),
            ),
        ]
    )


def build_regression_pipeline(
    *, max_features: int = 200_000, ngram_max: int = 2, alpha: float = 1.0
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, ngram_max),
                    max_features=max_features,
                    min_df=3,
                    sublinear_tf=True,
                    stop_words="english",
                ),
            ),
            ("reg", Ridge(alpha=alpha, random_state=42)),
        ]
    )


# --- training ---------------------------------------------------------------


def _split(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.loc[df["split"] == name]
    if out.empty:
        raise ValueError(f"split '{name}' is empty — did you run ingest.py?")
    return out


def fit_task_b(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    train = _split(df, "train")
    val = _split(df, "val")
    test = _split(df, "test")

    pipe = build_classification_pipeline()
    logger.info("Fitting Task B baseline on %d rows", len(train))
    pipe.fit(train["review"].to_numpy(), train["sentiment_3"].astype(str).to_numpy())

    out: dict = {}
    preds: dict = {}
    for name, frame in (("val", val), ("test", test)):
        y_true = frame["sentiment_3"].astype(str).to_numpy()
        y_pred = pipe.predict(frame["review"].to_numpy())
        m = classification_metrics(y_true, y_pred)
        out[name] = m.as_dict()
        preds[name] = pd.DataFrame(
            {
                "review_id": frame["review_id"].to_numpy(),
                "y_true": y_true,
                "y_pred": y_pred,
            }
        )
        logger.info(
            "[%s] acc=%.3f macro-F1=%.3f", name, m.accuracy, m.macro_f1
        )
        logger.info("\n%s", pretty_classification_report(y_true, y_pred))
    return pipe, {"metrics": out, "predictions": preds}


def fit_task_a(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    train = _split(df, "train")
    val = _split(df, "val")
    test = _split(df, "test")

    pipe = build_regression_pipeline()
    logger.info("Fitting Task A baseline (Ridge) on %d rows", len(train))
    pipe.fit(train["review"].to_numpy(), train["rating"].to_numpy())

    out: dict = {}
    preds: dict = {}
    for name, frame in (("val", val), ("test", test)):
        y_true = frame["rating"].to_numpy()
        y_pred = np.clip(pipe.predict(frame["review"].to_numpy()), 1.0, 10.0)
        m = regression_metrics(y_true, y_pred)
        out[name] = m.as_dict()
        preds[name] = pd.DataFrame(
            {
                "review_id": frame["review_id"].to_numpy(),
                "y_true": y_true,
                "y_pred": y_pred,
            }
        )
        logger.info(
            "[%s] MAE=%.3f RMSE=%.3f acc_within_1=%.3f",
            name, m.mae, m.rmse, m.acc_within_1,
        )
    return pipe, {"metrics": out, "predictions": preds}


# --- entrypoint -------------------------------------------------------------


def run(data_path: Path, task: Task, models_dir: Path, results_dir: Path) -> dict:
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(data_path)

    if task == "B":
        pipe, payload = fit_task_b(df)
    elif task == "A":
        pipe, payload = fit_task_a(df)
    else:
        raise ValueError(f"unsupported task: {task!r}")

    model_path = models_dir / f"baseline_{task}.joblib"
    joblib.dump(pipe, model_path)
    logger.info("Saved %s", model_path)

    metrics_path = results_dir / f"baseline_{task}_metrics.json"
    metrics_path.write_text(json.dumps(payload["metrics"], indent=2, default=float))

    pred_path = results_dir / f"baseline_{task}_predictions.parquet"
    pd.concat(
        {name: payload["predictions"][name] for name in ("val", "test")},
        names=["split", "row"],
    ).reset_index(level=0).to_parquet(pred_path, index=False)

    return payload["metrics"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/clean.parquet"))
    p.add_argument("--task", choices=["A", "B"], default="B")
    p.add_argument("--models_dir", type=Path, default=Path("models"))
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    metrics = run(args.data, args.task, args.models_dir, args.results_dir)
    print(json.dumps(metrics, indent=2, default=float))


if __name__ == "__main__":
    main()
