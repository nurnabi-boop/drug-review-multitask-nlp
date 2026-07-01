"""Metrics for sentiment regression / classification / ADR extraction.

Centralized so the baseline, transformer, and multi-task scripts all report
the exact same numbers in the exact same shape.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)


SENTIMENT_LABELS = ["negative", "neutral", "positive"]


@dataclass
class RegressionMetrics:
    mae: float
    rmse: float
    acc_within_1: float  # fraction with |pred - true| <= 1 after rounding
    n: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    per_class: dict
    confusion: list  # 3x3 list-of-list
    n: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SpanMetrics:
    precision: float
    recall: float
    f1: float
    n_pred: int
    n_true: int
    n_match: int

    def as_dict(self) -> dict:
        return asdict(self)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    acc1 = float((np.abs(np.round(y_pred) - np.round(y_true)) <= 1).mean())
    return RegressionMetrics(mae=mae, rmse=rmse, acc_within_1=acc1, n=len(y_true))


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str] | None = None
) -> ClassificationMetrics:
    labels = labels or SENTIMENT_LABELS
    acc = float(accuracy_score(y_true, y_pred))
    macro = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
        for i, lab in enumerate(labels)
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return ClassificationMetrics(
        accuracy=acc, macro_f1=macro, per_class=per_class, confusion=cm, n=len(y_true)
    )


def span_metrics(
    pred_spans: list[set[tuple[int, int, str]]],
    true_spans: list[set[tuple[int, int, str]]],
) -> SpanMetrics:
    """Strict span-match P/R/F1: a prediction is correct iff (start, end, label)
    matches an annotated span. Inputs are per-document sets of (start, end, label).
    """
    if len(pred_spans) != len(true_spans):
        raise ValueError("pred_spans and true_spans must align by document")
    n_pred = sum(len(s) for s in pred_spans)
    n_true = sum(len(s) for s in true_spans)
    n_match = sum(len(p & t) for p, t in zip(pred_spans, true_spans))
    precision = n_match / n_pred if n_pred else 0.0
    recall = n_match / n_true if n_true else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return SpanMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        n_pred=n_pred,
        n_true=n_true,
        n_match=n_match,
    )


def pretty_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str] | None = None
) -> str:
    return classification_report(
        y_true, y_pred, labels=labels or SENTIMENT_LABELS, digits=3, zero_division=0
    )
