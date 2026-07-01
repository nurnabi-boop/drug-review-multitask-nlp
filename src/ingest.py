"""Convert raw UCI/Kaggle TSV exports into a single clean parquet.

Supports both the Drugs.com files (`drugsComTrain_raw.tsv`,
`drugsComTest_raw.tsv`) and the Druglib.com files (`drugLibTrain_raw.tsv`,
`drugLibTest_raw.tsv`). The two have slightly different schemas so we
normalize them to a common one and write a single clean parquet plus a
sidecar JSON of stats.

The original train/test split on disk is *not* preserved — we re-split
80/10/10 stratified on `sentiment_3` so all downstream scripts share
identical partitions regardless of which raw file was provided.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .preprocessing import clean_dataframe

logger = logging.getLogger(__name__)


# --- raw schema mapping ----------------------------------------------------


def _read_drugscom(raw_dir: Path) -> pd.DataFrame:
    """Drugs.com schema:
    uniqueID, drugName, condition, review, rating, date, usefulCount.
    """
    parts = []
    for fname in ("drugsComTrain_raw.tsv", "drugsComTest_raw.tsv"):
        p = raw_dir / fname
        if not p.exists():
            continue
        df = pd.read_csv(p, sep="\t", quoting=3)  # QUOTE_NONE
        df = df.rename(
            columns={
                "uniqueID": "review_id",
                "drugName": "drug",
                "usefulCount": "useful_count",
            }
        )
        df["source"] = "drugscom"
        parts.append(df)
    if not parts:
        raise FileNotFoundError(
            f"No Drugs.com files in {raw_dir}. Expected drugsCom{{Train,Test}}_raw.tsv"
        )
    return pd.concat(parts, ignore_index=True)


def _read_druglib(raw_dir: Path) -> pd.DataFrame:
    """Druglib.com schema is richer: separate benefitsReview, sideEffectsReview,
    commentsReview, plus sideEffects (categorical) and effectiveness columns.

    For multi-task training we concatenate the three free-text columns into a
    single `review` and keep `side_effects_label` (categorical) as auxiliary.
    """
    parts = []
    for fname in ("drugLibTrain_raw.tsv", "drugLibTest_raw.tsv"):
        p = raw_dir / fname
        if not p.exists():
            continue
        df = pd.read_csv(p, sep="\t")
        df = df.rename(
            columns={
                "Unnamed: 0": "review_id",
                "urlDrugName": "drug",
                "effectiveness": "effectiveness_label",
                "sideEffects": "side_effects_label",
            }
        )
        df["review"] = (
            df["benefitsReview"].fillna("").astype(str)
            + " "
            + df["sideEffectsReview"].fillna("").astype(str)
            + " "
            + df["commentsReview"].fillna("").astype(str)
        ).str.strip()
        df["useful_count"] = 0  # not present in druglib
        df["date"] = pd.NaT
        df["source"] = "druglib"
        parts.append(df)
    if not parts:
        raise FileNotFoundError(
            f"No Druglib files in {raw_dir}. Expected drugLib{{Train,Test}}_raw.tsv"
        )
    return pd.concat(parts, ignore_index=True)


def _detect_and_load(raw_dir: Path) -> pd.DataFrame:
    drugscom_present = (raw_dir / "drugsComTrain_raw.tsv").exists()
    druglib_present = (raw_dir / "drugLibTrain_raw.tsv").exists()
    parts = []
    if drugscom_present:
        parts.append(_read_drugscom(raw_dir))
    if druglib_present:
        parts.append(_read_druglib(raw_dir))
    if not parts:
        raise FileNotFoundError(
            f"No supported raw files found in {raw_dir}. Expected one of:\n"
            "  drugsComTrain_raw.tsv / drugsComTest_raw.tsv  (Drugs.com)\n"
            "  drugLibTrain_raw.tsv / drugLibTest_raw.tsv    (Druglib.com)"
        )
    df = pd.concat(parts, ignore_index=True)
    # Parse date if string
    if df["date"].dtype == object:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


# --- split ------------------------------------------------------------------


def _stratified_three_split(
    df: pd.DataFrame, seed: int = 42, val_frac: float = 0.10, test_frac: float = 0.10
) -> pd.Series:
    """Return a 'split' Series with values train/val/test, stratified on sentiment_3."""
    idx = df.index.to_numpy()
    y = df["sentiment_3"].astype(str).to_numpy()

    train_idx, test_idx = train_test_split(
        idx, test_size=test_frac, random_state=seed, stratify=y
    )
    rel_val = val_frac / (1.0 - test_frac)
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=rel_val,
        random_state=seed,
        stratify=df.loc[train_idx, "sentiment_3"].astype(str).to_numpy(),
    )
    split = pd.Series("train", index=df.index, dtype="object")
    split.loc[val_idx] = "val"
    split.loc[test_idx] = "test"
    return split.astype("category")


# --- entrypoint -------------------------------------------------------------


def run(raw_dir: Path, out_dir: Path, seed: int = 42) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = _detect_and_load(raw_dir)
    logger.info("Loaded %d raw rows from %s", len(raw), raw_dir)

    # canonicalize columns we care about
    keep = [
        "review_id",
        "drug",
        "condition",
        "review",
        "rating",
        "date",
        "useful_count",
        "source",
    ]
    aux_cols = [c for c in ("side_effects_label", "effectiveness_label") if c in raw.columns]
    raw = raw[[c for c in keep if c in raw.columns] + aux_cols]

    clean, stats = clean_dataframe(raw)
    clean["split"] = _stratified_three_split(clean, seed=seed)

    out_path = out_dir / "clean.parquet"
    clean.to_parquet(out_path, index=False)

    stats_payload = {
        "n_in": stats.n_in,
        "n_out": stats.n_out,
        "n_dropped_empty_review": stats.n_dropped_empty_review,
        "n_dropped_bad_condition": stats.n_dropped_bad_condition,
        "n_dropped_bad_rating": stats.n_dropped_bad_rating,
        "split_counts": clean["split"].value_counts().to_dict(),
        "sentiment_counts": clean["sentiment_3"].value_counts().to_dict(),
        "n_drugs": int(clean["drug"].nunique()),
        "n_conditions": int(clean["condition"].dropna().nunique()),
        "median_tokens": int(clean["n_tokens"].median()),
    }
    (out_dir / "ingest_stats.json").write_text(json.dumps(stats_payload, indent=2, default=str))
    logger.info("Wrote %s (%d rows)", out_path, len(clean))
    return stats_payload


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    payload = run(args.raw_dir, args.out, seed=args.seed)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
