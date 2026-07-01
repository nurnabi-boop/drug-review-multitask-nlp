"""ADR (adverse drug reaction) mention extraction with scispaCy.

We use `en_ner_bc5cdr_md` because BC5CDR was trained on chemical/disease
mentions in PubMed abstracts, which transfers reasonably well to patient
language about side effects ("nausea", "headache", "constipation"). The
model emits CHEMICAL and DISEASE entities; for ADR purposes we treat
DISEASE entities as candidate ADR mentions.

We additionally run a small heuristic over an ADR cue lexicon (`gave me`,
`caused`, `made me feel`, ...) to bias downstream filtering. The lexicon is
NOT used to *create* spans, only to mark a per-mention `is_adr_cue` flag
that the multi-task model and dashboard can filter on.

Outputs a tidy parquet with one row per (review_id, span):
  review_id, start, end, text, label, sent_text, is_adr_cue, drug, condition

Also provides `bootstrap_annotations()` which writes the empty 200-review
sample at `annotations/adr_manual_200.csv` for the user to hand-label.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


# --- ADR cue lexicon -------------------------------------------------------
# Conservative phrases that strongly suggest the writer is reporting an
# adverse effect they personally experienced. Used as a per-span flag, not
# as a span generator.
ADR_CUES = [
    r"\bside[- ]effect(s)?\b",
    r"\bgave me\b",
    r"\bmade me (feel )?(sick|nauseous|dizzy|drowsy|tired|exhausted)\b",
    r"\b(experienc|suffer|develop)(ed|ing)\b",
    r"\bcaus(ed|ing|es)\b",
    r"\bafter taking\b",
    r"\bafter (a|the) (dose|pill)\b",
    r"\bhad (terrible|bad|awful|severe|horrible)\b",
    r"\bi (got|started|began) (to )?\b",
    r"\b(adverse|unwanted) (reaction|effect)s?\b",
]
_CUE_RE = re.compile("|".join(ADR_CUES), flags=re.IGNORECASE)


def has_adr_cue(text: str) -> bool:
    return bool(_CUE_RE.search(text or ""))


# --- scispaCy model --------------------------------------------------------


def load_scispacy(model_name: str = "en_ner_bc5cdr_md"):
    import spacy

    try:
        nlp = spacy.load(model_name)
    except OSError as e:
        raise RuntimeError(
            f"scispaCy model '{model_name}' not installed. Run:\n"
            "  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz"
        ) from e
    return nlp


# --- extraction -----------------------------------------------------------


def _iter_doc_spans(doc, *, keep_labels: tuple[str, ...] = ("DISEASE",)):
    """Yield dict rows for entities of interest in a parsed Doc."""
    for ent in doc.ents:
        if ent.label_ not in keep_labels:
            continue
        sent_text = ent.sent.text if ent.sent is not None else ""
        yield {
            "start": int(ent.start_char),
            "end": int(ent.end_char),
            "text": ent.text,
            "label": ent.label_,
            "sent_text": sent_text,
            "is_adr_cue": has_adr_cue(sent_text),
        }


def extract_corpus(
    df: pd.DataFrame,
    *,
    nlp,
    text_col: str = "review",
    id_col: str = "review_id",
    batch_size: int = 64,
    n_process: int = 1,
    keep_labels: tuple[str, ...] = ("DISEASE", "CHEMICAL"),
) -> pd.DataFrame:
    """Run scispaCy NER over `df[text_col]` and return a tidy span dataframe.

    `n_process > 1` is faster on Linux/macOS but unstable on Windows + spaCy +
    transformer-component models, so we default to 1.
    """
    rows: list[dict] = []
    texts = df[text_col].astype(str).to_list()
    ids = df[id_col].to_list()
    drugs = df["drug"].to_list() if "drug" in df.columns else [None] * len(df)
    conds = df["condition"].to_list() if "condition" in df.columns else [None] * len(df)

    for i, doc in enumerate(
        nlp.pipe(texts, batch_size=batch_size, n_process=n_process)
    ):
        for span in _iter_doc_spans(doc, keep_labels=keep_labels):
            span["review_id"] = ids[i]
            span["drug"] = drugs[i]
            span["condition"] = conds[i]
            rows.append(span)
        if (i + 1) % 5000 == 0:
            logger.info("scispaCy processed %d / %d reviews", i + 1, len(texts))

    if not rows:
        return pd.DataFrame(
            columns=[
                "review_id", "start", "end", "text", "label",
                "sent_text", "is_adr_cue", "drug", "condition",
            ]
        )
    return pd.DataFrame(rows)


# --- 200-review annotation bootstrap --------------------------------------


def bootstrap_annotations(
    df: pd.DataFrame,
    out_path: Path,
    *,
    n: int = 200,
    seed: int = 42,
) -> Path:
    """Sample 200 reviews stratified by sentiment_3 and write an empty
    annotation CSV for the user to fill in.

    Schema:
        review_id, drug, condition, sentiment_3, review,
        adr_spans  (semicolon-separated 'start,end,label' triples; empty)

    The user opens this in Excel / a CSV editor, types span triples in the
    `adr_spans` column for each row, and saves. `load_annotations()` parses
    the result back into per-row sets of (start, end, label) tuples.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng_seed = seed
    per_class = max(1, n // 3)

    sub_frames = []
    for cls in ("negative", "neutral", "positive"):
        pool = df.loc[df["sentiment_3"].astype(str) == cls]
        if pool.empty:
            continue
        sub_frames.append(pool.sample(min(per_class, len(pool)), random_state=rng_seed))
    out = pd.concat(sub_frames, ignore_index=True).sample(frac=1, random_state=rng_seed)
    if len(out) > n:
        out = out.iloc[:n]

    annotation_df = pd.DataFrame(
        {
            "review_id": out["review_id"].to_numpy(),
            "drug": out["drug"].to_numpy(),
            "condition": out["condition"].to_numpy(),
            "sentiment_3": out["sentiment_3"].astype(str).to_numpy(),
            "review": out["review"].to_numpy(),
            "adr_spans": "",
        }
    )
    annotation_df.to_csv(out_path, index=False, encoding="utf-8")

    readme = out_path.parent / "README.md"
    readme.write_text(
        "# ADR manual annotations (200-review subset)\n\n"
        "Open `adr_manual_200.csv` in Excel / VS Code / your editor of choice.\n"
        "For each row, fill the `adr_spans` column with semicolon-separated\n"
        "`start,end,label` triples, where `start` and `end` are character\n"
        "offsets into the `review` text.\n\n"
        "Labels:\n"
        "- **ADR**: a patient-reported adverse effect attributable to the drug.\n"
        "- **DRUG**: a drug name mentioned in the body (often a brand / synonym).\n"
        "- **SYMPTOM**: a symptom NOT clearly caused by the drug (negative class).\n\n"
        "Example: `42,49,ADR; 90,98,SYMPTOM`\n",
        encoding="utf-8",
    )
    logger.info("Wrote %d annotation seeds to %s", len(annotation_df), out_path)
    return out_path


def parse_span_field(s: str) -> set[tuple[int, int, str]]:
    """Parse 'start,end,label; start,end,label' into a set of triples."""
    if not isinstance(s, str) or not s.strip():
        return set()
    out: set[tuple[int, int, str]] = set()
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [c.strip() for c in chunk.split(",")]
        if len(parts) != 3:
            continue
        try:
            start = int(parts[0])
            end = int(parts[1])
        except ValueError:
            continue
        label = parts[2].upper()
        out.add((start, end, label))
    return out


def load_annotations(path: Path) -> pd.DataFrame:
    """Read a hand-labeled CSV and parse the `adr_spans` column to a set."""
    df = pd.read_csv(path)
    df["spans"] = df["adr_spans"].fillna("").map(parse_span_field)
    return df


# --- entrypoint ------------------------------------------------------------


def run(
    data_path: Path,
    out_path: Path,
    *,
    model_name: str = "en_ner_bc5cdr_md",
    sample: int | None = None,
    batch_size: int = 64,
    n_process: int = 1,
) -> dict:
    df = pd.read_parquet(data_path)
    if sample is not None:
        df = df.sample(min(len(df), sample), random_state=42).reset_index(drop=True)
    nlp = load_scispacy(model_name)
    spans = extract_corpus(
        df, nlp=nlp, batch_size=batch_size, n_process=n_process
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    spans.to_parquet(out_path, index=False)

    coverage = (
        spans["review_id"].nunique() / max(len(df), 1) if len(spans) else 0.0
    )
    summary = {
        "n_reviews": int(len(df)),
        "n_spans": int(len(spans)),
        "n_reviews_with_span": int(spans["review_id"].nunique()) if len(spans) else 0,
        "coverage": float(coverage),
        "label_counts": (
            spans["label"].value_counts().to_dict() if len(spans) else {}
        ),
    }
    (out_path.parent / (out_path.stem + "_summary.json")).write_text(
        json.dumps(summary, indent=2)
    )
    logger.info("Wrote %d spans to %s (coverage %.1f%%)", len(spans), out_path, 100 * coverage)
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/clean.parquet"))
    p.add_argument("--out", type=Path, default=Path("results/adr_mentions.parquet"))
    p.add_argument("--model", default="en_ner_bc5cdr_md")
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_process", type=int, default=1)
    p.add_argument(
        "--bootstrap-annotations",
        action="store_true",
        help="Write an empty 200-review CSV at annotations/adr_manual_200.csv",
    )
    p.add_argument("--annotations_out", type=Path, default=Path("annotations/adr_manual_200.csv"))
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()

    if args.bootstrap_annotations:
        df = pd.read_parquet(args.data)
        bootstrap_annotations(df, args.annotations_out)
        return

    summary = run(
        data_path=args.data,
        out_path=args.out,
        model_name=args.model,
        sample=args.sample,
        batch_size=args.batch_size,
        n_process=args.n_process,
    )
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
