"""Text cleaning and condition normalization for drug reviews.

The Drugs.com export ships with HTML-escaped reviews ("&quot;", "&#039;") and
extremely noisy `condition` strings — including a literal sentinel that contains
the substring "users found this comment helpful" appearing for ~1.2k rows. This
module normalizes both into something downstream tasks can rely on.

The text cleaner is intentionally conservative: it does NOT lowercase or strip
punctuation, since the transformer tokenizers handle that for us, and the
TF-IDF baseline does its own lowercasing through `TfidfVectorizer`.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

import pandas as pd


# Sentinel garbage condition strings observed in the Drugs.com export.
_BAD_CONDITION_SUBSTRINGS = (
    "users found this comment helpful",
    "</span>",
)

# Regex for collapsing repeated whitespace including the literal `\r\n` that
# survived TSV escaping in the raw download.
_WHITESPACE_RE = re.compile(r"\s+")
_LITERAL_NEWLINE_RE = re.compile(r"\\r\\n|\r\n|\\n|\\r")


@dataclass(frozen=True)
class CleaningStats:
    n_in: int
    n_out: int
    n_dropped_empty_review: int
    n_dropped_bad_condition: int
    n_dropped_bad_rating: int


def clean_review_text(text: str | None) -> str:
    """Unescape HTML entities and collapse whitespace.

    Returns the empty string for non-string / null input.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)
    s = _LITERAL_NEWLINE_RE.sub(" ", s)
    s = html.unescape(s)
    s = s.replace("&#039;", "'").replace("&quot;", '"').replace("&amp;", "&")
    # Some reviews are wrapped in a single pair of double quotes from CSV escaping.
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def normalize_condition(cond: str | None) -> str | None:
    """Lower-case and trim a condition label; return None if it's the sentinel
    garbage string or empty."""
    if cond is None or (isinstance(cond, float) and pd.isna(cond)):
        return None
    s = str(cond).strip().lower()
    if not s:
        return None
    for bad in _BAD_CONDITION_SUBSTRINGS:
        if bad in s:
            return None
    # Drugs.com sometimes writes "Not Listed / Othe" — preserve it as a label
    # but normalize whitespace.
    s = _WHITESPACE_RE.sub(" ", s)
    return s


def rating_to_3class(rating: float) -> str:
    """Bucket a 1-10 rating into negative / neutral / positive (Task B)."""
    if rating <= 4:
        return "negative"
    if rating <= 6:
        return "neutral"
    return "positive"


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningStats]:
    """Apply review/condition/rating cleaning to a raw frame and return stats."""
    n_in = len(df)
    df = df.copy()
    df["review"] = df["review"].map(clean_review_text)
    df["condition"] = df["condition"].map(normalize_condition)

    # rating may arrive as string in the raw TSV
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    bad_rating = df["rating"].isna() | (df["rating"] < 1) | (df["rating"] > 10)
    empty_review = df["review"].str.len().fillna(0) < 5
    bad_condition = df["condition"].isna()

    n_bad_rating = int(bad_rating.sum())
    n_empty = int(empty_review.sum())
    n_bad_cond = int(bad_condition.sum())

    keep = ~(bad_rating | empty_review)
    df = df.loc[keep].reset_index(drop=True)

    df["sentiment_3"] = df["rating"].map(rating_to_3class).astype("category")
    df["n_tokens"] = df["review"].str.split().str.len().astype("int32")

    return df, CleaningStats(
        n_in=n_in,
        n_out=len(df),
        n_dropped_empty_review=n_empty,
        n_dropped_bad_condition=n_bad_cond,
        n_dropped_bad_rating=n_bad_rating,
    )
