"""Sentence-Transformers + FAISS retrieval for the 'similar reviews' tab.

Indexed corpus = the train split (so we never accidentally retrieve a held-out
test review against itself). Queries can be either the user's pasted-in review
or a stored review_id.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class RetrievalIndex:
    index: faiss.Index
    review_ids: np.ndarray
    embedder_name: str


_DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"


def build_index(
    df: pd.DataFrame,
    *,
    embedder_name: str = _DEFAULT_EMBEDDER,
    batch_size: int = 64,
    out_dir: Path | None = None,
) -> RetrievalIndex:
    embedder = SentenceTransformer(embedder_name)
    texts = df["review"].astype(str).to_list()
    logger.info("Encoding %d reviews with %s", len(texts), embedder_name)
    emb = embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(emb.shape[1])  # cosine via normalized inner product
    index.add(emb)
    review_ids = df["review_id"].to_numpy()

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(out_dir / "reviews.faiss"))
        np.save(out_dir / "review_ids.npy", review_ids)
        (out_dir / "embedder.txt").write_text(embedder_name, encoding="utf-8")

    return RetrievalIndex(index=index, review_ids=review_ids, embedder_name=embedder_name)


def load_index(index_dir: Path) -> RetrievalIndex:
    embedder_name = (index_dir / "embedder.txt").read_text(encoding="utf-8").strip()
    index = faiss.read_index(str(index_dir / "reviews.faiss"))
    review_ids = np.load(index_dir / "review_ids.npy")
    return RetrievalIndex(index=index, review_ids=review_ids, embedder_name=embedder_name)


def query(index: RetrievalIndex, text: str, *, k: int = 5) -> list[tuple[int, float]]:
    embedder = SentenceTransformer(index.embedder_name)
    q = embedder.encode([text], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, idxs = index.index.search(q, k)
    return [
        (int(index.review_ids[i]), float(s))
        for s, i in zip(scores[0], idxs[0])
        if i >= 0
    ]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/clean.parquet"))
    p.add_argument("--out", type=Path, default=Path("models/retrieval"))
    p.add_argument("--embedder", default=_DEFAULT_EMBEDDER)
    p.add_argument("--split", default="train")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    df = pd.read_parquet(args.data)
    sub = df.loc[df["split"] == args.split].reset_index(drop=True)
    build_index(sub, embedder_name=args.embedder, out_dir=args.out)
    print(f"Indexed {len(sub)} reviews in {args.out}")


if __name__ == "__main__":
    main()
