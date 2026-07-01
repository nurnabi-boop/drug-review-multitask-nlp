"""Streamlit demo for drug-reviews multi-task system.

Three tabs:
  1. Predict — paste a review, get rating + sentiment + highlighted ADR mentions.
  2. Drug safety dashboard — per-drug rating + ADR mention rate.
  3. Similar reviews — retrieve closest reviews from the indexed corpus.

Run:
    streamlit run app.py
"""

from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# --- paths ----------------------------------------------------------------

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "processed" / "clean.parquet"
ADR_PATH = ROOT / "results" / "adr_mentions.parquet"
DEFAULT_TX_DIR = ROOT / "models" / "distilbert-base-uncased_B" / "best"
RETRIEVAL_DIR = ROOT / "models" / "retrieval"


# --- caching --------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_reviews(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_adr(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_resource(show_spinner=False)
def load_classifier(model_dir: Path):
    if not model_dir.exists():
        return None
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner=False)
def load_scispacy_pipe():
    try:
        from src.adr_extraction import load_scispacy

        return load_scispacy()
    except Exception as e:  # noqa: BLE001
        st.warning(f"scispaCy not available: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_retrieval(index_dir: Path):
    if not index_dir.exists():
        return None
    from src.retrieval import load_index

    return load_index(index_dir)


# --- prediction helpers ---------------------------------------------------


SENTIMENT_LABELS = ["negative", "neutral", "positive"]


def predict_sentiment(text: str, tokenizer, model) -> tuple[str, np.ndarray]:
    import torch

    enc = tokenizer(text, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits[0]
    probs = torch.softmax(logits, dim=-1).numpy()
    label = SENTIMENT_LABELS[int(np.argmax(probs))]
    return label, probs


def extract_adr_spans(text: str, nlp) -> list[tuple[int, int, str]]:
    if nlp is None:
        return []
    doc = nlp(text)
    return [
        (ent.start_char, ent.end_char, ent.label_)
        for ent in doc.ents
        if ent.label_ in ("DISEASE", "CHEMICAL")
    ]


def render_highlighted(text: str, spans: list[tuple[int, int, str]]) -> str:
    """Render text with <mark> spans for ADR mentions."""
    if not spans:
        return f"<div style='line-height:1.6;font-size:1rem'>{escape(text)}</div>"
    spans_sorted = sorted(spans, key=lambda x: x[0])
    out = []
    cursor = 0
    color = {"DISEASE": "#ffd6d6", "CHEMICAL": "#d6ecff", "ADR": "#ffd6d6"}
    for s, e, label in spans_sorted:
        if s < cursor:
            continue
        out.append(escape(text[cursor:s]))
        bg = color.get(label, "#fff3a3")
        out.append(
            f"<mark style='background:{bg};padding:0 2px;border-radius:3px'>"
            f"{escape(text[s:e])}"
            f"<sub style='font-size:0.6em;opacity:.6;margin-left:2px'>{label}</sub>"
            f"</mark>"
        )
        cursor = e
    out.append(escape(text[cursor:]))
    return f"<div style='line-height:1.6;font-size:1rem'>{''.join(out)}</div>"


# --- UI -------------------------------------------------------------------


st.set_page_config(page_title="Drug Reviews — multi-task NLP", layout="wide")
st.title("Drug Reviews — multi-task NLP demo")
st.caption(
    "UCI Drug Review Dataset · sentiment + ADR extraction · "
    "research artifact, not a regulated pharmacovigilance tool."
)

reviews_df = load_reviews(DATA_PATH)
adr_df = load_adr(ADR_PATH)

with st.sidebar:
    st.header("Models")
    model_dir = Path(
        st.text_input(
            "Sentiment model directory",
            value=str(DEFAULT_TX_DIR),
            help="HF checkpoint dir from `python -m src.transformer_models ...`",
        )
    )
    use_scispacy = st.checkbox("Highlight ADR with scispaCy", value=True)
    st.markdown("---")
    st.header("Status")
    st.write(f"Reviews loaded: **{len(reviews_df):,}**" if len(reviews_df) else "_no clean.parquet_")
    st.write(f"ADR mentions loaded: **{len(adr_df):,}**" if len(adr_df) else "_no adr_mentions.parquet_")

clf = load_classifier(model_dir)
nlp = load_scispacy_pipe() if use_scispacy else None

tab_predict, tab_safety, tab_similar = st.tabs(
    ["Predict", "Drug safety dashboard", "Similar reviews"]
)


# ------------------- Tab 1: Predict ---------------------------------------

with tab_predict:
    st.subheader("Paste a patient review")
    default_text = (
        "I started taking this medication a week ago for chronic migraines. "
        "It actually helped with the headaches, but it gave me terrible nausea "
        "and I felt dizzy whenever I stood up. After three days I had to stop."
    )
    text = st.text_area("Review text", value=default_text, height=180)
    go = st.button("Predict")

    if go and text.strip():
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Sentiment / rating**")
            if clf is None:
                st.info(
                    f"No fine-tuned model at `{model_dir}`. Train one with:\n"
                    "`python -m src.transformer_models --task B`"
                )
            else:
                tokenizer, model = clf
                label, probs = predict_sentiment(text, tokenizer, model)
                st.metric("Predicted bucket", label)
                prob_df = pd.DataFrame({"label": SENTIMENT_LABELS, "prob": probs})
                st.bar_chart(prob_df.set_index("label"))
        with col2:
            st.markdown("**ADR mentions (scispaCy bc5cdr)**")
            spans = extract_adr_spans(text, nlp) if use_scispacy else []
            st.markdown(render_highlighted(text, spans), unsafe_allow_html=True)
            if spans:
                st.caption(f"{len(spans)} mention(s) flagged.")


# ------------------- Tab 2: Drug safety dashboard -------------------------

with tab_safety:
    st.subheader("Per-drug summary")
    if reviews_df.empty:
        st.info("Run `python -m src.ingest` first.")
    else:
        min_n = st.slider("Minimum reviews per drug", 30, 1000, 100, step=10)
        agg = (
            reviews_df.groupby("drug")
            .agg(n=("review_id", "count"), mean_rating=("rating", "mean"))
            .reset_index()
        )
        agg = agg.loc[agg["n"] >= min_n]

        if not adr_df.empty:
            adr_per_review = (
                adr_df.groupby("review_id").size().rename("n_spans").reset_index()
            )
            merged = reviews_df[["review_id", "drug"]].merge(
                adr_per_review, on="review_id", how="left"
            )
            merged["has_adr"] = merged["n_spans"].fillna(0) > 0
            adr_rate = (
                merged.groupby("drug")["has_adr"].mean().rename("adr_rate").reset_index()
            )
            agg = agg.merge(adr_rate, on="drug", how="left")
        else:
            agg["adr_rate"] = np.nan

        agg = agg.sort_values("mean_rating")
        st.markdown(f"**{len(agg):,} drugs** with ≥ {min_n} reviews.")
        st.dataframe(
            agg.rename(
                columns={
                    "drug": "Drug",
                    "n": "Reviews",
                    "mean_rating": "Mean rating",
                    "adr_rate": "ADR mention rate",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        if not agg.empty:
            top_low = agg.nsmallest(20, "mean_rating")
            fig = px.bar(
                top_low,
                x="mean_rating",
                y="drug",
                orientation="h",
                color="adr_rate" if "adr_rate" in top_low else None,
                color_continuous_scale="Reds",
                title="20 lowest-rated drugs (color = ADR mention rate)",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=600)
            st.plotly_chart(fig, use_container_width=True)


# ------------------- Tab 3: Similar reviews -------------------------------

with tab_similar:
    st.subheader("Find similar reviews from the corpus")
    if reviews_df.empty:
        st.info("Run `python -m src.ingest` first.")
    else:
        index = load_retrieval(RETRIEVAL_DIR)
        if index is None:
            st.info(
                f"No retrieval index at `{RETRIEVAL_DIR}`. Build one with:\n"
                "`python -m src.retrieval`"
            )
        else:
            q_text = st.text_area(
                "Query review",
                value="terrible nausea after starting this medication",
                height=120,
            )
            k = st.slider("k", 1, 20, 5)
            if st.button("Search"):
                from src.retrieval import query

                hits = query(index, q_text, k=k)
                rows = []
                lookup = reviews_df.set_index("review_id")
                for rid, score in hits:
                    if rid not in lookup.index:
                        continue
                    r = lookup.loc[rid]
                    rows.append(
                        {
                            "score": round(score, 3),
                            "drug": r["drug"],
                            "condition": r["condition"],
                            "rating": r["rating"],
                            "review": r["review"][:300] + ("…" if len(r["review"]) > 300 else ""),
                        }
                    )
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
