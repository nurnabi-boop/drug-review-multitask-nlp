# Drug Reviews — Multi-task NLP

Multi-task NLP system on the **UCI Drug Review Dataset** (Druglib.com + Drugs.com,
~215k patient reviews). Predicts sentiment from free-text reviews and flags
adverse drug reaction (ADR) mentions, with a deployable Streamlit demo.

- Dataset: <https://archive.ics.uci.edu/dataset/461/drug+review+dataset+drugs+com>
- Kaggle mirror: <https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018>

## Tasks

| Task | Description | Metric |
|------|-------------|--------|
| **A** | Sentiment regression: predict 1–10 rating from text | MAE, accuracy ±1 rating |
| **B** | Sentiment 3-class: negative (1–4) / neutral (5–6) / positive (7–10) | Macro-F1 |
| **C** | ADR mention extraction (scispaCy `en_ner_bc5cdr_md`) | P/R/F1 on 200-review hand-labeled set |

Multi-task model: shared encoder, two heads (rating regression + ADR token classification).

## Models compared

1. **TF-IDF + Logistic Regression** — bag-of-words baseline (Task B).
2. **DistilBERT** — fine-tuned on review text.
3. **PubMedBERT** — biomedical-pretrained transformer (the "domain pretraining
   advantage" hypothesis).
4. **Multi-task PubMedBERT** — shared encoder, dual heads.

## Quickstart

```bash
# 1. environment
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# scispacy NER + biomedical tokenizer
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

# 2. data — drop the Kaggle CSVs into ./data/raw/
#    drugsComTrain_raw.tsv, drugsComTest_raw.tsv  (Drugs.com split)
#    or  drugLibTrain_raw.tsv, drugLibTest_raw.tsv  (Druglib split)
python -m src.ingest --raw_dir data/raw --out data/processed

# 3. baseline — Task B reference number to beat
python -m src.baselines --data data/processed/clean.parquet --task B

# 4. transformer fine-tunes
python -m src.transformer_models --model distilbert-base-uncased --task B
python -m src.transformer_models --model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext --task B

# 5. ADR extraction over the corpus
python -m src.adr_extraction --data data/processed/clean.parquet --out results/adr_mentions.parquet

# 6. multi-task
python -m src.multitask_model --data data/processed/clean.parquet

# 7. demo
streamlit run app.py
```

## Project layout

```
drug-reviews/
├── data/
│   ├── raw/                    # original TSV/CSV from UCI/Kaggle
│   └── processed/              # cleaned parquet
├── src/
│   ├── ingest.py               # raw -> processed parquet
│   ├── preprocessing.py        # text cleaning, condition normalization
│   ├── baselines.py            # TF-IDF + LogReg
│   ├── transformer_models.py   # DistilBERT / PubMedBERT fine-tune
│   ├── multitask_model.py      # shared encoder, rating + ADR heads
│   ├── adr_extraction.py       # scispaCy bc5cdr NER pipeline
│   ├── evaluate.py             # metrics for tasks A/B/C
│   ├── interpret.py            # SHAP word-level attributions
│   └── retrieval.py            # sentence-transformer + FAISS for "similar reviews"
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_transformer_finetuning.ipynb
│   ├── 04_adr_extraction.ipynb
│   └── 05_drug_safety_dashboard.ipynb
├── annotations/
│   └── adr_manual_200.csv      # hand-labeled ADR spans (the defensible bit)
├── models/                     # serialized fitted estimators / HF checkpoints
├── results/                    # metric dumps, prediction parquets, plots
├── app.py                      # Streamlit demo
├── requirements.txt
└── README.md
```

## Data layout (after `ingest.py`)

`data/processed/clean.parquet` — schema:

| column | dtype | notes |
|--------|-------|-------|
| `review_id` | int64 | from original index |
| `drug` | string | original `drugName` |
| `condition` | string | normalized lower-case, NA-cleaned |
| `review` | string | unescaped HTML, deduped whitespace |
| `rating` | float | 1–10 (Drugs.com) or 1–10 (Druglib) |
| `date` | datetime64 | parsed from `date` column |
| `useful_count` | int | helpful votes |
| `n_tokens` | int | whitespace token count (cached for EDA) |
| `sentiment_3` | category | neg / neu / pos derived from rating |
| `split` | category | train / val / test |

A stratified 80/10/10 split on `sentiment_3` is materialized at ingestion time so
all downstream scripts share the exact same partitions.

## Evaluation gates

- Task B baseline target: **macro-F1 > 0.65** with TF-IDF + LogReg.
- DistilBERT target: macro-F1 > 0.78.
- PubMedBERT target: macro-F1 > 0.80 (the biomedical-pretrained advantage).
- Multi-task: matches single-task macro-F1 within 1pt while achieving ADR F1 > 0.55
  on the 200-review hand-labeled set.

## ADR annotation protocol

The 200-review hand-labeled set in `annotations/adr_manual_200.csv` is the most
defensible artifact in this repo. It was sampled stratified by `sentiment_3` and
annotated for character-level spans of:

- **ADR**: any patient-reported adverse effect (e.g. *"gave me terrible nausea"*).
- **DRUG**: drug names mentioned in the review body (often synonyms / brands).
- **SYMPTOM**: a symptom reference that is NOT clearly attributed to the drug
  (used as a negative class to test ADR vs. baseline-condition disambiguation).

Spans were entered as `start,end,label` triples. See `annotations/README.md`
(generated on first run of `src/adr_extraction.py --bootstrap-annotations`).

## Drug safety dashboard

`notebooks/05_drug_safety_dashboard.ipynb` and the Streamlit "Safety" tab compute,
per drug:

- review count
- mean & median rating
- fraction of reviews with ≥1 ADR mention
- top 10 ADR terms by frequency
- temporal trend in ADR mention rate

Drugs with N < 30 reviews are suppressed in dashboard views.

## Interpretability

`src/interpret.py` runs SHAP `Explainer` on the fine-tuned transformer over a
sample of negative-sentiment reviews. Token attributions are written to
`results/shap_negative_samples.html`. The Streamlit demo renders the same
attributions inline for any pasted review.

## Research extension

- **Pharmacovigilance from social media**: the same pipeline applied to Twitter /
  Reddit (`r/AskDocs`, drug-specific subreddits) for detection of adverse events
  ahead of FAERS reporting.
- **Few-shot LLM ADR extraction**: prompt engineering with Claude / GPT-4 over
  the same 200-review set, compared against the scispaCy-based pipeline.
- **FAERS comparison**: cross-reference top ADR terms per drug against FDA FAERS
  signal counts, looking for under-reported reactions.

## Caveats

- Self-reported reviews are **not** clinical data. They suffer from recall bias,
  selection bias (only motivated patients write reviews), and confound between
  the drug and the underlying condition.
- "ADR mention" ≠ confirmed adverse drug reaction; the model surfaces *mentions*,
  not adjudicated events.
- This repo is a research/portfolio artifact, not a regulated pharmacovigilance
  system.
