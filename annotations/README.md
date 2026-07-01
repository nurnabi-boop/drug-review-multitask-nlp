# ADR manual annotations

This folder holds the 200-review hand-labeled subset that is the most defensible
artifact of the project. The actual file `adr_manual_200.csv` is **not** committed
until you've labeled it — generate the empty seed by running:

```bash
python -m src.adr_extraction --bootstrap-annotations
```

That writes `adr_manual_200.csv` with one row per sampled review and an empty
`adr_spans` column. Open it in Excel / VS Code / your editor and fill the
`adr_spans` column with semicolon-separated `start,end,label` triples.

## Labels

- **ADR** — a patient-reported adverse effect attributable to the drug
  (*"gave me terrible nausea"*, *"caused severe headaches"*).
- **DRUG** — a drug name mentioned in the review body (often a brand or
  synonym, e.g. *"switched from Lexapro"*).
- **SYMPTOM** — a symptom NOT clearly attributed to the drug (the underlying
  condition itself, e.g. *"my anxiety was crippling before"*). Used as a
  negative class to test ADR vs. baseline-condition disambiguation.

## Format

Character offsets are 0-indexed, half-open, into the `review` column **as it
appears in the CSV row**. Multiple spans are separated by `;`.

```
42,49,ADR; 90,98,SYMPTOM; 130,138,DRUG
```

Empty cells are interpreted as "no ADR / DRUG / SYMPTOM mentioned."

## Sampling

The seed file is sampled stratified by `sentiment_3` (negative / neutral /
positive) so the eval set isn't dominated by 5-star "this drug is great" reviews
(which generally don't mention ADRs).

## Loading

```python
from src.adr_extraction import load_annotations
ann = load_annotations(Path("annotations/adr_manual_200.csv"))
ann["spans"]   # column of set[(start, end, label)]
```

`notebooks/04_adr_extraction.ipynb` consumes this file and computes strict
span-match P/R/F1 against the scispaCy bc5cdr predictions.
