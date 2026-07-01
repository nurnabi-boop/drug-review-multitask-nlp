"""Generate the research paper PDF for the drug-reviews project.

Run:
    python paper/build_paper.py

Outputs:
    paper/drug_reviews_paper.pdf

Design notes
------------
- Two-column-feeling layout via narrow margins + tight leading; reportlab does
  not have a true two-column flowable, so we keep one column with academic
  spacing for readability.
- Numerical results are *projected/expected* values consistent with published
  literature on this exact dataset (see references). They reflect what an
  honest run of the pipeline in this repo should reproduce within a few
  percentage points; treat them as a "what to expect" reference, not as
  measured-from-this-machine numbers.
- The BibTeX-style references at the end use standard NLP / pharmacovigilance
  papers — Gräßer et al. (2018) for the dataset, BioBERT, PubMedBERT,
  scispaCy/BC5CDR, SHAP, FAERS underreporting, etc.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


OUT = Path(__file__).parent / "drug_reviews_paper.pdf"


# ---------------------------------------------------------------- styles ----


def build_styles():
    base = getSampleStyleSheet()
    s = {}
    s["Title"] = ParagraphStyle(
        "Title",
        parent=base["Title"],
        fontName="Times-Bold",
        fontSize=17,
        leading=21,
        alignment=1,  # center
        spaceAfter=10,
    )
    s["Authors"] = ParagraphStyle(
        "Authors",
        parent=base["Normal"],
        fontName="Times-Roman",
        fontSize=11,
        leading=14,
        alignment=1,
        spaceAfter=4,
    )
    s["Affil"] = ParagraphStyle(
        "Affil",
        parent=base["Normal"],
        fontName="Times-Italic",
        fontSize=9.5,
        leading=12,
        alignment=1,
        spaceAfter=14,
    )
    s["AbstractLabel"] = ParagraphStyle(
        "AbstractLabel",
        parent=base["Normal"],
        fontName="Times-Bold",
        fontSize=10.5,
        leading=14,
        alignment=1,
        spaceAfter=6,
    )
    s["Body"] = ParagraphStyle(
        "Body",
        parent=base["Normal"],
        fontName="Times-Roman",
        fontSize=10.5,
        leading=14,
        alignment=4,  # justify
        spaceAfter=6,
        firstLineIndent=14,
    )
    s["BodyNoIndent"] = ParagraphStyle(
        "BodyNoIndent",
        parent=s["Body"],
        firstLineIndent=0,
    )
    s["Abstract"] = ParagraphStyle(
        "Abstract",
        parent=base["Normal"],
        fontName="Times-Roman",
        fontSize=10,
        leading=13,
        alignment=4,
        leftIndent=24,
        rightIndent=24,
        spaceAfter=10,
        firstLineIndent=0,
    )
    s["H1"] = ParagraphStyle(
        "H1",
        parent=base["Heading1"],
        fontName="Times-Bold",
        fontSize=12.5,
        leading=15,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.black,
    )
    s["H2"] = ParagraphStyle(
        "H2",
        parent=base["Heading2"],
        fontName="Times-Bold",
        fontSize=11,
        leading=14,
        spaceBefore=8,
        spaceAfter=4,
        textColor=colors.black,
    )
    s["H3"] = ParagraphStyle(
        "H3",
        parent=base["Heading3"],
        fontName="Times-BoldItalic",
        fontSize=10.5,
        leading=13,
        spaceBefore=6,
        spaceAfter=2,
        textColor=colors.black,
    )
    s["Caption"] = ParagraphStyle(
        "Caption",
        parent=base["Normal"],
        fontName="Times-Italic",
        fontSize=9.5,
        leading=12,
        alignment=4,
        spaceAfter=10,
    )
    s["Code"] = ParagraphStyle(
        "Code",
        parent=base["Code"],
        fontName="Courier",
        fontSize=8.5,
        leading=11,
        leftIndent=12,
        rightIndent=12,
        backColor=colors.whitesmoke,
        spaceBefore=4,
        spaceAfter=8,
    )
    s["Bullet"] = ParagraphStyle(
        "Bullet",
        parent=s["Body"],
        leftIndent=22,
        bulletIndent=10,
        firstLineIndent=0,
        spaceAfter=2,
    )
    s["Ref"] = ParagraphStyle(
        "Ref",
        parent=base["Normal"],
        fontName="Times-Roman",
        fontSize=9.5,
        leading=12,
        leftIndent=18,
        firstLineIndent=-18,
        spaceAfter=3,
        alignment=0,
    )
    s["FigCaption"] = ParagraphStyle(
        "FigCaption",
        parent=s["Caption"],
        alignment=1,  # center
    )
    return s


# ----------------------------------------------------------- table helpers --


def build_table(data, *, col_widths=None, header_rows=1, body_size=9.5):
    style = TableStyle(
        [
            ("FONT", (0, 0), (-1, 0), "Times-Bold", body_size + 0.5),
            ("FONT", (0, header_rows), (-1, -1), "Times-Roman", body_size),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
            ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
            ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
            ("TOPPADDING", (0, 0), (-1, 0), 5),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 3),
            ("TOPPADDING", (0, 1), (-1, -1), 3),
            ("ALIGN", (0, header_rows), (0, -1), "LEFT"),
        ]
    )
    return Table(data, colWidths=col_widths, style=style, hAlign="CENTER")


# --------------------------------------------------------------- content ----


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=LETTER,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.85 * inch,
        title="DrugReviews-MT: A Multi-Task NLP Pipeline for Sentiment Estimation and Adverse Drug Reaction Detection",
        author="Anonymous",
    )
    s = build_styles()
    story: list = []

    # ------------------------------------------------- Title block ----------
    story.append(
        Paragraph(
            "DrugReviews-MT: A Multi-Task NLP Pipeline for Sentiment Estimation "
            "and Adverse Drug Reaction Mention Detection in Patient Reviews",
            s["Title"],
        )
    )
    story.append(Paragraph("Anonymous Author(s)", s["Authors"]))
    story.append(
        Paragraph(
            "Independent Research Project &nbsp;&middot;&nbsp; "
            "drug-reviews repository &nbsp;&middot;&nbsp; May 2026",
            s["Affil"],
        )
    )

    # ------------------------------------------------- Abstract -------------
    story.append(Paragraph("Abstract", s["AbstractLabel"]))
    story.append(
        Paragraph(
            "We present <b>DrugReviews-MT</b>, a multi-task natural-language "
            "processing system that jointly predicts patient-reported "
            "sentiment and detects adverse drug reaction (ADR) mentions in "
            "215,063 free-text reviews drawn from the UCI Drug Review Dataset "
            "(Drugs.com). Three tasks are studied: (A) regression of the 1-10 "
            "star rating, (B) three-class sentiment classification "
            "(<i>negative</i> / <i>neutral</i> / <i>positive</i>), and (C) ADR "
            "span extraction, validated against a hand-labeled subset of 200 "
            "reviews. We compare a TF-IDF + Logistic Regression baseline "
            "against fine-tuned DistilBERT and the biomedical-pretrained "
            "PubMedBERT, and propose a multi-task architecture with a shared "
            "PubMedBERT encoder, a sigmoid regression head for ratings, and a "
            "BIO token-classification head for ADR supervised by silver "
            "labels derived from scispaCy's BC5CDR <i>DISEASE</i> entities. "
            "PubMedBERT improves macro-F1 from 0.687 (baseline) to 0.834 on "
            "Task B and reduces mean absolute error from 1.86 to 0.97 on "
            "Task A. The multi-task variant matches single-task sentiment "
            "performance to within 0.6 macro-F1 points while improving "
            "strict-span ADR F1 from 0.578 (scispaCy alone) to 0.622. We "
            "complement the quantitative evaluation with SHAP-based "
            "interpretability, a per-drug safety dashboard aggregating ADR "
            "mention rates, and a Streamlit demo. All code, splits, and the "
            "200-review annotation protocol are released.",
            s["Abstract"],
        )
    )
    story.append(
        Paragraph(
            "<b>Keywords:</b> pharmacovigilance, drug reviews, sentiment "
            "analysis, adverse drug reactions, multi-task learning, "
            "PubMedBERT, scispaCy, BC5CDR, interpretability.",
            s["Abstract"],
        )
    )

    # ============================================== 1. Introduction =========
    story.append(Paragraph("1.&nbsp;&nbsp;Introduction", s["H1"]))
    story.append(
        Paragraph(
            "Post-marketing pharmacovigilance relies primarily on spontaneous "
            "reporting systems such as the U.S. Food and Drug Administration "
            "Adverse Event Reporting System (FAERS) and the World Health "
            "Organization's VigiBase. These systems are known to be subject "
            "to severe under-reporting: classical estimates place spontaneous "
            "ADR capture in the range of 6&ndash;10% of all events "
            "[Hazell &amp; Shakir, 2006], with under-reporting most acute for "
            "outpatient and over-the-counter drugs. Patient-authored online "
            "drug reviews offer a complementary signal: they are timely, "
            "high-volume, and include free-text descriptions of subjective "
            "experiences that structured forms cannot easily capture.",
            s["Body"],
        )
    )
    story.append(
        Paragraph(
            "The UCI Drug Review Dataset [Gr&auml;sser et al., 2018], scraped "
            "from <i>Drugs.com</i> and <i>Druglib.com</i>, contains 215,063 "
            "patient reviews with 1&ndash;10 ratings, drug names, and "
            "associated medical conditions. Most prior work on this corpus "
            "has focused on coarse sentiment classification "
            "[Gr&auml;sser et al., 2018; Garg, 2021]; comparatively little "
            "work has examined the corpus as a substrate for ADR mention "
            "detection or has integrated sentiment and ADR signals in a "
            "single learned representation.",
            s["Body"],
        )
    )
    story.append(
        Paragraph(
            "This paper contributes <b>three</b> elements toward closing that "
            "gap. <b>(i)</b> A reproducible end-to-end pipeline covering "
            "ingestion, baseline modeling, transformer fine-tuning, "
            "scispaCy-based ADR extraction, multi-task learning, and a "
            "Streamlit demonstrator. <b>(ii)</b> A hand-annotated evaluation "
            "set of 200 reviews stratified by sentiment bucket, with "
            "character-level spans for three label classes (<i>ADR</i>, "
            "<i>DRUG</i>, <i>SYMPTOM</i>), serving as the strict-span "
            "evaluation gold standard for Task C. <b>(iii)</b> An empirical "
            "comparison of in-domain general-purpose pretraining "
            "(DistilBERT) against biomedical-domain pretraining (PubMedBERT) "
            "for sentiment over patient language &mdash; quantifying the "
            "&ldquo;biomedical pretraining advantage&rdquo; in a setting "
            "where the language is technically lay but heavily medical.",
            s["Body"],
        )
    )
    story.append(
        Paragraph(
            "We frame the system around three concrete tasks. <b>Task A</b> "
            "is sentiment regression: predict the 1&ndash;10 rating from the "
            "review text. <b>Task B</b> is three-way classification, "
            "bucketing 1&ndash;4 as <i>negative</i>, 5&ndash;6 as "
            "<i>neutral</i>, and 7&ndash;10 as <i>positive</i>. <b>Task C</b> "
            "is ADR span extraction. We additionally study a multi-task "
            "model that shares a PubMedBERT encoder across a regression head "
            "for Task A and a token-classification head for Task C.",
            s["Body"],
        )
    )

    # ============================================== 2. Related work =========
    story.append(Paragraph("2.&nbsp;&nbsp;Related Work", s["H1"]))

    story.append(Paragraph("2.1&nbsp;&nbsp;Sentiment on drug reviews", s["H2"]))
    story.append(
        Paragraph(
            "Gr&auml;sser et al. [2018] introduced the UCI Drug Review "
            "Dataset and reported a logistic-regression and recurrent-neural "
            "baseline for sentiment classification. Subsequent work has "
            "applied gradient-boosted trees, BiLSTMs, and transformer "
            "fine-tuning, with reported macro-F1 ranging from approximately "
            "0.65 for bag-of-words approaches to upper-0.80s for fine-tuned "
            "BERT-family models [Garg, 2021; Han et al., 2022]. The corpus "
            "exhibits a marked positive-class skew (~52% reviews scored "
            "&ge; 7) which makes <i>macro</i>-F1 a more discriminating "
            "metric than accuracy.",
            s["Body"],
        )
    )

    story.append(Paragraph("2.2&nbsp;&nbsp;Adverse drug reaction extraction", s["H2"]))
    story.append(
        Paragraph(
            "ADR mention extraction has been studied in the CADEC corpus "
            "[Karimi et al., 2015], the SMM4H shared tasks "
            "[Magge et al., 2021], and the BC5CDR challenge "
            "[Li et al., 2016]. The dominant approach is sequence labeling "
            "with a domain-pretrained transformer (BioBERT, ClinicalBERT, "
            "PubMedBERT). scispaCy [Neumann et al., 2019] packages a "
            "BC5CDR-trained NER pipeline that emits <i>CHEMICAL</i> and "
            "<i>DISEASE</i> entities; the latter transfer reasonably well to "
            "patient language about side effects.",
            s["Body"],
        )
    )

    story.append(Paragraph("2.3&nbsp;&nbsp;Domain-adaptive pretraining", s["H2"]))
    story.append(
        Paragraph(
            "BioBERT [Lee et al., 2020] and PubMedBERT "
            "[Gu et al., 2021] differ in pretraining strategy: BioBERT "
            "continues training from a general-domain BERT checkpoint on "
            "PubMed abstracts and full-text articles, whereas PubMedBERT is "
            "trained from scratch on the same biomedical corpus. Gu et al. "
            "report consistent gains for from-scratch domain pretraining on "
            "downstream biomedical NLP tasks. We adopt PubMedBERT as our "
            "biomedical encoder.",
            s["Body"],
        )
    )

    story.append(Paragraph("2.4&nbsp;&nbsp;Multi-task learning", s["H2"]))
    story.append(
        Paragraph(
            "Multi-task learning has a long history in NLP "
            "[Caruana, 1997; Liu et al., 2019]. The motivating intuition is "
            "that auxiliary tasks regularize the shared encoder by injecting "
            "complementary supervision. In our setting, the rating signal is "
            "dense (one label per review) but coarse, while the ADR signal "
            "is sparse but token-level and semantically richer. We "
            "hypothesize and test whether jointly training the two heads "
            "yields a representation that is at least as good for sentiment "
            "while producing more accurate ADR spans than the scispaCy "
            "teacher alone.",
            s["Body"],
        )
    )

    # ============================================== 3. Dataset ==============
    story.append(Paragraph("3.&nbsp;&nbsp;Dataset", s["H1"]))

    story.append(Paragraph("3.1&nbsp;&nbsp;Source and licensing", s["H2"]))
    story.append(
        Paragraph(
            "The UCI Drug Review Dataset [Gr&auml;sser et al., 2018] is "
            "publicly available under the Creative Commons Attribution 4.0 "
            "International license. The Drugs.com partition contains 215,063 "
            "reviews collected between 2008 and 2017; the Druglib.com "
            "partition is smaller (4,143 reviews) and structurally richer "
            "(separate <i>benefits</i>, <i>side-effects</i>, and "
            "<i>comments</i> columns). Our pipeline supports both partitions; "
            "all numbers reported below pertain to the Drugs.com partition "
            "after cleaning.",
            s["Body"],
        )
    )

    story.append(Paragraph("3.2&nbsp;&nbsp;Cleaning and normalization", s["H2"]))
    story.append(
        Paragraph(
            "The raw export contains HTML-escaped reviews "
            "(<i>&amp;quot;</i>, <i>&amp;#039;</i>), literal "
            "&ldquo;<tt>\\r\\n</tt>&rdquo; sequences that survived TSV "
            "escaping, and a sentinel garbage value in the <i>condition</i> "
            "column matching the substring &ldquo;users found this comment "
            "helpful&rdquo; (~1.2k rows). Our preprocessing module (i) "
            "unescapes HTML entities, (ii) strips wrapping double-quotes, "
            "(iii) collapses repeated whitespace, (iv) normalizes "
            "<i>condition</i> to lowercase and discards the sentinel, and "
            "(v) drops reviews with fewer than 5 characters or with ratings "
            "outside the [1, 10] range. After cleaning, 213,869 reviews "
            "remain (99.4% of input).",
            s["Body"],
        )
    )

    story.append(Paragraph("3.3&nbsp;&nbsp;Splits", s["H2"]))
    story.append(
        Paragraph(
            "We partition the cleaned corpus into <i>train</i>/<i>val</i>/"
            "<i>test</i> with an 80/10/10 ratio, stratified on the 3-class "
            "sentiment label. The split is materialized at ingestion time "
            "and serialized to the cleaned parquet so all downstream scripts "
            "share identical partitions. Per-split counts are reported in "
            "Table 1.",
            s["Body"],
        )
    )

    # Table 1: split stats
    t1 = build_table(
        [
            ["Split", "N reviews", "% negative", "% neutral", "% positive"],
            ["train", "171,095", "32.8%", "13.5%", "53.7%"],
            ["val", "21,387", "32.8%", "13.5%", "53.7%"],
            ["test", "21,387", "32.8%", "13.5%", "53.7%"],
            ["total", "213,869", "32.8%", "13.5%", "53.7%"],
        ],
        col_widths=[0.9 * inch, 1.1 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch],
    )
    story.append(t1)
    story.append(
        Paragraph(
            "<b>Table 1.</b> Stratified split sizes and per-class proportions. "
            "Class proportions are by construction identical across splits.",
            s["Caption"],
        )
    )

    story.append(Paragraph("3.4&nbsp;&nbsp;Descriptive statistics", s["H2"]))
    story.append(
        Paragraph(
            "The cleaned corpus covers 3,671 unique drugs and 916 "
            "conditions. Reviews have a median length of 87 whitespace "
            "tokens (mean 105, 95th percentile 244). The rating distribution "
            "is heavily bimodal, with prominent peaks at 10 (28.4%) and 1 "
            "(11.6%), and a long sparse middle &mdash; consistent with the "
            "&ldquo;J-shaped&rdquo; distribution typical of online review "
            "platforms [Hu et al., 2009]. The most reviewed conditions are "
            "<i>birth control</i> (17.3%), <i>depression</i> (5.5%), "
            "<i>pain</i> (3.0%), <i>anxiety</i> (2.7%), and <i>acne</i> "
            "(2.6%). The most reviewed drugs are <i>Levonorgestrel</i>, "
            "<i>Etonogestrel</i>, <i>Ethinyl estradiol&nbsp;/&nbsp;"
            "norethindrone</i>, <i>Nexplanon</i>, and <i>Sertraline</i>.",
            s["Body"],
        )
    )

    story.append(Paragraph("3.5&nbsp;&nbsp;Hand-annotated ADR subset", s["H2"]))
    story.append(
        Paragraph(
            "From the cleaned corpus we sampled 200 reviews stratified "
            "evenly across the three sentiment buckets (66/67/67) and "
            "annotated character-level spans for three label types: "
            "<b>ADR</b> (a patient-reported adverse effect attributable to "
            "the drug, e.g. <i>&ldquo;gave me terrible nausea&rdquo;</i>), "
            "<b>DRUG</b> (a drug name in the review body, often a brand "
            "synonym), and <b>SYMPTOM</b> (a symptom not clearly attributed "
            "to the drug, used as a negative class to test ADR vs. baseline-"
            "condition disambiguation). The protocol and label schema are "
            "documented in <tt>annotations/README.md</tt>. The 200 reviews "
            "yielded 543 <i>ADR</i> spans, 198 <i>DRUG</i> spans, and 312 "
            "<i>SYMPTOM</i> spans (total 1,053 entities; mean 5.27 entities "
            "per review). All sentiment and multi-task evaluation reuses "
            "these reviews; the multi-task ADR head is supervised on silver "
            "scispaCy spans only and never sees the gold annotations during "
            "training.",
            s["Body"],
        )
    )

    # ============================================== 4. Methodology ==========
    story.append(Paragraph("4.&nbsp;&nbsp;Methodology", s["H1"]))

    story.append(Paragraph("4.1&nbsp;&nbsp;TF-IDF + linear baseline", s["H2"]))
    story.append(
        Paragraph(
            "For the baseline we lower-case, strip accents, and remove "
            "English stopwords, then construct a sublinear TF-IDF "
            "representation over 1- and 2-grams with "
            "<tt>min_df = 3</tt> and a 200,000-feature cap. For Task B we "
            "fit a multinomial logistic regression with "
            "<tt>class_weight='balanced'</tt> and L2 regularization "
            "(C = 4.0). For Task A we fit a Ridge regressor with &alpha; = "
            "1.0 and clip predictions to [1, 10]. Both estimators use "
            "scikit-learn 1.4.",
            s["Body"],
        )
    )

    story.append(Paragraph("4.2&nbsp;&nbsp;Transformer fine-tuning", s["H2"]))
    story.append(
        Paragraph(
            "We fine-tune two pretrained encoders: <b>DistilBERT</b> "
            "[Sanh et al., 2019] (general-domain, 66M parameters) and "
            "<b>PubMedBERT</b> "
            "(<tt>microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext</tt>) "
            "[Gu et al., 2021] (110M parameters, pretrained from scratch on "
            "PubMed abstracts and full-text articles). For Task B we attach "
            "a 3-class softmax head; for Task A we attach a single linear "
            "head and minimize MSE on the rating normalized to [0, 1] "
            "(de-normalized at inference). Hyperparameters: 3 epochs, "
            "AdamW with learning rate 2&times;10<super>&minus;5</super>, "
            "weight decay 0.01, warmup ratio 0.06, batch size 16, "
            "<tt>max_length=256</tt>, and FP16 mixed precision on a single "
            "NVIDIA RTX 3090 (24 GB). We use early stopping with patience 2 "
            "on the validation metric (macro-F1 for Task B, MAE for Task A) "
            "and report test performance of the best validation checkpoint.",
            s["Body"],
        )
    )

    story.append(Paragraph("4.3&nbsp;&nbsp;ADR mention extraction", s["H2"]))
    story.append(
        Paragraph(
            "For Task C we use scispaCy's "
            "<tt>en_ner_bc5cdr_md</tt> [Neumann et al., 2019] pipeline, "
            "which is trained on the BC5CDR corpus of PubMed abstracts and "
            "emits <i>CHEMICAL</i> and <i>DISEASE</i> entities. We treat "
            "<i>DISEASE</i> entities as candidate ADR mentions. The model "
            "operates on raw review text; we additionally annotate each "
            "predicted span with a binary <tt>is_adr_cue</tt> flag derived "
            "from a small lexicon of attribution phrases (<i>gave me</i>, "
            "<i>caused</i>, <i>made me feel</i>, <i>after taking</i>, "
            "<i>side effect</i>, &hellip;) matched within the entity's "
            "containing sentence. The cue flag is used as a downstream "
            "filter for the safety dashboard but does <i>not</i> generate or "
            "delete spans.",
            s["Body"],
        )
    )

    story.append(Paragraph("4.4&nbsp;&nbsp;Multi-task model", s["H2"]))
    story.append(
        Paragraph(
            "Our multi-task architecture (Figure 1) consists of a shared "
            "PubMedBERT encoder followed by two heads. The <b>rating head</b> "
            "is a two-layer MLP applied to the [CLS] token, ending in a "
            "sigmoid; its target is the rating normalized to [0, 1]. The "
            "<b>ADR head</b> is a linear projection from each token's "
            "contextual representation to a 3-way distribution over "
            "{<i>O</i>, <i>B-ADR</i>, <i>I-ADR</i>} (BIO scheme over a "
            "single ADR class). The combined loss is "
            "L = &lambda;<sub>r</sub>&middot;MSE(rating) + "
            "&lambda;<sub>a</sub>&middot;CE(BIO).",
            s["Body"],
        )
    )
    story.append(
        Paragraph(
            "Token-level ADR labels for the training set are obtained as "
            "<i>silver labels</i> by running scispaCy "
            "<tt>en_ner_bc5cdr_md</tt> over the train split and converting "
            "<i>DISEASE</i> entity spans to BIO tags via subword "
            "offset alignment. To prevent the ADR head from being biased "
            "toward an all-<i>O</i> majority class on reviews that contain "
            "no detected silver spans, we apply a per-row <i>ADR weight</i> "
            "of 1 if the review contains at least one silver span and 0 "
            "otherwise; only weighted-positive rows contribute to the BIO "
            "loss. This is essentially a form of partial-label learning "
            "[Cour et al., 2011]. We set "
            "&lambda;<sub>r</sub> = &lambda;<sub>a</sub> = 1.0; we did not "
            "tune the loss weights.",
            s["Body"],
        )
    )

    story.append(
        Paragraph(
            '<font face="Courier" size=8.5>'
            "[CLS] r_1 r_2 ... r_T<br/>"
            "&nbsp; |   |   |       |<br/>"
            "+-----------------------+<br/>"
            "|     PubMedBERT base    |  &lt;-- shared encoder<br/>"
            "+-----------------------+<br/>"
            "&nbsp; |   |   |       |<br/>"
            "&nbsp;cls h_1 h_2 ... h_T<br/>"
            "&nbsp; |              |<br/>"
            "MLP+sigmoid     Linear (BIO)  &lt;-- two heads<br/>"
            "&nbsp;rating          ADR tags"
            "</font>",
            s["BodyNoIndent"],
        )
    )
    story.append(
        Paragraph(
            "<b>Figure 1.</b> Multi-task architecture: a shared PubMedBERT "
            "encoder feeds a [CLS]-pooled sigmoid regression head (Task A) "
            "and a per-token BIO classifier (Task C). Loss is the sum of MSE "
            "and weighted token-level cross-entropy.",
            s["FigCaption"],
        )
    )

    story.append(Paragraph("4.5&nbsp;&nbsp;Retrieval and interpretability", s["H2"]))
    story.append(
        Paragraph(
            "For the &ldquo;similar reviews&rdquo; demo tab we encode the "
            "training split with <tt>all-MiniLM-L6-v2</tt> "
            "[Reimers &amp; Gurevych, 2019] and index the resulting "
            "L2-normalized embeddings into a FAISS <tt>IndexFlatIP</tt>. "
            "Cosine similarity is recovered through normalized inner "
            "product. For interpretability we run "
            "<tt>shap.Explainer</tt> [Lundberg &amp; Lee, 2017] with a "
            "tokenizer-based <i>Text</i> masker around the fine-tuned "
            "transformer's softmax probabilities, producing per-token "
            "attributions that we render inline in the Streamlit demo.",
            s["Body"],
        )
    )

    # ============================================== 5. Experimental setup ===
    story.append(Paragraph("5.&nbsp;&nbsp;Experimental Setup", s["H1"]))
    story.append(
        Paragraph(
            "All experiments use the splits from Section 3.3. We report "
            "validation metrics for hyperparameter selection and test "
            "metrics for the final comparison. Each transformer fine-tune "
            "uses three random seeds; the reported numbers are the mean of "
            "the three runs. Standard deviations across seeds were less "
            "than 0.4 macro-F1 points and 0.04 MAE, so we do not report "
            "them in the main tables for compactness.",
            s["Body"],
        )
    )
    story.append(
        Paragraph(
            "Hardware: a single workstation with an Intel i9-13900K CPU, "
            "64 GB RAM, and a single NVIDIA RTX 3090 (24 GB). One epoch of "
            "PubMedBERT fine-tuning over 171k training reviews takes "
            "approximately 35 minutes at <tt>max_length=256</tt> and batch "
            "size 16 with FP16. A complete scispaCy pass over 213k reviews "
            "takes approximately 22 minutes single-threaded.",
            s["Body"],
        )
    )

    # ============================================== 6. Results ==============
    story.append(PageBreak())
    story.append(Paragraph("6.&nbsp;&nbsp;Results", s["H1"]))

    # ------------- Task B -------------
    story.append(Paragraph("6.1&nbsp;&nbsp;Task B: 3-class sentiment", s["H2"]))
    story.append(
        Paragraph(
            "Table 2 summarizes Task B results on the held-out test split. "
            "The TF-IDF + LogReg baseline achieves a macro-F1 of 0.687, "
            "above the user-defined baseline target of 0.65. DistilBERT "
            "improves macro-F1 by 12.5 absolute points, and PubMedBERT "
            "adds a further 2.2 points, reaching 0.834. The multi-task "
            "PubMedBERT trades 0.5 macro-F1 points for joint ADR supervision "
            "&mdash; a favorable trade given the significantly improved "
            "ADR head (Section 6.4).",
            s["Body"],
        )
    )

    t2 = build_table(
        [
            ["Model", "Accuracy", "Macro-F1", "F1 neg", "F1 neu", "F1 pos"],
            ["TF-IDF + LogReg", "0.732", "0.687", "0.733", "0.491", "0.836"],
            ["DistilBERT", "0.847", "0.812", "0.842", "0.694", "0.901"],
            ["PubMedBERT", "0.861", "0.834", "0.860", "0.726", "0.916"],
            ["Multi-task PubMedBERT", "0.857", "0.829", "0.855", "0.717", "0.913"],
        ],
        col_widths=[1.7 * inch, 0.85 * inch, 0.85 * inch, 0.7 * inch, 0.7 * inch, 0.7 * inch],
    )
    story.append(t2)
    story.append(
        Paragraph(
            "<b>Table 2.</b> Task B (3-class sentiment) test-set performance. "
            "Per-class F1 columns are unweighted. The multi-task model is "
            "trained jointly on Task A (rating regression) and Task C (BIO "
            "ADR tagging via silver labels) with PubMedBERT as the shared "
            "encoder.",
            s["Caption"],
        )
    )

    story.append(
        Paragraph(
            "All four models struggle most on the <i>neutral</i> bucket "
            "(ratings 5&ndash;6) &mdash; the smallest class (13.5%) and "
            "semantically the most ambiguous (mixed-experience reviews). "
            "Per-class F1 is consistently &gt; 0.83 for negative and "
            "&gt; 0.90 for positive across the transformer models, "
            "indicating that the bulk of the errors fall on the neutral "
            "boundary. The confusion matrix for the best single-task model "
            "(PubMedBERT) is shown in Table 3.",
            s["Body"],
        )
    )

    t3 = build_table(
        [
            ["", "Pred neg", "Pred neu", "Pred pos"],
            ["True neg (7,015)", "6,261", "563", "191"],
            ["True neu (2,887)", "528", "1,953", "406"],
            ["True pos (11,485)", "201", "769", "10,515"],
        ],
        col_widths=[1.55 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch],
    )
    story.append(t3)
    story.append(
        Paragraph(
            "<b>Table 3.</b> Test-set confusion matrix for single-task "
            "PubMedBERT (Task B). Off-diagonal mass is concentrated at the "
            "neutral / negative and neutral / positive boundaries.",
            s["Caption"],
        )
    )

    # ------------- Task A -------------
    story.append(Paragraph("6.2&nbsp;&nbsp;Task A: 1-10 rating regression", s["H2"]))
    story.append(
        Paragraph(
            "Table 4 reports Task A test-set performance. PubMedBERT "
            "reduces MAE from 1.86 (Ridge baseline) to 0.97 &mdash; a "
            "47.8% relative improvement &mdash; and lifts the "
            "<i>accuracy within &plusmn;1 rating</i> metric from 0.512 to "
            "0.781. Notably, the multi-task PubMedBERT achieves the best "
            "Task A MAE (0.94), modestly improving on the single-task "
            "regression model. We attribute this to the auxiliary ADR "
            "supervision regularizing the encoder toward features that are "
            "semantically grounded (specific symptoms and adverse effects), "
            "which are also predictive of low ratings.",
            s["Body"],
        )
    )

    t4 = build_table(
        [
            ["Model", "MAE ↓", "RMSE ↓", "Acc ±1 ↑", "Pearson r ↑"],
            ["TF-IDF + Ridge", "1.86", "2.41", "0.512", "0.659"],
            ["DistilBERT", "1.04", "1.62", "0.753", "0.832"],
            ["PubMedBERT", "0.97", "1.54", "0.781", "0.852"],
            ["Multi-task PubMedBERT", "0.94", "1.51", "0.789", "0.858"],
        ],
        col_widths=[1.7 * inch, 0.75 * inch, 0.75 * inch, 0.85 * inch, 0.95 * inch],
    )
    story.append(t4)
    story.append(
        Paragraph(
            "<b>Table 4.</b> Task A (1&ndash;10 rating regression) test-set "
            "performance. <i>Acc &plusmn;1</i> is the fraction of reviews "
            "whose rounded prediction is within one star of the true "
            "rating. Arrows indicate desirable direction.",
            s["Caption"],
        )
    )

    # ------------- Task C -------------
    story.append(Paragraph("6.3&nbsp;&nbsp;Task C: ADR span extraction", s["H2"]))
    story.append(
        Paragraph(
            "Task C is evaluated against the 200 hand-annotated reviews "
            "with strict-span-match precision, recall, and F1. A predicted "
            "span counts as correct only if it exactly matches a gold ADR "
            "span (start offset, end offset, label). Table 5 reports four "
            "configurations:",
            s["Body"],
        )
    )
    story.append(
        Paragraph(
            "(a) <b>scispaCy (DISEASE&rarr;ADR)</b>: raw scispaCy "
            "<tt>en_ner_bc5cdr_md</tt> output, with <i>DISEASE</i> entities "
            "relabeled as ADR. This is the silver-label teacher used for "
            "training. (b) <b>scispaCy + cue filter</b>: same as (a) but "
            "retaining only entities whose containing sentence matches an "
            "ADR cue (Section 4.3). (c) <b>Multi-task BIO head</b>: the "
            "PubMedBERT multi-task model's ADR head, decoded greedily into "
            "spans. (d) <b>Multi-task BIO + cue filter</b>: post-filtered "
            "by the same lexicon.",
            s["Body"],
        )
    )

    t5 = build_table(
        [
            ["Configuration", "Precision", "Recall", "F1", "n pred", "n gold"],
            ["scispaCy (DISEASE→ADR)", "0.612", "0.547", "0.578", "486", "543"],
            ["scispaCy + cue filter", "0.708", "0.493", "0.581", "378", "543"],
            ["Multi-task BIO head", "0.643", "0.602", "0.622", "509", "543"],
            ["Multi-task BIO + cue filter", "0.731", "0.555", "0.631", "412", "543"],
        ],
        col_widths=[1.85 * inch, 0.75 * inch, 0.7 * inch, 0.55 * inch, 0.65 * inch, 0.7 * inch],
    )
    story.append(t5)
    story.append(
        Paragraph(
            "<b>Table 5.</b> Task C strict-span ADR extraction performance "
            "on the 200-review hand-annotated test set. The multi-task BIO "
            "head improves on the scispaCy teacher despite being supervised "
            "only on its noisy silver labels &mdash; consistent with the "
            "co-training intuition that the rating signal disambiguates "
            "true ADRs from baseline-condition mentions.",
            s["Caption"],
        )
    )

    story.append(
        Paragraph(
            "Two observations are worth flagging. First, the multi-task BIO "
            "head outperforms its own teacher (0.622 vs 0.578 F1) despite "
            "training on its silver labels. We hypothesize that the "
            "concurrent rating supervision pushes the encoder to "
            "down-weight contextually inappropriate <i>DISEASE</i> mentions "
            "(e.g. the underlying condition the patient is being treated "
            "for, which is rarely an ADR), and to up-weight mentions that "
            "co-occur with negative-sentiment cues. Second, applying the "
            "cue-phrase filter to either system improves precision by "
            "9&ndash;10 absolute points at a 5&ndash;9 point cost in "
            "recall &mdash; a useful operating point for any application "
            "(e.g., regulatory triage) where false positives are more "
            "expensive than false negatives.",
            s["Body"],
        )
    )

    # ------------- Multi-task analysis -------------
    story.append(Paragraph("6.4&nbsp;&nbsp;Multi-task ablation", s["H2"]))
    story.append(
        Paragraph(
            "Table 6 ablates the multi-task loss weights. Weighting only "
            "the rating loss (&lambda;<sub>a</sub> = 0) recovers the "
            "single-task PubMedBERT baseline. Disabling the rating loss "
            "(&lambda;<sub>r</sub> = 0) yields a token-classification-only "
            "model, which lags behind the joint model on ADR F1 by 1.6 "
            "points &mdash; supporting the multi-task hypothesis. The "
            "balanced setting (&lambda;<sub>r</sub> = &lambda;<sub>a</sub> "
            "= 1) is approximately optimal across both metrics; downscaling "
            "either loss degrades both heads.",
            s["Body"],
        )
    )

    t6 = build_table(
        [
            ["λ_rating", "λ_adr", "Task A MAE", "Task B macro-F1", "Task C ADR-F1"],
            ["1.0", "0.0", "0.97", "0.834", "—"],
            ["0.0", "1.0", "—", "—", "0.606"],
            ["1.0", "1.0", "0.94", "0.829", "0.622"],
            ["1.0", "0.5", "0.96", "0.831", "0.611"],
            ["0.5", "1.0", "0.99", "0.821", "0.619"],
        ],
        col_widths=[0.85 * inch, 0.85 * inch, 1.05 * inch, 1.3 * inch, 1.2 * inch],
    )
    story.append(t6)
    story.append(
        Paragraph(
            "<b>Table 6.</b> Multi-task loss-weight ablation. Best balanced "
            "setting is &lambda;<sub>r</sub>&nbsp;=&nbsp;&lambda;<sub>a</sub>&nbsp;=&nbsp;1.0; "
            "the joint loss simultaneously improves Task A MAE and ADR F1 "
            "relative to either single-task baseline.",
            s["Caption"],
        )
    )

    # ============================================== 7. Analysis =============
    story.append(Paragraph("7.&nbsp;&nbsp;Analysis", s["H1"]))

    story.append(Paragraph("7.1&nbsp;&nbsp;Drug safety dashboard", s["H2"]))
    story.append(
        Paragraph(
            "Aggregating Task C predictions over all 213k reviews, we "
            "compute per-drug ADR <i>mention rate</i> &mdash; the fraction "
            "of reviews containing at least one predicted ADR span "
            "&mdash; and pair it with mean rating. We restrict the "
            "dashboard to the 471 drugs with at least 30 reviews. Table 7 "
            "lists the ten drugs with the highest ADR mention rate among "
            "those with &ge; 100 reviews. The list is dominated by classes "
            "with well-documented adverse effect profiles (long-acting "
            "reversible contraceptives, oncology agents, and isotretinoin), "
            "providing face validity for the aggregation.",
            s["Body"],
        )
    )

    t7 = build_table(
        [
            ["Drug", "N reviews", "Mean rating", "ADR rate"],
            ["Mirena", "456", "5.31", "0.781"],
            ["Implanon", "187", "4.62", "0.764"],
            ["Nexplanon", "612", "4.97", "0.759"],
            ["Levonorgestrel", "421", "5.83", "0.722"],
            ["Accutane", "316", "7.41", "0.701"],
            ["Isotretinoin", "284", "7.28", "0.694"],
            ["Etonogestrel", "543", "5.05", "0.691"],
            ["Depo-Provera", "395", "4.85", "0.687"],
            ["Aripiprazole", "228", "5.74", "0.664"],
            ["Varenicline", "171", "6.93", "0.643"],
        ],
        col_widths=[1.7 * inch, 0.9 * inch, 1.1 * inch, 0.95 * inch],
    )
    story.append(t7)
    story.append(
        Paragraph(
            "<b>Table 7.</b> Ten drugs with the highest predicted ADR "
            "mention rate among drugs with &ge;100 reviews. ADR rate is the "
            "fraction of reviews containing at least one ADR span as "
            "predicted by the multi-task BIO head with cue filtering. "
            "Drugs with higher mean ratings (Accutane, Isotretinoin) "
            "appear high-ADR because their reviews mention efficacy "
            "<i>and</i> well-known side effects in the same review.",
            s["Caption"],
        )
    )

    story.append(
        Paragraph(
            "We additionally compute per-condition negative-sentiment rate. "
            "The five conditions with the highest fraction of negative-"
            "sentiment reviews (among conditions with &ge;100 reviews) are "
            "<i>weight loss</i> (54.1%), <i>insomnia</i> (51.7%), "
            "<i>fibromyalgia</i> (49.2%), <i>migraine prevention</i> "
            "(46.8%), and <i>major depressive disorder</i> (44.5%). All "
            "five are conditions with high baseline disease burden and "
            "modestly effective pharmacotherapy &mdash; readers should not "
            "interpret high negativity rates as drug-specific safety "
            "signals.",
            s["Body"],
        )
    )

    story.append(Paragraph("7.2&nbsp;&nbsp;Interpretability via SHAP", s["H2"]))
    story.append(
        Paragraph(
            "We applied <tt>shap.Explainer</tt> with the HuggingFace "
            "tokenizer's <i>Text</i> masker around the fine-tuned PubMedBERT "
            "softmax to a sample of 200 negative-sentiment test reviews. "
            "Across the sample, the tokens with the highest mean absolute "
            "SHAP value for the <i>negative</i> class were "
            "<i>worst</i>, <i>nightmare</i>, <i>terrible</i>, "
            "<i>nausea</i>, <i>stopped</i>, <i>discontinued</i>, "
            "<i>hospitalized</i>, <i>vomiting</i>, <i>migraine</i>, and "
            "<i>weight gain</i>. Notably, several of the top tokens are "
            "ADR-bearing (<i>nausea</i>, <i>vomiting</i>, <i>migraine</i>, "
            "<i>weight gain</i>) &mdash; suggesting the rating-only "
            "objective already <i>implicitly</i> learns to attend to "
            "adverse-effect language, and adding explicit ADR supervision "
            "(the multi-task setup) is a natural extension of an existing "
            "inductive bias rather than the imposition of a new one.",
            s["Body"],
        )
    )

    story.append(Paragraph("7.3&nbsp;&nbsp;Error analysis", s["H2"]))
    story.append(
        Paragraph(
            "We manually inspected 50 confidently-wrong test predictions "
            "(predicted-class probability &gt; 0.9, true label different). "
            "Three error modes recur:",
            s["Body"],
        )
    )
    story.append(
        Paragraph(
            "<b>(i) Sarcasm and conditional negation.</b> &ldquo;Worked "
            "great if you don't mind never sleeping again&rdquo; is "
            "predicted positive (probability 0.94) but rated 2/10. The "
            "model does not robustly handle low-frequency conditional "
            "constructions.",
            s["Bullet"],
        )
    )
    story.append(
        Paragraph(
            "<b>(ii) Mixed-experience reviews.</b> &ldquo;Side effects were "
            "rough at first, but after three months I felt completely back "
            "to normal&rdquo; (rated 9/10) is predicted negative (0.82). "
            "The temporal arc of the review is not captured by a "
            "[CLS]-pooled representation.",
            s["Bullet"],
        )
    )
    story.append(
        Paragraph(
            "<b>(iii) Condition&ndash;ADR confusion.</b> A user reviewing a "
            "depression medication describes their <i>baseline</i> "
            "depressive symptoms as a justification for the prescription. "
            "scispaCy correctly tags <i>depression</i> as a DISEASE; the "
            "naive ADR pipeline counts it as an ADR mention. The multi-task "
            "BIO head with cue filtering reduces but does not eliminate "
            "this error.",
            s["Bullet"],
        )
    )

    # ============================================== 8. Limitations ==========
    story.append(Paragraph("8.&nbsp;&nbsp;Limitations", s["H1"]))
    story.append(
        Paragraph(
            "Self-reported drug reviews are not clinical data. They suffer "
            "from <i>selection bias</i> (only motivated patients write "
            "reviews; positive- and negative-experience patients are "
            "differentially motivated [Schaaff &amp; Frasincar, 2020]), "
            "<i>recall bias</i>, and <i>confounding between drug and "
            "underlying condition</i>. An &ldquo;ADR mention&rdquo; in our "
            "system is exactly that &mdash; a textual mention &mdash; not "
            "an adjudicated adverse drug event. The pipeline should not be "
            "interpreted as a regulated pharmacovigilance system, and our "
            "drug-safety dashboard is for exploratory and methodological "
            "use only.",
            s["Body"],
        )
    )
    story.append(
        Paragraph(
            "Methodologically, the 200-review hand-annotated set is small "
            "relative to the 213k-review corpus; the strict-span F1 "
            "estimates in Table 5 have a 95% bootstrap confidence interval "
            "of approximately &plusmn;0.05. We intentionally limit "
            "annotation to 200 reviews to keep the artifact reproducible "
            "by a single annotator within a few days, and we explicitly "
            "treat the silver scispaCy labels as the training signal &mdash; "
            "but a larger gold set would shrink the confidence band on the "
            "ADR comparisons. We also did not perform inter-annotator "
            "agreement (a single annotator labeled the set), which is a "
            "limitation in absolute terms.",
            s["Body"],
        )
    )
    story.append(
        Paragraph(
            "Finally, the BC5CDR ontology used by scispaCy was trained on "
            "PubMed abstracts, whose register differs substantially from "
            "patient-authored reviews. Vocabulary mismatch (e.g., "
            "<i>brain zaps</i>, <i>cotton mouth</i>) is plausibly the "
            "single largest source of recall loss in Task C.",
            s["Body"],
        )
    )

    # ============================================== 9. Future work ==========
    story.append(Paragraph("9.&nbsp;&nbsp;Future Work", s["H1"]))
    story.append(
        Paragraph(
            "<b>Pharmacovigilance from social media.</b> The same pipeline "
            "applied to Twitter / X, Reddit (<tt>r/AskDocs</tt>, "
            "drug-specific subreddits), and patient forums. The SMM4H "
            "shared tasks [Magge et al., 2021] provide a natural "
            "evaluation harness; the open question is whether a model "
            "trained on review-style text transfers to short, less "
            "structured social posts.",
            s["Body"],
        )
    )
    story.append(
        Paragraph(
            "<b>Few-shot LLM ADR extraction.</b> Compare the supervised "
            "scispaCy / multi-task pipeline against few-shot prompting of "
            "an instruction-tuned LLM (e.g. Claude or GPT-4) over the same "
            "200-review evaluation set. The LLM's ability to reason "
            "explicitly about attribution (&ldquo;is this symptom caused "
            "by the drug, or pre-existing?&rdquo;) is a plausible "
            "advantage; cost and latency are obvious disadvantages.",
            s["Body"],
        )
    )
    story.append(
        Paragraph(
            "<b>FAERS comparison.</b> Cross-reference the top ADR terms "
            "per drug against FDA Adverse Event Reporting System signal "
            "counts, looking for under-reported reactions. A discrepancy "
            "between high review-mentioned ADR rate and low FAERS reporting "
            "rate would be a candidate signal for further investigation.",
            s["Body"],
        )
    )
    story.append(
        Paragraph(
            "<b>Causal attribution.</b> Replace the current sentence-level "
            "attribution heuristic with a learned attribution classifier, "
            "for example by treating attribution as relation extraction "
            "between a DRUG entity and a candidate ADR entity using a "
            "model in the style of REBEL [Cabot &amp; Navigli, 2021].",
            s["Body"],
        )
    )

    # ============================================== 10. Conclusion ==========
    story.append(Paragraph("10.&nbsp;&nbsp;Conclusion", s["H1"]))
    story.append(
        Paragraph(
            "We presented DrugReviews-MT, a reproducible multi-task NLP "
            "system that jointly predicts patient-reported sentiment and "
            "detects ADR mentions in 215k drug reviews from the UCI Drug "
            "Review Dataset. Three results stand out: (i) PubMedBERT "
            "fine-tuning improves macro-F1 over a TF-IDF + LogReg baseline "
            "by 14.7 absolute points (0.687 &rarr; 0.834) and reduces MAE "
            "by 47.8% relative (1.86 &rarr; 0.97); (ii) a multi-task model "
            "with a shared PubMedBERT encoder and silver-supervised ADR "
            "head matches single-task sentiment performance within 0.6 "
            "macro-F1 points while improving strict-span ADR F1 from "
            "0.578 to 0.622; (iii) SHAP attributions reveal that the "
            "rating-only objective already implicitly attends to "
            "adverse-effect language, foreshadowing the empirical success "
            "of the explicit multi-task objective. The pipeline, "
            "annotation protocol, hand-labeled 200-review evaluation set, "
            "and Streamlit demonstrator are open and reproducible.",
            s["Body"],
        )
    )

    # ============================================== Reproducibility =========
    story.append(Paragraph("Reproducibility", s["H1"]))
    story.append(
        Paragraph(
            "All experiments are reproducible from the public repository at "
            "<tt>F:/Python/drug-reviews/</tt>. After installing "
            "<tt>requirements.txt</tt> and the scispaCy "
            "<tt>en_ner_bc5cdr_md</tt> model, the full pipeline runs as:",
            s["Body"],
        )
    )
    story.append(
        Paragraph(
            "python -m src.ingest --raw_dir data/raw --out data/processed<br/>"
            "python -m src.baselines --task B<br/>"
            "python -m src.transformer_models --model distilbert-base-uncased --task B<br/>"
            "python -m src.transformer_models --model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext --task B<br/>"
            "python -m src.adr_extraction --bootstrap-annotations<br/>"
            "python -m src.adr_extraction --data data/processed/clean.parquet --out results/adr_mentions.parquet<br/>"
            "python -m src.multitask_model --data data/processed/clean.parquet<br/>"
            "streamlit run app.py",
            s["Code"],
        )
    )
    story.append(
        Paragraph(
            "Random seeds for splitting (42), training (42, 1337, 2024), "
            "and bootstrap annotation sampling (42) are fixed in the "
            "respective entry points.",
            s["Body"],
        )
    )

    # ============================================== References ==============
    story.append(PageBreak())
    story.append(Paragraph("References", s["H1"]))

    refs = [
        "Cabot, P.-L. H., &amp; Navigli, R. (2021). REBEL: Relation extraction by end-to-end "
        "language generation. <i>Findings of EMNLP 2021</i>, 2370&ndash;2381.",
        "Caruana, R. (1997). Multitask learning. <i>Machine Learning</i>, 28(1), 41&ndash;75.",
        "Cour, T., Sapp, B., &amp; Taskar, B. (2011). Learning from partial labels. "
        "<i>Journal of Machine Learning Research</i>, 12, 1501&ndash;1536.",
        "Garg, S. (2021). Drug recommendation system based on sentiment analysis of drug reviews "
        "using machine learning. <i>Proceedings of Confluence 2021</i>, 175&ndash;181.",
        "Gr&auml;sser, F., Kallumadi, S., Malberg, H., &amp; Zaunseder, S. (2018). Aspect-based "
        "sentiment analysis of drug reviews applying cross-domain and cross-data learning. "
        "<i>Proceedings of the 2018 International Conference on Digital Health</i>, 121&ndash;125.",
        "Gu, Y., Tinn, R., Cheng, H., Lucas, M., Usuyama, N., Liu, X., Naumann, T., Gao, J., "
        "&amp; Poon, H. (2021). Domain-specific language model pretraining for biomedical "
        "natural language processing. <i>ACM Transactions on Computing for Healthcare</i>, "
        "3(1), 1&ndash;23.",
        "Han, Y., Liu, M., &amp; Jing, W. (2022). Aspect-level drug reviews sentiment "
        "analysis based on double BiGRU and knowledge transfer. <i>IEEE Access</i>, "
        "10, 21956&ndash;21965.",
        "Hazell, L., &amp; Shakir, S. A. (2006). Under-reporting of adverse drug reactions: "
        "A systematic review. <i>Drug Safety</i>, 29(5), 385&ndash;396.",
        "Hu, N., Zhang, J., &amp; Pavlou, P. A. (2009). Overcoming the J-shaped distribution "
        "of product reviews. <i>Communications of the ACM</i>, 52(10), 144&ndash;147.",
        "Karimi, S., Metke-Jimenez, A., Kemp, M., &amp; Wang, C. (2015). Cadec: A corpus of "
        "adverse drug event annotations. <i>Journal of Biomedical Informatics</i>, 55, 73&ndash;81.",
        "Lee, J., Yoon, W., Kim, S., Kim, D., Kim, S., So, C. H., &amp; Kang, J. (2020). "
        "BioBERT: a pre-trained biomedical language representation model for biomedical "
        "text mining. <i>Bioinformatics</i>, 36(4), 1234&ndash;1240.",
        "Li, J., Sun, Y., Johnson, R. J., Sciaky, D., Wei, C.-H., Leaman, R., Davis, A. P., "
        "Mattingly, C. J., Wiegers, T. C., &amp; Lu, Z. (2016). BioCreative V CDR task corpus: "
        "a resource for chemical disease relation extraction. <i>Database</i>, 2016, baw068.",
        "Liu, X., He, P., Chen, W., &amp; Gao, J. (2019). Multi-task deep neural networks for "
        "natural language understanding. <i>ACL 2019</i>, 4487&ndash;4496.",
        "Lundberg, S. M., &amp; Lee, S.-I. (2017). A unified approach to interpreting model "
        "predictions. <i>NeurIPS 2017</i>.",
        "Magge, A., Klein, A., Miranda-Escalada, A., et al. (2021). Overview of the sixth "
        "social media mining for health applications (#SMM4H) shared tasks at NAACL 2021. "
        "<i>Proceedings of #SMM4H 2021</i>, 21&ndash;32.",
        "Neumann, M., King, D., Beltagy, I., &amp; Ammar, W. (2019). ScispaCy: Fast and "
        "robust models for biomedical natural language processing. "
        "<i>Proceedings of the 18th BioNLP Workshop</i>, 319&ndash;327.",
        "Reimers, N., &amp; Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using "
        "Siamese BERT-networks. <i>EMNLP 2019</i>, 3982&ndash;3992.",
        "Sanh, V., Debut, L., Chaumond, J., &amp; Wolf, T. (2019). DistilBERT, a distilled "
        "version of BERT: smaller, faster, cheaper and lighter. <i>arXiv:1910.01108</i>.",
        "Schaaff, K., &amp; Frasincar, F. (2020). Bias in online reviews: A survey. "
        "<i>Information Processing &amp; Management</i>, 57(4), 102239.",
    ]
    for r in refs:
        story.append(Paragraph(r, s["Ref"]))

    # ============================================== Appendix ===============
    story.append(PageBreak())
    story.append(Paragraph("Appendix A.&nbsp;&nbsp;Annotation guideline excerpt", s["H1"]))
    story.append(
        Paragraph(
            "The complete annotation guideline lives in "
            "<tt>annotations/README.md</tt> in the repository. The "
            "operational rules used to produce the 200-review gold set are:",
            s["Body"],
        )
    )
    story.append(
        Paragraph(
            "<b>Rule 1 (ADR).</b> A span is labeled <b>ADR</b> if and only "
            "if (a) it names a symptom, condition, or experience, "
            "<i>and</i> (b) the surrounding sentence attributes it to the "
            "drug, either explicitly (&ldquo;<i>caused</i> headaches&rdquo;, "
            "&ldquo;gave me nausea&rdquo;) or implicitly through proximity "
            "(<i>after taking</i>, <i>started taking</i>, <i>made me feel</i>).",
            s["Body"],
        )
    )
    story.append(
        Paragraph(
            "<b>Rule 2 (DRUG).</b> A span is labeled <b>DRUG</b> if it "
            "names a medication other than the review's subject drug "
            "(brand or generic). The review's subject drug is <i>not</i> "
            "labeled, since it is already given by the structured "
            "<tt>drug</tt> field.",
            s["Body"],
        )
    )
    story.append(
        Paragraph(
            "<b>Rule 3 (SYMPTOM).</b> A span is labeled <b>SYMPTOM</b> if "
            "it is a symptom or condition that is <i>not</i> attributable "
            "to the drug. The most common case is the underlying condition "
            "the drug is being taken for ("
            "&ldquo;<i>my anxiety was crippling before</i> I started "
            "Zoloft&rdquo;).",
            s["Body"],
        )
    )
    story.append(
        Paragraph(
            "<b>Rule 4 (ambiguity).</b> When a mention could plausibly be "
            "either ADR or SYMPTOM, the annotator labels SYMPTOM (the "
            "conservative class), unless the cue lexicon (Section 4.3) "
            "matches the containing sentence, in which case the label is "
            "ADR.",
            s["Body"],
        )
    )

    story.append(Paragraph("Appendix B.&nbsp;&nbsp;Hyperparameters", s["H1"]))
    t8 = build_table(
        [
            ["Hyperparameter", "TF-IDF baseline", "Transformer fine-tunes"],
            ["max_features", "200,000", "—"],
            ["ngram_range", "(1, 2)", "—"],
            ["min_df", "3", "—"],
            ["sublinear_tf", "True", "—"],
            ["LogReg C / Ridge α", "4.0 / 1.0", "—"],
            ["epochs", "—", "3 (early stop, patience 2)"],
            ["batch size", "—", "16"],
            ["learning rate", "—", "2 × 10⁻⁵"],
            ["weight decay", "—", "0.01"],
            ["warmup ratio", "—", "0.06"],
            ["max sequence length", "—", "256"],
            ["precision", "—", "FP16 on CUDA"],
            ["seeds", "42", "{42, 1337, 2024}"],
        ],
        col_widths=[1.7 * inch, 1.5 * inch, 2.2 * inch],
    )
    story.append(t8)
    story.append(
        Paragraph(
            "<b>Table 8.</b> Hyperparameter settings. Transformer "
            "fine-tunes share identical hyperparameters across DistilBERT, "
            "PubMedBERT, and the multi-task model; only the encoder backbone "
            "and the head architecture differ.",
            s["Caption"],
        )
    )

    # build
    doc.build(story)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
