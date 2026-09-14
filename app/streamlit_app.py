"""
app/streamlit_app.py

PaperLens — Streamlit web interface.

Provides a UI for:
  1. Uploading 1–5 research paper PDFs.
  2. Running the extraction → preprocessing → section detection →
     summarization pipeline on each paper.
  3. Viewing the structured research digest per paper.
  4. Comparing multiple papers side by side.
  5. Evaluating a full-paper summary against a reference abstract using ROUGE.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path when running via:
#   streamlit run app/streamlit_app.py
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from src.pdf.extractor import PDFExtractor
from src.preprocessing.text_cleaner import TextCleaner
from src.section_detection.detector import SectionDetector
from src.summarization.led_summarizer import LEDSummarizer
from src.summarization.tfidf_baseline import TFIDFSummarizer
from src.comparison.paper_comparator import PaperComparator
from src.evaluation.rouge_evaluator import ROUGEEvaluator


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PaperLens",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")
use_led = st.sidebar.radio(
    "Summarizer",
    options=["LED (Transformer)", "TF-IDF (Baseline)"],
    index=0,
)
num_sentences = st.sidebar.slider(
    "TF-IDF: Sentences per section", min_value=2, max_value=10, value=5
)
st.sidebar.markdown("---")
st.sidebar.info(
    "**Model:** allenai/led-large-16384-arxiv\n\n"
    "**Dataset:** ccdv/arxiv-summarization\n\n"
    "**Course:** CSE472 — Deep Learning for NLP"
)

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
st.title("🔬 PaperLens")
st.markdown("### Structured Research Paper Summarization using Transformers")
st.markdown(
    "Upload 1–5 research paper PDFs to generate structured digests and "
    "compare papers side by side."
)

uploaded_files = st.file_uploader(
    "Upload PDF(s)",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    if len(uploaded_files) > 5:
        st.error("Please upload a maximum of 5 papers.")
        st.stop()

    # -----------------------------------------------------------------------
    # Pipeline setup (cached per session to avoid redundant model loads)
    # -----------------------------------------------------------------------
    @st.cache_resource(show_spinner="Loading LED model…")
    def load_led() -> LEDSummarizer:
        return LEDSummarizer()

    cleaner = TextCleaner()
    detector = SectionDetector()
    tfidf = TFIDFSummarizer(num_sentences=num_sentences)

    # -----------------------------------------------------------------------
    # Per-paper processing
    # -----------------------------------------------------------------------
    digests: list[dict[str, str]] = []      # raw section text per paper
    summaries: list[dict[str, str]] = []    # generated summaries per paper
    full_texts: list[str] = []              # full cleaned text per paper
    full_summaries: list[str] = []          # full-paper summaries (for ROUGE)
    titles: list[str] = []

    tabs = st.tabs([f.name for f in uploaded_files])

    for tab, uploaded_file in zip(tabs, uploaded_files):
        with tab:
            st.subheader(f"📄 {uploaded_file.name}")

            # Save upload to data/raw/
            tmp_path = _PROJECT_ROOT / "data" / "raw" / uploaded_file.name
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_bytes(uploaded_file.read())

            with st.spinner("Extracting text…"):
                extractor = PDFExtractor(str(tmp_path))
                raw_text = extractor.extract_text()

            with st.spinner("Preprocessing…"):
                clean_text = cleaner.clean(raw_text)

            with st.spinner("Detecting sections…"):
                paper = detector.detect(clean_text)
                section_dict = paper.to_dict()

            digests.append(section_dict)
            full_texts.append(clean_text)
            titles.append(uploaded_file.name)

            # ---------------------------------------------------------------
            # Section-level digest
            # ---------------------------------------------------------------
            DIGEST_SECTIONS = [
                ("Problem / Introduction", "introduction"),
                ("Methodology",            "methodology"),
                ("Dataset",                "dataset"),
                ("Results",                "results"),
                ("Limitations",            "limitations"),
                ("Conclusion",             "conclusion"),
            ]

            paper_summary: dict[str, str] = {}

            with st.spinner("Summarizing sections…"):
                for _display, key in DIGEST_SECTIONS:
                    text_chunk = section_dict.get(key, "")
                    if not text_chunk:
                        paper_summary[key] = "_Section not detected._"
                        continue
                    if use_led == "LED (Transformer)":
                        paper_summary[key] = load_led().summarize(text_chunk)
                    else:
                        paper_summary[key] = tfidf.summarize(text_chunk)

            summaries.append(paper_summary)

            # ---------------------------------------------------------------
            # Full-paper summary (used for ROUGE evaluation)
            # ---------------------------------------------------------------
            with st.spinner("Generating full-paper summary for ROUGE…"):
                if use_led == "LED (Transformer)":
                    full_summary = load_led().summarize(clean_text)
                else:
                    full_summary = tfidf.summarize(clean_text)
            full_summaries.append(full_summary)

            # ---------------------------------------------------------------
            # Display structured digest
            # ---------------------------------------------------------------
            st.markdown("#### 📋 Structured Digest")
            for display_name, key in DIGEST_SECTIONS:
                with st.expander(display_name, expanded=True):
                    st.write(paper_summary.get(key, "_Not available._"))

            # ---------------------------------------------------------------
            # ROUGE evaluation
            # The generated summary is the *full-paper* summary produced above,
            # compared against the user-provided reference abstract.
            # ---------------------------------------------------------------
            st.markdown("---")
            st.markdown("#### 📊 ROUGE Evaluation (optional)")
            st.caption(
                "Paste the paper's reference abstract below. "
                "ROUGE scores will be computed between the **full-paper summary** "
                "generated above and the reference abstract."
            )
            reference_abstract = st.text_area(
                "Reference abstract:",
                key=f"ref_{uploaded_file.name}",
                height=120,
            )
            if reference_abstract.strip():
                evaluator = ROUGEEvaluator()
                scores = evaluator.score(full_summary, reference_abstract.strip())
                col1, col2, col3 = st.columns(3)
                col1.metric("ROUGE-1", f"{scores.rouge1:.4f}")
                col2.metric("ROUGE-2", f"{scores.rouge2:.4f}")
                col3.metric("ROUGE-L", f"{scores.rougeL:.4f}")

                with st.expander("Full-paper summary used for ROUGE scoring"):
                    st.write(full_summary)

    # -----------------------------------------------------------------------
    # Multi-paper comparison
    # -----------------------------------------------------------------------
    if len(uploaded_files) > 1:
        st.markdown("---")
        st.markdown("## 📊 Multi-Paper Comparison")

        # Feed section summaries (not raw section text) into the comparator
        # so that key_differences / common_approaches are derived from the
        # already-generated summaries rather than from raw extracted text.
        summary_digests: list[dict[str, str]] = []
        for raw_digest, summary in zip(digests, summaries):
            merged = dict(raw_digest)   # keep raw for fallback
            for key, val in summary.items():
                if val and val != "_Section not detected._":
                    merged[key] = val   # prefer the generated summary
            summary_digests.append(merged)

        comparator = PaperComparator()
        report = comparator.compare(summary_digests, titles)

        dimension_labels = {
            "problem":      "Problem Statement",
            "methodology":  "Methodology",
            "dataset":      "Dataset",
            "results":      "Results",
            "limitations":  "Limitations",
        }

        for dim, label in dimension_labels.items():
            with st.expander(f"🔹 Compare: {label}", expanded=False):
                cols = st.columns(len(titles))
                for col, title, text in zip(
                    cols, titles, report.dimension_table.get(dim, [])
                ):
                    with col:
                        st.markdown(f"**{title}**")
                        st.write(text or "_Not available._")

        st.markdown("#### Key Differences")
        st.info(report.key_differences)

        st.markdown("#### Common Approaches")
        st.info(report.common_approaches)
