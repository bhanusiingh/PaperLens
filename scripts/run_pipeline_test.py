"""
scripts/run_pipeline_test.py

Phase 1 verification: runs the complete PaperLens NLP pipeline on a
real or synthetic input and prints every intermediate result.

Usage (with a real PDF):
    python scripts/run_pipeline_test.py --pdf path/to/paper.pdf --abstract "Author abstract here."

Usage (synthetic text, no PDF needed — good for first-pass verification):
    python scripts/run_pipeline_test.py --synthetic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure the project root is importable.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.text_cleaner import TextCleaner
from src.section_detection.detector import SectionDetector
from src.summarization.tfidf_baseline import TFIDFSummarizer
from src.evaluation.rouge_evaluator import ROUGEEvaluator

# Digest dimension → source section label(s)
DIGEST_MAP = {
    "Problem":     ["introduction", "abstract"],
    "Method":      ["methodology"],
    "Dataset":     ["dataset"],
    "Results":     ["results", "experiments"],
    "Limitations": ["limitations"],
    "Conclusion":  ["conclusion"],
}

SYNTHETIC_TEXT = """\
Abstract
This paper proposes PaperLens, a system for structured summarization of research papers
using the LED transformer model and TF-IDF baseline. We evaluate on ccdv/arxiv-summarization.

Introduction
Long research papers are difficult to read in full. We propose an automated pipeline that
detects sections, summarizes the full paper, and computes ROUGE scores against reference
abstracts. The LED model handles documents up to 16384 tokens.

Methodology
We use allenai/led-large-16384-arxiv for abstractive summarization and a TF-IDF baseline
for extractive summarization. Rule-based section detection maps headings to canonical labels.

Dataset
We evaluate on the ccdv/arxiv-summarization dataset (document configuration),
using the article field as input and the abstract field as the reference summary.

Results
The LED model achieves ROUGE-1 of 0.45, ROUGE-2 of 0.18, ROUGE-L of 0.40 on the test set.
The TF-IDF baseline achieves ROUGE-1 of 0.31, ROUGE-2 of 0.10, ROUGE-L of 0.29.

Limitations
The model is slow on CPU. Very long papers (>16K tokens) require hierarchical chunking
which may lose some coherence across chunk boundaries.

Conclusion
PaperLens accurately extracts and summarises key information from research papers.
The LED model outperforms the TF-IDF baseline on all ROUGE metrics.
"""

SYNTHETIC_REFERENCE = (
    "We propose PaperLens, a research paper summarization system using the LED "
    "transformer and TF-IDF baseline. Evaluated on ccdv/arxiv-summarization, "
    "LED achieves ROUGE-1 0.45, outperforming the extractive TF-IDF baseline."
)


def sep(title: str = "") -> None:
    line = "=" * 60
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


def run(clean_text: str, reference_abstract: str, use_led: bool) -> None:
    cleaner = TextCleaner()
    detector = SectionDetector()
    tfidf = TFIDFSummarizer(num_sentences=5)
    evaluator = ROUGEEvaluator()

    # --- Clean ---
    sep("1. TEXT CLEANING")
    clean = cleaner.clean(clean_text)
    print(f"  Input length : {len(clean_text):,} chars")
    print(f"  Cleaned length: {len(clean):,} chars")

    # --- Detect sections ---
    sep("2. SECTION DETECTION")
    paper = detector.detect(clean)
    section_dict = paper.to_dict()
    detected = [s.label for s in paper.sections]
    print(f"  Detected sections ({len(detected)}): {detected}")

    # --- Structured digest ---
    sep("3. STRUCTURED DIGEST (from section_dict)")
    for dim, source_keys in DIGEST_MAP.items():
        text = ""
        for key in source_keys:
            text = section_dict.get(key, "").strip()
            if text:
                break
        preview = (text[:120] + "…") if len(text) > 120 else text or "[not detected]"
        print(f"\n  {dim}:\n    {preview}")

    # --- TF-IDF summary (always) ---
    sep("4. TF-IDF SUMMARY (full paper)")
    tfidf_summary = tfidf.summarize(clean)
    print(f"  Length: {len(tfidf_summary)} chars")
    print(f"  Preview: {tfidf_summary[:200]}…")

    # --- LED summary (optional — requires torch + transformers) ---
    led_summary: str | None = None
    if use_led:
        sep("5. LED SUMMARY (full paper) — loading model…")
        from src.summarization.led_summarizer import LEDSummarizer
        led = LEDSummarizer()
        led_summary = led.summarize(clean)
        print(f"  Length: {len(led_summary)} chars")
        print(f"  Preview: {led_summary[:200]}…")
    else:
        sep("5. LED SUMMARY — skipped (use --led to enable)")
        print("  Pass --led flag to run LED inference (requires torch + model download)")

    # --- ROUGE ---
    sep("6. ROUGE EVALUATION")
    print(f"\n  Reference abstract:\n    {reference_abstract[:200]}\n")

    tfidf_scores = evaluator.score(tfidf_summary, reference_abstract)
    print(f"  TF-IDF vs Reference: {tfidf_scores}")

    if led_summary is not None:
        led_scores = evaluator.score(led_summary, reference_abstract)
        print(f"  LED    vs Reference: {led_scores}")

    sep()
    print("  Pipeline verification complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="PaperLens pipeline verification")
    parser.add_argument("--pdf", type=str, default=None, help="Path to PDF file")
    parser.add_argument("--abstract", type=str, default=None,
                        help="Reference abstract for ROUGE evaluation")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use built-in synthetic text (no PDF needed)")
    parser.add_argument("--led", action="store_true",
                        help="Run LED inference (slow; requires torch + model download)")
    args = parser.parse_args()

    if args.synthetic:
        print("Using synthetic text (no PDF required).")
        run(SYNTHETIC_TEXT, SYNTHETIC_REFERENCE, use_led=args.led)
        return

    if args.pdf is None:
        print("Provide --pdf <path> or --synthetic. Use --help for usage.")
        sys.exit(1)

    from src.pdf.extractor import PDFExtractor
    print(f"Extracting text from: {args.pdf}")
    raw = PDFExtractor(args.pdf).extract_text()

    reference = args.abstract or ""
    if not reference:
        print("\n[WARNING] No --abstract provided. ROUGE will compare against an empty string.")
        print("          Pass --abstract \"paste abstract here\" for meaningful scores.\n")

    run(raw, reference, use_led=args.led)


if __name__ == "__main__":
    main()
