"""
app/pipeline.py

Orchestrates the PaperLens NLP pipeline for one or more PDF files.

Pipeline per paper:
    PDFExtractor  → extract full text
    TextCleaner   → clean text
    SectionDetector → detect sections → section_dict
    LEDSummarizer  → summarize(full_clean_text) → led_summary
    TFIDFSummarizer → summarize(full_clean_text) → tfidf_summary

ROUGE is NOT computed here — it requires a reference abstract that
the user provides in the app.  Compute it separately via /api/rouge.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from src.pdf.extractor import PDFExtractor
from src.preprocessing.text_cleaner import TextCleaner
from src.section_detection.detector import SectionDetector
from src.summarization.tfidf_baseline import TFIDFSummarizer

# Digest dimension → canonical section keys (priority order)
DIGEST_MAP: dict[str, list[str]] = {
    "problem":     ["introduction", "abstract"],
    "methodology": ["methodology"],
    "dataset":     ["dataset"],
    "results":     ["results", "experiments"],
    "limitations": ["limitations"],
    "conclusion":  ["conclusion"],
}


def process_paper(pdf_path: str) -> dict:
    """
    Run the full pipeline on a single PDF.

    Returns a dict with all data needed by the API.
    LED summary is computed lazily on first request to avoid blocking
    startup; TF-IDF is fast and always computed.
    """
    paper_id = str(uuid.uuid4())
    filename = Path(pdf_path).name

    # 1. Extract
    raw_text = PDFExtractor(pdf_path).extract_text()

    # 2. Clean
    cleaner = TextCleaner()
    clean_text = cleaner.clean(raw_text)

    # 3. Detect sections
    detector = SectionDetector()
    detected_paper = detector.detect(clean_text)
    section_dict = detected_paper.to_dict()
    detected_section_labels = [s.label for s in detected_paper.sections]

    # 4. Build structured digest from raw section text
    digest: dict[str, str] = {}
    for dim, source_keys in DIGEST_MAP.items():
        text = ""
        for key in source_keys:
            text = section_dict.get(key, "").strip()
            if text:
                break
        digest[dim] = text

    # 5. TF-IDF summary of full paper (fast — always run)
    tfidf = TFIDFSummarizer(num_sentences=5)
    tfidf_summary = tfidf.summarize(clean_text)

    # 6. LED summary — deferred: compute if not already cached
    # LED is expensive; run separately via run_led_summary()
    led_summary = None

    return {
        "paper_id": paper_id,
        "filename": filename,
        "full_text_length": len(clean_text),
        "led_summary": led_summary,      # None until computed
        "tfidf_summary": tfidf_summary,
        "rouge": None,                   # None until reference abstract provided
        "digest": digest,
        "metadata": {
            "detected_sections": detected_section_labels,
        },
        "_clean_text": clean_text,       # retained for LED on-demand
    }


def run_led_summary(clean_text: str) -> str:
    """Run LED on clean_text (heavy — requires torch + model)."""
    from src.summarization.led_summarizer import LEDSummarizer
    led = LEDSummarizer()
    return led.summarize(clean_text)
