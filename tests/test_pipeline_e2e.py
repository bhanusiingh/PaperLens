"""
tests/test_pipeline_e2e.py

End-to-end pipeline test (no GPU required, no model download).

This test exercises the complete text-only pipeline:
  TextCleaner → SectionDetector → TFIDFSummarizer → PaperComparator → ROUGEEvaluator

The LED model is NOT loaded so the test runs quickly without GPU or network.
"""

import pytest
from src.preprocessing.text_cleaner import TextCleaner
from src.section_detection.detector import SectionDetector
from src.summarization.tfidf_baseline import TFIDFSummarizer
from src.comparison.paper_comparator import PaperComparator, COMPARISON_DIMENSIONS
from src.evaluation.rouge_evaluator import ROUGEEvaluator


PAPER_A = (
    "Abstract\n"
    "This paper proposes a transformer-based approach for text summarization "
    "using the LED model on arXiv papers.\n\n"
    "Introduction\n"
    "Long document summarization is challenging for standard transformers because "
    "of their limited context window. LED extends the context to 16384 tokens.\n\n"
    "Methodology\n"
    "We use the allenai/led-large-16384-arxiv pre-trained model. "
    "Global attention is placed on the first token. "
    "Beam search with four beams generates the final summary.\n\n"
    "Dataset\n"
    "We use the ccdv/arxiv-summarization dataset with the document configuration. "
    "Input is the article field; target is the abstract.\n\n"
    "Results\n"
    "Our transformer model achieves ROUGE-1 of 0.45, ROUGE-2 of 0.18, ROUGE-L of 0.40 "
    "on the arXiv test set, outperforming the TF-IDF baseline.\n\n"
    "Limitations\n"
    "The model requires significant GPU memory and is slow on CPU hardware.\n\n"
    "Conclusion\n"
    "LED-based summarization significantly improves structured digest quality.\n"
)

PAPER_B = (
    "Abstract\n"
    "We propose an extractive baseline using TF-IDF for research paper summarization.\n\n"
    "Introduction\n"
    "Extractive summarization selects sentences directly from the source document "
    "without generating new text. It is fast and interpretable.\n\n"
    "Methodology\n"
    "Sentences are ranked by their TF-IDF weight sums using scikit-learn. "
    "The top-k sentences in document order form the summary.\n\n"
    "Dataset\n"
    "We evaluate on CNN/DM and the arXiv benchmark corpus.\n\n"
    "Results\n"
    "TF-IDF achieves ROUGE-1 of 0.32, ROUGE-2 of 0.10, ROUGE-L of 0.29. "
    "BLEU score is 0.15.\n\n"
    "Limitations\n"
    "Extractive methods cannot generate abstractions not present in the text.\n\n"
    "Conclusion\n"
    "TF-IDF provides a simple but effective extractive baseline.\n"
)

REFERENCE_ABSTRACT = (
    "We present a transformer-based long document summarization system using LED. "
    "The system supports arXiv research papers and achieves strong ROUGE scores."
)


class TestEndToEndPipeline:
    """Full text pipeline without LED model loading."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.cleaner = TextCleaner()
        self.detector = SectionDetector()
        self.tfidf = TFIDFSummarizer(num_sentences=3)
        self.comparator = PaperComparator()
        self.evaluator = ROUGEEvaluator()

    def _process_paper(self, raw_text: str) -> tuple[dict[str, str], dict[str, str]]:
        """Run cleaning → detection → TF-IDF summarization on raw_text."""
        clean = self.cleaner.clean(raw_text)
        paper = self.detector.detect(clean)
        section_dict = paper.to_dict()

        sections_to_summarize = [
            "introduction", "methodology", "dataset", "results", "limitations", "conclusion"
        ]
        summaries: dict[str, str] = {}
        for key in sections_to_summarize:
            text = section_dict.get(key, "")
            summaries[key] = self.tfidf.summarize(text) if text else "_Section not detected._"
        return section_dict, summaries

    # ------------------------------------------------------------------

    def test_cleaning_runs(self):
        clean = self.cleaner.clean(PAPER_A)
        assert isinstance(clean, str)
        assert len(clean) > 0

    def test_section_detection_finds_sections(self):
        clean = self.cleaner.clean(PAPER_A)
        paper = self.detector.detect(clean)
        labels = [s.label for s in paper.sections]
        assert "introduction" in labels
        assert "methodology" in labels

    def test_tfidf_summarization_produces_output(self):
        _, summaries = self._process_paper(PAPER_A)
        for key in ["introduction", "methodology", "results"]:
            assert isinstance(summaries[key], str)
            assert len(summaries[key]) > 0

    def test_full_paper_tfidf_summary(self):
        clean = self.cleaner.clean(PAPER_A)
        summary = self.tfidf.summarize(clean)
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_rouge_evaluation(self):
        clean = self.cleaner.clean(PAPER_A)
        generated = self.tfidf.summarize(clean)
        scores = self.evaluator.score(generated, REFERENCE_ABSTRACT)
        assert 0.0 <= scores.rouge1 <= 1.0
        assert 0.0 <= scores.rouge2 <= 1.0
        assert 0.0 <= scores.rougeL <= 1.0
        assert scores.rouge2 <= scores.rouge1 + 1e-6

    def test_multi_paper_comparison(self):
        _, sums_a = self._process_paper(PAPER_A)
        _, sums_b = self._process_paper(PAPER_B)

        clean_a = self.cleaner.clean(PAPER_A)
        clean_b = self.cleaner.clean(PAPER_B)
        sect_a = self.detector.detect(clean_a).to_dict()
        sect_b = self.detector.detect(clean_b).to_dict()

        # Merge summaries into section dicts
        for key, val in sums_a.items():
            if val and val != "_Section not detected._":
                sect_a[key] = val
        for key, val in sums_b.items():
            if val and val != "_Section not detected._":
                sect_b[key] = val

        report = self.comparator.compare([sect_a, sect_b], ["Paper A", "Paper B"])

        assert report.paper_titles == ["Paper A", "Paper B"]
        for dim in COMPARISON_DIMENSIONS:
            assert dim in report.dimension_table
            assert len(report.dimension_table[dim]) == 2

    def test_key_differences_no_placeholder(self):
        _, sums_a = self._process_paper(PAPER_A)
        _, sums_b = self._process_paper(PAPER_B)
        clean_a = self.cleaner.clean(PAPER_A)
        clean_b = self.cleaner.clean(PAPER_B)
        sect_a = self.detector.detect(clean_a).to_dict()
        sect_b = self.detector.detect(clean_b).to_dict()
        for key, val in sums_a.items():
            if val and val != "_Section not detected._":
                sect_a[key] = val
        for key, val in sums_b.items():
            if val and val != "_Section not detected._":
                sect_b[key] = val
        report = self.comparator.compare([sect_a, sect_b])
        assert "Manually review" not in report.key_differences
        assert "Manually review" not in report.common_approaches

    def test_rouge_improves_with_related_text(self):
        """A related summary should score higher than a completely unrelated one."""
        reference = "LED transformer summarizes arXiv research papers using attention."
        related   = self.tfidf.summarize(PAPER_A)
        unrelated = "The weather today is sunny with light winds from the west."
        s_related   = self.evaluator.score(related, reference)
        s_unrelated = self.evaluator.score(unrelated, reference)
        assert s_related.rouge1 > s_unrelated.rouge1
