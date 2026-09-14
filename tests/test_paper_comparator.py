"""
tests/test_paper_comparator.py

Unit tests for src/comparison/paper_comparator.py (PaperComparator).
"""

import pytest
from src.comparison.paper_comparator import (
    PaperComparator,
    ComparisonReport,
    COMPARISON_DIMENSIONS,
)


@pytest.fixture
def comparator() -> PaperComparator:
    return PaperComparator()


@pytest.fixture
def two_digests() -> list[dict]:
    """Two minimal paper digests with plausible section summaries."""
    return [
        {
            "introduction": (
                "This paper addresses the problem of machine translation using "
                "transformer models. We propose a novel attention mechanism."
            ),
            "methodology": (
                "We use a transformer encoder-decoder architecture with multi-head "
                "attention. The model is pre-trained on a large multilingual corpus."
            ),
            "dataset": (
                "We evaluate on the WMT dataset and the OPUS corpus benchmark."
            ),
            "results": (
                "Our model achieves a BLEU score of 32.5 on the WMT test set. "
                "ROUGE-L is 0.42. F1 is 0.87."
            ),
            "limitations": (
                "The model is slow to train and requires significant GPU memory."
            ),
            "conclusion": "Transformer models are highly effective for translation.",
            "abstract": "",
            "related_work": "",
            "experiments": "",
            "discussion": "",
            "other": "",
        },
        {
            "introduction": (
                "We study text summarization using pre-trained language models. "
                "The problem of long-document summarization is addressed."
            ),
            "methodology": (
                "Our approach uses the LED transformer with sparse attention. "
                "We fine-tune the pre-trained model on arXiv papers."
            ),
            "dataset": (
                "We use the CNN/DM dataset and the arXiv dataset for evaluation."
            ),
            "results": (
                "ROUGE-1: 0.45, ROUGE-2: 0.18, ROUGE-L: 0.40 on the test split. "
                "BERTScore F1 is 0.89."
            ),
            "limitations": (
                "Inference is slow on CPU. OCR is not supported for scanned PDFs."
            ),
            "conclusion": "LED effectively summarizes long research papers.",
            "abstract": "",
            "related_work": "",
            "experiments": "",
            "discussion": "",
            "other": "",
        },
    ]


@pytest.fixture
def three_digests(two_digests) -> list[dict]:
    """Three digests for multi-paper tests."""
    third = {
        "introduction": "Graph neural networks are applied to knowledge reasoning tasks.",
        "methodology": "We use a graph convolutional network with attention layers.",
        "dataset": "The FB15k and WN18RR benchmark datasets are used.",
        "results": "Accuracy: 0.92. mAP: 0.78. ROUGE-1: 0.31.",
        "limitations": "The model does not scale to very large knowledge graphs.",
        "conclusion": "GNNs are effective for knowledge graph completion.",
        "abstract": "",
        "related_work": "",
        "experiments": "",
        "discussion": "",
        "other": "",
    }
    return two_digests + [third]


class TestPaperComparator:

    def test_compare_returns_report(self, comparator, two_digests):
        report = comparator.compare(two_digests)
        assert isinstance(report, ComparisonReport)

    def test_dimension_table_has_all_dimensions(self, comparator, two_digests):
        report = comparator.compare(two_digests)
        for dim in COMPARISON_DIMENSIONS:
            assert dim in report.dimension_table

    def test_dimension_table_aligned_with_papers(self, comparator, two_digests):
        titles = ["Paper A", "Paper B"]
        report = comparator.compare(two_digests, titles=titles)
        for dim in COMPARISON_DIMENSIONS:
            assert len(report.dimension_table[dim]) == len(titles)

    def test_titles_default_fallback(self, comparator, two_digests):
        report = comparator.compare(two_digests)
        assert report.paper_titles == ["Paper 1", "Paper 2"]

    def test_custom_titles(self, comparator, two_digests):
        titles = ["MT Paper", "Summarization Paper"]
        report = comparator.compare(two_digests, titles=titles)
        assert report.paper_titles == titles

    def test_key_differences_not_placeholder(self, comparator, two_digests):
        report = comparator.compare(two_digests)
        placeholder = "Manually review the dimension table"
        assert placeholder not in report.key_differences, (
            "key_differences still contains placeholder text"
        )

    def test_common_approaches_not_placeholder(self, comparator, two_digests):
        report = comparator.compare(two_digests)
        placeholder = "Manually review the dimension table"
        assert placeholder not in report.common_approaches, (
            "common_approaches still contains placeholder text"
        )

    def test_key_differences_is_non_empty_string(self, comparator, two_digests):
        report = comparator.compare(two_digests)
        assert isinstance(report.key_differences, str)
        assert len(report.key_differences.strip()) > 0

    def test_common_approaches_is_non_empty_string(self, comparator, two_digests):
        report = comparator.compare(two_digests)
        assert isinstance(report.common_approaches, str)
        assert len(report.common_approaches.strip()) > 0

    def test_transformer_in_shared_methods(self, comparator, two_digests):
        """Both digests mention 'transformer' — it should appear in common approaches."""
        report = comparator.compare(two_digests)
        assert "transformer" in report.common_approaches.lower()

    def test_different_datasets_in_key_differences(self, comparator, two_digests):
        """The two digests reference different datasets — check differences mention datasets."""
        report = comparator.compare(two_digests)
        # The differences section should mention dataset names
        assert any(
            kw in report.key_differences
            for kw in ["WMT", "CNN", "Dataset", "dataset", "Datasets"]
        )

    def test_three_papers(self, comparator, three_digests):
        report = comparator.compare(three_digests)
        assert len(report.paper_titles) == 3
        for dim in COMPARISON_DIMENSIONS:
            assert len(report.dimension_table[dim]) == 3

    def test_too_few_papers_raises(self, comparator, two_digests):
        with pytest.raises(ValueError, match="2.5 papers"):
            comparator.compare([two_digests[0]])

    def test_too_many_papers_raises(self, comparator, two_digests):
        six = two_digests * 3
        with pytest.raises(ValueError, match="2.5 papers"):
            comparator.compare(six)

    def test_to_dict_structure(self, comparator, two_digests):
        report = comparator.compare(two_digests)
        d = report.to_dict()
        assert "papers" in d
        assert "comparison" in d
        assert "key_differences" in d
        assert "common_approaches" in d

    def test_metric_extraction_bleu(self, comparator):
        metrics = comparator._extract_metric_terms("BLEU score of 32.5, F1 = 0.87")
        assert any("BLEU" in m.upper() for m in metrics)

    def test_vocabulary_extraction(self, comparator):
        vocab = comparator._extract_vocabulary("we use a transformer encoder", ["transformer", "lstm"])
        assert "transformer" in vocab
        assert "lstm" not in vocab
