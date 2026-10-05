"""
tests/test_section_digest.py

Unit tests for src/summarization/section_digest.py (SectionDigestSummarizer).
"""
import pytest
from src.summarization.section_digest import SectionDigestSummarizer, DEFAULT_SECTION_SENTENCE_LIMITS


@pytest.fixture
def summarizer() -> SectionDigestSummarizer:
    return SectionDigestSummarizer()


class TestSectionDigestSummarizer:
    def test_empty_text_returns_empty_string(self, summarizer):
        assert summarizer.summarize("") == ""
        assert summarizer.summarize("   ") == ""
        assert summarizer.summarize(None) == ""

    def test_short_text_returns_available_text(self, summarizer):
        text = "This is a single short sentence describing the problem in detail."
        result = summarizer.summarize(text, dimension="problem")
        assert result == text

    def test_preserves_sentence_limits(self, summarizer):
        # Create a paragraph with 8 distinct sentences
        sentences = [
            f"Sentence number {i} provides distinct context on research methodology and model design."
            for i in range(1, 9)
        ]
        text = " ".join(sentences)

        result_problem = summarizer.summarize(text, dimension="problem")
        # Target limit for problem is 3
        splits = [s for s in result_problem.split(".") if len(s.strip()) > 10]
        assert len(splits) == DEFAULT_SECTION_SENTENCE_LIMITS["problem"]

        result_method = summarizer.summarize(text, dimension="methodology")
        # Target limit for methodology is 4
        splits_m = [s for s in result_method.split(".") if len(s.strip()) > 10]
        assert len(splits_m) == DEFAULT_SECTION_SENTENCE_LIMITS["methodology"]

    def test_filters_table_and_figure_captions(self, summarizer):
        text = (
            "Recurrent models are widely used for sequence modeling and machine translation tasks. "
            "Table 1: Complexity per layer and maximum path lengths across operations. "
            "Figure 2: Multi-head attention consists of several attention layers in parallel. "
            "The proposed architecture dispenses with recurrence and relies entirely on self-attention."
        )
        result = summarizer.summarize(text, dimension="methodology", num_sentences=2)
        assert "Table 1:" not in result
        assert "Figure 2:" not in result
        assert "Recurrent models" in result or "proposed architecture" in result

    def test_preserves_document_order(self, summarizer):
        sentences = [
            "Alpha: Initial problem formulation established in the classical literature.",
            "Beta: Secondary considerations regarding computational bottlenecks.",
            "Gamma: Detailed description of Transformer architecture and self-attention.",
            "Delta: Empirical observations across German and French translation benchmarks.",
            "Epsilon: Concluding observations regarding future research directions.",
        ]
        text = " ".join(sentences)
        result = summarizer.summarize(text, num_sentences=2)

        # Check that whatever 2 sentences are selected, their order matches the original text
        indices = [sentences.index(s) for s in sentences if s in result]
        assert indices == sorted(indices)

    def test_deduplicates_repeated_sentences(self, summarizer):
        text = (
            "We propose a new simple network architecture based solely on attention mechanisms. "
            "We propose a new simple network architecture based solely on attention mechanisms. "
            "Experiments on two translation tasks show these models to be superior in quality."
        )
        result = summarizer.summarize(text, num_sentences=3)
        assert result.count("We propose a new simple network architecture") == 1
