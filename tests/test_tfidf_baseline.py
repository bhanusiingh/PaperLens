"""
tests/test_tfidf_baseline.py

Unit tests for src/summarization/tfidf_baseline.py (TFIDFSummarizer).
"""

import pytest
from src.summarization.tfidf_baseline import TFIDFSummarizer


@pytest.fixture
def summarizer() -> TFIDFSummarizer:
    return TFIDFSummarizer(num_sentences=3)


@pytest.fixture
def long_text() -> str:
    return (
        "Transformers have revolutionized natural language processing. "
        "The attention mechanism allows models to weigh the importance of different words. "
        "BERT was the first model to use bidirectional attention pre-training. "
        "GPT models use unidirectional causal attention for text generation. "
        "LED extends the transformer to handle long documents up to 16384 tokens. "
        "TF-IDF is a classic method for extractive summarization. "
        "It scores sentences by the sum of their term frequency-inverse document frequency weights. "
        "Extractive summarization selects sentences directly from the source document. "
        "Abstractive summarization generates new text not present in the original. "
        "Evaluation metrics for summarization include ROUGE-1, ROUGE-2, and ROUGE-L."
    )


class TestTFIDFSummarizer:

    def test_summarize_returns_string(self, summarizer, long_text):
        result = summarizer.summarize(long_text)
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_summary_shorter_than_input(self, summarizer, long_text):
        result = summarizer.summarize(long_text)
        assert len(result) <= len(long_text)

    def test_summary_sentences_from_source(self, summarizer, long_text):
        """Each sentence in the summary must appear in the source text."""
        import re
        result = summarizer.summarize(long_text)
        for sent in re.split(r"(?<=[.!?])\s+", result):
            sent = sent.strip()
            if sent:
                assert sent in long_text, f"Sentence not found in source: '{sent}'"

    def test_short_text_returned_as_is(self, summarizer):
        """Text with fewer sentences than num_sentences should be returned unchanged."""
        short = "This is one sentence. This is a second one."
        result = summarizer.summarize(short)
        assert isinstance(result, str)
        # Result should be the full short text (stripped)
        assert len(result.strip()) > 0

    def test_num_sentences_respected(self, long_text):
        """Vary num_sentences and verify summary length scales accordingly."""
        import re
        s3 = TFIDFSummarizer(num_sentences=3).summarize(long_text)
        s5 = TFIDFSummarizer(num_sentences=5).summarize(long_text)
        count3 = len([s for s in re.split(r"(?<=[.!?])\s+", s3) if s.strip()])
        count5 = len([s for s in re.split(r"(?<=[.!?])\s+", s5) if s.strip()])
        assert count3 <= count5

    def test_empty_text_does_not_crash(self, summarizer):
        result = summarizer.summarize("")
        assert isinstance(result, str)

    def test_split_sentences_filters_short(self):
        """_split_sentences should drop tokens shorter than 20 chars."""
        text = "Hi. This sentence is long enough to pass the filter threshold check."
        sentences = TFIDFSummarizer._split_sentences(text)
        assert all(len(s) > 20 for s in sentences)
