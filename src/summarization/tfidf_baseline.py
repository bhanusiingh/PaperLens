"""
src/summarization/tfidf_baseline.py

TF-IDF extractive summarization baseline.

Ranks sentences by their TF-IDF importance scores and returns the top-k
sentences as an extractive summary. Used as a baseline against the LED
transformer model.
"""

from __future__ import annotations

import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class TFIDFSummarizer:
    """
    Extractive summarization using TF-IDF sentence scoring.

    Sentences are ranked by the sum of their token TF-IDF weights.
    The top-ranked sentences (in document order) form the summary.

    Usage:
        baseline = TFIDFSummarizer(num_sentences=5)
        summary = baseline.summarize(paper_text)
    """

    def __init__(self, num_sentences: int = 5):
        """
        Args:
            num_sentences: Number of top sentences to include in the summary.
        """
        self.num_sentences = num_sentences

    def summarize(self, text: str) -> str:
        """
        Return an extractive summary of the input text.

        Args:
            text: Cleaned paper text.

        Returns:
            A string containing the top-ranked sentences joined by spaces.
        """
        sentences = self._split_sentences(text)
        if len(sentences) <= self.num_sentences:
            return text.strip()

        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(sentences)  # (n_sentences, vocab)

        # Score each sentence by summing its TF-IDF weights
        scores: np.ndarray = tfidf_matrix.sum(axis=1).A1  # type: ignore[attr-defined]

        # Pick top-k sentence indices, preserve document order
        top_indices = sorted(
            np.argsort(scores)[-self.num_sentences :].tolist()
        )

        summary_sentences = [sentences[i] for i in top_indices]
        return " ".join(summary_sentences)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """
        Naively split text into sentences on common sentence-ending punctuation.
        Returns only non-empty sentence strings.
        """
        # TODO: Replace with a proper sentence tokenizer (e.g., nltk punkt)
        #       if sentence boundary detection accuracy becomes a bottleneck.
        raw = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in raw if len(s.strip()) > 20]
