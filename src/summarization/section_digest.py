"""
src/summarization/section_digest.py

Concise extractive summarization for canonical research paper sections.
Uses TF-IDF sentence scoring to distill detected section text into 2-5
representative sentences while preserving original sentence order.
"""
from __future__ import annotations

import re
from typing import Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Canonical sentence count targets per digest dimension
DEFAULT_SECTION_SENTENCE_LIMITS: dict[str, int] = {
    "problem": 3,       # 2-4 sentences
    "methodology": 4,   # 3-5 sentences
    "dataset": 3,       # 2-4 sentences
    "results": 3,       # 2-4 sentences
    "limitations": 2,   # 2-4 sentences
    "conclusion": 3,    # 2-4 sentences
}


class SectionDigestSummarizer:
    """
    Distills raw detected section text into a concise, representative summary
    using TF-IDF sentence salience ranking.
    """

    def __init__(self, sentence_limits: Optional[dict[str, int]] = None):
        self.limits = sentence_limits or DEFAULT_SECTION_SENTENCE_LIMITS

    def summarize(
        self,
        section_text: str,
        dimension: Optional[str] = None,
        num_sentences: Optional[int] = None,
    ) -> str:
        """
        Extract the most salient sentences from section text.

        Args:
            section_text: Raw text of the detected canonical section.
            dimension: Canonical dimension name (e.g. 'problem', 'methodology').
            num_sentences: Optional override for sentence count target.

        Returns:
            A concise string containing the top-ranked sentences in document order.
        """
        if not section_text or not section_text.strip():
            return ""

        target_k = num_sentences or (self.limits.get(dimension, 3) if dimension else 3)

        # 1. Split into raw sentences
        raw_splits = re.split(r"(?<=[.!?])\s+", section_text)

        # 2. Filter and normalize sentences
        candidate_sents: list[str] = []
        seen: set[str] = set()

        for raw in raw_splits:
            cleaned = self._clean_sentence(raw)
            if self._is_valid_sentence(cleaned):
                norm = cleaned.lower()
                if norm not in seen:
                    seen.add(norm)
                    candidate_sents.append(cleaned)

        # Fallback if filters were overly restrictive
        if not candidate_sents:
            candidate_sents = [
                self._clean_sentence(s) for s in raw_splits if len(s.strip()) > 20
            ]

        # If section is very short, return all available candidate sentences
        if len(candidate_sents) <= target_k:
            return " ".join(candidate_sents)

        # 3. TF-IDF sentence ranking
        try:
            vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
            tfidf_matrix = vectorizer.fit_transform(candidate_sents)
            scores: np.ndarray = tfidf_matrix.sum(axis=1).A1

            # Select top-k sentences and preserve original document order
            top_indices = sorted(np.argsort(scores)[-target_k:].tolist())
            selected = [candidate_sents[i] for i in top_indices]
            return " ".join(selected)
        except Exception:
            # Fallback to first k sentences if vectorizer fails (e.g. no English terms)
            return " ".join(candidate_sents[:target_k])

    @staticmethod
    def _clean_sentence(s: str) -> str:
        """Collapse multiple spaces and internal newlines into single spaces."""
        return re.sub(r"\s+", " ", s).strip()

    @staticmethod
    def _is_valid_sentence(s: str) -> bool:
        """Filter out non-sentence artifacts, captions, and formulas."""
        if len(s) < 25 or len(s) > 450:
            return False

        # Filter out table and figure captions
        if re.match(r"^(Table|Figure|Fig\.)\s+\d+", s, re.IGNORECASE):
            return False

        # Must have a reasonable ratio of alphabetic characters
        letters = sum(c.isalpha() for c in s)
        if letters / len(s) < 0.60:
            return False

        # Filter out standalone reference numbers e.g. "[12]"
        if re.match(r"^\[\d+\]", s):
            return False

        return True
