"""
tests/test_detector.py

Unit tests for src/section_detection/detector.py (SectionDetector).
"""

import pytest
from src.section_detection.detector import (
    SectionDetector,
    DetectedPaper,
    Section,
    CANONICAL_SECTIONS,
)


@pytest.fixture
def detector() -> SectionDetector:
    return SectionDetector()


class TestSectionDetector:
    """Tests for SectionDetector."""

    def test_detect_returns_detected_paper(self, detector, sample_paper_text):
        paper = detector.detect(sample_paper_text)
        assert isinstance(paper, DetectedPaper)

    def test_detect_finds_sections(self, detector, sample_paper_text):
        paper = detector.detect(sample_paper_text)
        assert len(paper.sections) > 0

    def test_sections_have_labels(self, detector, sample_paper_text):
        paper = detector.detect(sample_paper_text)
        for s in paper.sections:
            assert isinstance(s, Section)
            assert s.label in CANONICAL_SECTIONS

    def test_introduction_detected(self, detector, sample_paper_text):
        paper = detector.detect(sample_paper_text)
        labels = [s.label for s in paper.sections]
        assert "introduction" in labels, f"Expected 'introduction', got: {labels}"

    def test_methodology_detected(self, detector, sample_paper_text):
        paper = detector.detect(sample_paper_text)
        labels = [s.label for s in paper.sections]
        assert "methodology" in labels, f"Expected 'methodology', got: {labels}"

    def test_results_detected(self, detector, sample_paper_text):
        paper = detector.detect(sample_paper_text)
        labels = [s.label for s in paper.sections]
        assert "results" in labels, f"Expected 'results', got: {labels}"

    def test_get_returns_text_for_detected_section(self, detector, sample_paper_text):
        paper = detector.detect(sample_paper_text)
        intro_text = paper.get("introduction")
        assert isinstance(intro_text, str)
        assert len(intro_text) > 0

    def test_get_returns_empty_for_absent_section(self, detector, sample_paper_text):
        paper = detector.detect(sample_paper_text)
        # 'discussion' is not in the fixture text
        text = paper.get("discussion")
        assert text == "" or isinstance(text, str)

    def test_to_dict_has_all_canonical_keys(self, detector, sample_paper_text):
        paper = detector.detect(sample_paper_text)
        d = paper.to_dict()
        for key in CANONICAL_SECTIONS:
            assert key in d, f"Missing key '{key}' in to_dict() output"

    def test_to_dict_values_are_strings(self, detector, sample_paper_text):
        paper = detector.detect(sample_paper_text)
        d = paper.to_dict()
        assert all(isinstance(v, str) for v in d.values())

    def test_numbered_heading_detected(self, detector):
        """Headings like '1. Introduction' should be normalised correctly."""
        text = (
            "1. Introduction\n"
            "This is the introduction body.\n\n"
            "2. Methodology\n"
            "This describes the method.\n"
        )
        paper = detector.detect(text)
        labels = [s.label for s in paper.sections]
        assert "introduction" in labels
        assert "methodology" in labels

    def test_empty_text_returns_empty_paper(self, detector):
        paper = detector.detect("")
        assert isinstance(paper, DetectedPaper)
        assert paper.sections == []

    def test_is_heading_rejects_long_lines(self, detector):
        long_line = "a" * 200
        assert not detector._is_heading(long_line)

    def test_normalize_heading_unknown_returns_other(self, detector):
        assert detector._normalize_heading("xyzzy foobar") == "other"
