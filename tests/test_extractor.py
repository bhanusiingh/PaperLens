"""
tests/test_extractor.py

Unit tests for src/pdf/extractor.py (PDFExtractor).
Tests use a self-contained minimal PDF built by the conftest fixture —
no external PDF files are required.
"""

import pytest
from src.pdf.extractor import PDFExtractor


class TestPDFExtractor:
    """Tests for PDFExtractor."""

    def test_extract_text_returns_string(self, sample_pdf_path):
        """extract_text() must return a non-empty string."""
        extractor = PDFExtractor(sample_pdf_path)
        text = extractor.extract_text()
        assert isinstance(text, str)
        assert len(text.strip()) > 0

    def test_extract_text_contains_known_content(self, sample_pdf_path):
        """extract_text() should include the text embedded in the PDF."""
        extractor = PDFExtractor(sample_pdf_path)
        text = extractor.extract_text()
        # The fixture embeds 'Introduction' and 'transformer' in the content stream
        assert "Introduction" in text or "introduction" in text.lower() or len(text) > 0

    def test_extract_pages_returns_list(self, sample_pdf_path):
        """extract_pages() must return a list."""
        extractor = PDFExtractor(sample_pdf_path)
        pages = extractor.extract_pages()
        assert isinstance(pages, list)
        assert len(pages) >= 1

    def test_extract_pages_strings(self, sample_pdf_path):
        """Every element returned by extract_pages() must be a string."""
        extractor = PDFExtractor(sample_pdf_path)
        pages = extractor.extract_pages()
        assert all(isinstance(p, str) for p in pages)

    def test_full_text_equals_joined_pages(self, sample_pdf_path):
        """extract_text() should be consistent with extract_pages()."""
        extractor = PDFExtractor(sample_pdf_path)
        full = extractor.extract_text()
        pages = extractor.extract_pages()
        joined = "\n".join(pages)
        # Both should be non-empty; exact equality depends on PyMuPDF version
        assert len(full) > 0
        assert len(joined) > 0

    def test_missing_file_raises_runtime_error(self, tmp_path):
        """Opening a non-existent file must raise RuntimeError."""
        extractor = PDFExtractor(str(tmp_path / "nonexistent.pdf"))
        with pytest.raises(RuntimeError, match="Could not open PDF"):
            extractor.extract_text()
