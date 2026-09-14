"""
src/preprocessing/text_cleaner.py

Text preprocessing for raw PDF-extracted content.
Cleans noise artifacts introduced by PDF extraction before downstream processing.
"""

import re


class TextCleaner:
    """
    Cleans raw text extracted from research paper PDFs.

    Handles common PDF extraction artifacts such as:
    - Hyphenated line breaks (de- composed → decomposed)
    - Excessive whitespace and blank lines
    - Page headers / footers (heuristic removal)
    - Non-printable / control characters
    - Reference list markers (optional, configurable)

    Usage:
        cleaner = TextCleaner()
        clean = cleaner.clean(raw_text)
    """

    def __init__(self, remove_references: bool = False):
        """
        Args:
            remove_references: If True, attempts to strip the bibliography/references
                section from the end of the document. Disabled by default so that
                callers can decide.
        """
        self.remove_references = remove_references

    def clean(self, text: str) -> str:
        """
        Apply the full cleaning pipeline to raw extracted text.

        Args:
            text: Raw string from PDFExtractor.

        Returns:
            Cleaned string ready for section detection and summarization.
        """
        text = self._remove_control_characters(text)
        text = self._fix_hyphenation(text)
        text = self._normalize_whitespace(text)
        if self.remove_references:
            text = self._strip_references_section(text)
        return text.strip()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _remove_control_characters(text: str) -> str:
        """Remove non-printable / control characters except newlines and tabs."""
        return re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", " ", text)

    @staticmethod
    def _fix_hyphenation(text: str) -> str:
        """Rejoin words split across lines with a hyphen (common in two-column PDFs)."""
        return re.sub(r"-\n(\w)", r"\1", text)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Collapse multiple spaces and limit consecutive blank lines to two."""
        text = re.sub(r"[ \t]+", " ", text)          # collapse horizontal whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)        # max two consecutive newlines
        return text

    @staticmethod
    def _strip_references_section(text: str) -> str:
        """
        Heuristically remove the references section from the end of the document.
        Looks for common heading variants and truncates from there.
        """
        # TODO: Improve reference boundary detection if needed
        pattern = re.compile(
            r"\n\s*(References|Bibliography|Works Cited)\s*\n",
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if match:
            return text[: match.start()]
        return text
