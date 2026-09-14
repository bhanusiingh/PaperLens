"""
src/pdf/extractor.py

PDF text extraction using PyMuPDF (fitz).
Handles standard text-based research paper PDFs.
"""

import fitz  # PyMuPDF


class PDFExtractor:
    """
    Extracts raw text from a PDF file page by page.

    Usage:
        extractor = PDFExtractor("path/to/paper.pdf")
        text = extractor.extract_text()
    """

    def __init__(self, pdf_path: str):
        """
        Args:
            pdf_path: Absolute or relative path to the PDF file.
        """
        self.pdf_path = pdf_path

    def extract_text(self) -> str:
        """
        Extract and return the full text content of the PDF.

        Returns:
            A single string containing the concatenated text of all pages.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            RuntimeError: If the PDF cannot be opened or parsed.
        """
        # TODO: Add page-level metadata tracking if needed for section detection
        try:
            doc = fitz.open(self.pdf_path)
        except Exception as exc:
            raise RuntimeError(f"Could not open PDF '{self.pdf_path}': {exc}") from exc

        pages: list[str] = []
        for page in doc:
            pages.append(page.get_text("text"))  # type: ignore[attr-defined]

        doc.close()
        return "\n".join(pages)

    def extract_pages(self) -> list[str]:
        """
        Extract text page by page.

        Returns:
            A list of strings, one per page.
        """
        # TODO: Optionally preserve page boundaries for section detection heuristics
        try:
            doc = fitz.open(self.pdf_path)
        except Exception as exc:
            raise RuntimeError(f"Could not open PDF '{self.pdf_path}': {exc}") from exc

        pages = [page.get_text("text") for page in doc]  # type: ignore[attr-defined]
        doc.close()
        return pages
