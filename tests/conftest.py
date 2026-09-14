"""
tests/conftest.py

Shared pytest fixtures for PaperLens tests.
"""

import io
import pytest


# ---------------------------------------------------------------------------
# Minimal PDF fixture (created without PyMuPDF so tests run without a real PDF)
# ---------------------------------------------------------------------------

def _make_minimal_pdf(text: str) -> bytes:
    """
    Build a minimal, spec-compliant single-page PDF that contains ``text``
    as a simple text stream.  This avoids any dependency on external PDF
    files and keeps the test suite self-contained.
    """
    # Encode text safely for PDF content streams (ASCII subset)
    safe_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    content_stream = (
        f"BT\n/F1 12 Tf\n50 750 Td\n({safe_text}) Tj\nET"
    )
    stream_bytes = content_stream.encode("latin-1")
    stream_len = len(stream_bytes)

    objects: list[str] = []

    # Object 1 — Catalog
    objects.append("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    # Object 2 — Pages
    objects.append("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    # Object 3 — Page
    objects.append(
        "3 0 obj\n"
        "<< /Type /Page /Parent 2 0 R\n"
        "   /MediaBox [0 0 612 792]\n"
        "   /Contents 4 0 R\n"
        "   /Resources << /Font << /F1 5 0 R >> >> >>\n"
        "endobj\n"
    )
    # Object 4 — Content stream
    objects.append(
        f"4 0 obj\n<< /Length {stream_len} >>\nstream\n"
        + content_stream
        + "\nendstream\nendobj\n"
    )
    # Object 5 — Font (Type1 Helvetica — universally supported)
    objects.append(
        "5 0 obj\n"
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\n"
        "endobj\n"
    )

    body = "%PDF-1.4\n"
    offsets: list[int] = []
    for obj in objects:
        offsets.append(len(body))
        body += obj

    xref_offset = len(body)
    xref = f"xref\n0 {len(objects) + 1}\n"
    xref += "0000000000 65535 f \n"
    for off in offsets:
        xref += f"{off:010d} 00000 n \n"

    trailer = (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF"
    )

    return (body + xref + trailer).encode("latin-1")


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Return raw bytes of a minimal valid PDF with known text content."""
    return _make_minimal_pdf(
        "Introduction This paper proposes a new method for text summarization. "
        "Methodology We use a transformer-based approach. "
        "Results The method achieves state-of-the-art performance. "
        "Conclusion We conclude that transformers are effective."
    )


@pytest.fixture
def sample_pdf_path(tmp_path, sample_pdf_bytes) -> str:
    """Write the minimal PDF to a temp file and return its path."""
    pdf_file = tmp_path / "sample_paper.pdf"
    pdf_file.write_bytes(sample_pdf_bytes)
    return str(pdf_file)


@pytest.fixture
def sample_paper_text() -> str:
    """A short multi-section paper text for section detection tests."""
    return (
        "Abstract\n"
        "This paper introduces PaperLens, a system for research paper summarization.\n\n"
        "Introduction\n"
        "Research papers are long and time-consuming to read. "
        "We propose an automated summarization pipeline.\n\n"
        "Methodology\n"
        "We use the LED transformer model (allenai/led-large-16384-arxiv) "
        "fine-tuned on arXiv papers. "
        "The model supports up to 16384 tokens of input.\n\n"
        "Dataset\n"
        "We evaluate on the ccdv/arxiv-summarization dataset "
        "using the document configuration.\n\n"
        "Results\n"
        "Our method achieves ROUGE-1 of 0.45, ROUGE-2 of 0.18, ROUGE-L of 0.40.\n\n"
        "Limitations\n"
        "The model is slow on CPU and requires GPU for practical use.\n\n"
        "Conclusion\n"
        "PaperLens provides accurate structured summaries of research papers.\n"
    )
