"""
src/section_detection/detector.py

Rule-based section detector for research papers.

Uses heading keyword patterns to identify and normalize section boundaries
into a fixed set of canonical sections expected by the summarization pipeline.
No ML model is used — pattern matching only.

Heading detection heuristics
-----------------------------
A line is treated as a section heading if ALL of the following are true:
  1. It is non-empty and short (≤ max_heading_length characters).
  2. After stripping any leading section number (e.g. "1.", "2.3", "III."),
     it matches one of the heading keyword patterns.
  3. It does NOT start with a common sentence starter (we/our/the/this/a/an/
     in/for/by/with/it/these/those/such) which indicates a body sentence that
     happens to contain a keyword.
  4. It is NOT a long sentence containing multiple words before the keyword,
     which would indicate that the keyword appears mid-sentence rather than as
     the heading itself.
"""

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Canonical section labels used throughout the pipeline
# ---------------------------------------------------------------------------
CANONICAL_SECTIONS = [
    "abstract",
    "introduction",
    "related_work",
    "methodology",
    "dataset",
    "experiments",
    "results",
    "discussion",
    "limitations",
    "conclusion",
    "other",
]

# ---------------------------------------------------------------------------
# Heading → canonical section mapping
# Each key is a regex pattern (case-insensitive) that maps to a canonical label.
# Patterns use \b word boundaries and are anchored to the start of the core
# heading text (after stripping section numbers) to avoid false matches inside
# body sentences.
# ---------------------------------------------------------------------------
HEADING_PATTERNS: dict[str, str] = {
    r"^abstract$":                                    "abstract",
    r"^introduction":                                 "introduction",
    r"^related\s+work|^background|^prior\s+work|^literature\s+review": "related_work",
    r"^method(ology)?$|^approach$|^proposed\s+(method|model|approach|framework)": "methodology",
    r"^dataset$|^data\s+collection$|^data\s+preparation$": "dataset",
    r"^experiment(s|al\s+setup)?$|^experimental\s+results$": "experiments",
    r"^results?$|^findings$|^quantitative\s+results?$": "results",
    r"^discussion$|^analysis$":                       "discussion",
    r"^limitations?$":                                "limitations",
    r"^conclusion(s)?$|^summary$|^concluding\s+remarks$": "conclusion",
}

# Words that start a body sentence rather than a heading.
# A line whose first (non-numeric) token is one of these is not a heading.
_SENTENCE_STARTERS = {
    "we", "our", "the", "this", "these", "those", "a", "an",
    "in", "for", "by", "with", "it", "such", "their", "its",
    "here", "however", "furthermore", "moreover", "also", "given",
}


@dataclass
class Section:
    """Represents a detected section of a research paper."""
    label: str          # canonical label
    raw_heading: str    # heading text as found in the document
    text: str           # body text of the section


@dataclass
class DetectedPaper:
    """Container for all sections detected from a single paper."""
    sections: list[Section] = field(default_factory=list)

    def get(self, label: str) -> str:
        """Return concatenated text for a canonical section label, or empty string."""
        texts = [s.text for s in self.sections if s.label == label]
        return "\n\n".join(texts).strip()

    def to_dict(self) -> dict[str, str]:
        """Return a dict mapping canonical labels to section text."""
        result: dict[str, str] = {k: "" for k in CANONICAL_SECTIONS}
        for section in self.sections:
            if result[section.label]:
                result[section.label] += "\n\n" + section.text
            else:
                result[section.label] = section.text
        return result


class SectionDetector:
    """
    Splits a cleaned research paper into labelled sections using rule-based
    heading detection.

    Headings are identified by lines that:
      1. Are short (configurable max length).
      2. Match one of the heading keyword patterns (anchored to the start of
         the core text after stripping any leading section number).
      3. Do not start with a sentence-starter word (we, our, the, this, …).

    Usage:
        detector = SectionDetector()
        paper = detector.detect(cleaned_text)
        methodology_text = paper.get("methodology")
    """

    def __init__(self, max_heading_length: int = 80):
        """
        Args:
            max_heading_length: Lines longer than this are not treated as headings.
        """
        self.max_heading_length = max_heading_length
        self._compiled = [
            (re.compile(pattern, re.IGNORECASE), label)
            for pattern, label in HEADING_PATTERNS.items()
        ]

    def detect(self, text: str) -> DetectedPaper:
        """
        Detect sections in a cleaned paper text.

        Args:
            text: Cleaned paper text from TextCleaner.

        Returns:
            DetectedPaper containing a list of Section objects.
        """
        lines = text.splitlines()
        segments: list[tuple[str, str]] = []  # (raw_heading, body_text)
        current_heading = "preamble"
        current_body: list[str] = []

        for line in lines:
            stripped = line.strip()
            if self._is_heading(stripped):
                # Flush the previous segment
                segments.append((current_heading, "\n".join(current_body).strip()))
                current_heading = stripped
                current_body = []
            else:
                current_body.append(line)

        # Flush the last segment
        segments.append((current_heading, "\n".join(current_body).strip()))

        # Build DetectedPaper
        paper = DetectedPaper()
        for raw_heading, body in segments:
            if not body:
                continue
            label = self._normalize_heading(raw_heading)
            paper.sections.append(Section(label=label, raw_heading=raw_heading, text=body))

        return paper

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_heading(self, line: str) -> bool:
        """Return True if the line looks like a section heading."""
        if not line or len(line) > self.max_heading_length:
            return False

        # Strip optional leading section number, e.g. "1.", "2.3", "III."
        core = re.sub(r"^[\dIVXivx]+[.\s]+", "", line).strip()
        if not core:
            return False

        # Reject lines that start with a sentence-starter word.
        first_word = core.split()[0].lower().rstrip(".,;:")
        if first_word in _SENTENCE_STARTERS:
            return False

        return any(pat.search(core) for pat, _ in self._compiled)

    def _normalize_heading(self, raw: str) -> str:
        """Map a raw heading string to a canonical section label."""
        core = re.sub(r"^[\dIVXivx]+[.\s]+", "", raw).strip()
        for pattern, label in self._compiled:
            if pattern.search(core):
                return label
        return "other"
