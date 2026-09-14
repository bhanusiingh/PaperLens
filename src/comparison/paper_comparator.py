"""
src/comparison/paper_comparator.py

Multi-paper comparison module.

Given a list of structured paper digests (dicts from DetectedPaper.to_dict(),
after section summaries have been applied), produces:

  * A dimension-by-dimension table (problem / methodology / dataset / results /
    limitations) with one column per paper.
  * A ``key_differences`` narrative, derived from the already-generated
    section summaries using lightweight heuristics — no additional model
    inference.
  * A ``common_approaches`` narrative, similarly derived.

No RAG, no vector search, no external model calls.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Dimensions surfaced in the comparison output
# ---------------------------------------------------------------------------
COMPARISON_DIMENSIONS: list[str] = [
    "problem",
    "methodology",
    "dataset",
    "results",
    "limitations",
]


@dataclass
class ComparisonReport:
    """
    Holds the structured comparison of 2–5 research papers.

    Attributes:
        paper_titles: Display names for each paper (index-aligned with all
            list fields in ``dimension_table``).
        dimension_table: Mapping of dimension → list of per-paper summaries.
        key_differences: Narrative describing salient differences across
            papers, derived from their section summaries.
        common_approaches: Narrative describing shared themes or methods,
            derived from their section summaries.
    """

    paper_titles: list[str]
    dimension_table: dict[str, list[str]] = field(default_factory=dict)
    key_differences: str = ""
    common_approaches: str = ""

    def to_dict(self) -> dict:
        return {
            "papers": self.paper_titles,
            "comparison": self.dimension_table,
            "key_differences": self.key_differences,
            "common_approaches": self.common_approaches,
        }


# ---------------------------------------------------------------------------
# Vocabulary lists used for heuristic analysis
# ---------------------------------------------------------------------------

# Terms that typically indicate a method or approach — used to extract
# common methodological vocabulary across papers.
_METHOD_TERMS: list[str] = [
    "transformer", "bert", "gpt", "attention", "encoder", "decoder",
    "lstm", "rnn", "cnn", "convolutional", "recurrent", "fine-tun",
    "pretrain", "pre-train", "reinforcement", "reward", "policy",
    "graph", "neural network", "deep learning", "classification",
    "regression", "clustering", "embedding", "representation",
    "zero-shot", "few-shot", "prompt", "chain-of-thought",
    "contrastive", "diffusion", "generative", "adversarial",
]

# Terms that indicate the domain or task — used to extract shared problem areas.
_TASK_TERMS: list[str] = [
    "summarization", "summarisation", "question answering", "translation",
    "generation", "classification", "detection", "recognition",
    "segmentation", "retrieval", "extraction", "parsing",
    "sentiment", "named entity", "coreference", "dialogue",
    "reasoning", "commonsense", "machine reading",
]


class PaperComparator:
    """
    Compares structured digests from multiple research papers.

    The comparison is built entirely from section summaries already generated
    by ``LEDSummarizer`` or ``TFIDFSummarizer`` — no additional model calls
    are made here.

    Key differences and common approaches are derived using:
      * Vocabulary intersection / difference across papers' method and
        result sections.
      * Dataset name extraction from the dataset sections.
      * Metric term detection in the results sections.

    Usage::

        comparator = PaperComparator()
        report = comparator.compare(digests, titles)
        print(report.key_differences)
    """

    # Mapping from comparison dimension → canonical section keys (priority order)
    _DIMENSION_SOURCES: dict[str, list[str]] = {
        "problem":      ["introduction", "abstract"],
        "methodology":  ["methodology"],
        "dataset":      ["dataset"],
        "results":      ["results", "experiments"],
        "limitations":  ["limitations"],
    }

    def compare(
        self,
        digests: list[dict[str, str]],
        titles: list[str] | None = None,
    ) -> ComparisonReport:
        """
        Build a ``ComparisonReport`` from a list of paper section dicts.

        Args:
            digests: List of dicts returned by ``DetectedPaper.to_dict()``
                (or equivalent), one per paper, with section summaries already
                inserted as values where applicable.
            titles: Optional display names.  Defaults to ``["Paper 1", ...]``.

        Returns:
            ``ComparisonReport`` with dimension table, key differences, and
            common approaches all populated.

        Raises:
            ``ValueError`` if fewer than 2 or more than 5 digests are supplied.
        """
        if not (2 <= len(digests) <= 5):
            raise ValueError(
                f"PaperComparator requires 2–5 papers; got {len(digests)}."
            )

        n = len(digests)
        titles = titles or [f"Paper {i + 1}" for i in range(n)]

        # Build dimension table -----------------------------------------------
        dimension_table: dict[str, list[str]] = {}
        for dim, source_keys in self._DIMENSION_SOURCES.items():
            per_paper: list[str] = []
            for digest in digests:
                text = ""
                for key in source_keys:
                    text = digest.get(key, "").strip()
                    if text:
                        break
                per_paper.append(text or "Not available.")
            dimension_table[dim] = per_paper

        # Derive narratives ---------------------------------------------------
        key_differences = self._derive_key_differences(digests, titles, dimension_table)
        common_approaches = self._derive_common_approaches(digests, titles)

        return ComparisonReport(
            paper_titles=titles,
            dimension_table=dimension_table,
            key_differences=key_differences,
            common_approaches=common_approaches,
        )

    # ------------------------------------------------------------------
    # Narrative generation helpers
    # ------------------------------------------------------------------

    def _derive_key_differences(
        self,
        digests: list[dict[str, str]],
        titles: list[str],
        dimension_table: dict[str, list[str]],
    ) -> str:
        """
        Generate a key-differences narrative from the dimension table.

        Approach:
          - Compare dataset names across papers (extracted by regex).
          - Compare method vocabulary per paper (extracted from methodology).
          - Compare metric terms in results sections.
          - Summarise findings in readable bullet points.
        """
        lines: list[str] = []

        # 1. Dataset differences
        datasets = [
            self._extract_dataset_names(dimension_table["dataset"][i])
            for i in range(len(titles))
        ]
        unique_datasets: list[str] = []
        for title, ds in zip(titles, datasets):
            if ds:
                unique_datasets.append(f"{title}: {', '.join(sorted(ds))}")
            else:
                unique_datasets.append(f"{title}: dataset not detected")
        lines.append("**Datasets used:**")
        lines.extend(f"  • {s}" for s in unique_datasets)

        # 2. Method vocabulary differences
        method_vocabs: list[set[str]] = [
            self._extract_vocabulary(
                digests[i].get("methodology", ""), _METHOD_TERMS
            )
            for i in range(len(titles))
        ]
        # Terms unique to each paper
        for i, (title, vocab) in enumerate(zip(titles, method_vocabs)):
            others: set[str] = set().union(
                *(method_vocabs[j] for j in range(len(titles)) if j != i)
            )
            unique = vocab - others
            if unique:
                lines.append(
                    f"**{title}** uses methods not shared by other papers: "
                    + ", ".join(sorted(unique))
                    + "."
                )

        # 3. Results / metric differences
        result_vocabs: list[set[str]] = [
            self._extract_metric_terms(digests[i].get("results", ""))
            for i in range(len(titles))
        ]
        all_metrics: set[str] = set().union(*result_vocabs)
        if all_metrics:
            per_paper_metrics: list[str] = []
            for title, rv in zip(titles, result_vocabs):
                if rv:
                    per_paper_metrics.append(f"{title}: {', '.join(sorted(rv))}")
            if per_paper_metrics:
                lines.append("**Evaluation metrics detected:**")
                lines.extend(f"  • {s}" for s in per_paper_metrics)

        if not lines:
            return (
                "Section summaries were insufficient to extract automatic "
                "differences. Review the dimension table above."
            )
        return "\n".join(lines)

    def _derive_common_approaches(
        self,
        digests: list[dict[str, str]],
        titles: list[str],
    ) -> str:
        """
        Identify shared methodological vocabulary and task terminology across
        all papers and return a readable narrative.
        """
        method_vocabs: list[set[str]] = [
            self._extract_vocabulary(
                digests[i].get("methodology", ""), _METHOD_TERMS
            )
            for i in range(len(titles))
        ]
        task_vocabs: list[set[str]] = [
            self._extract_vocabulary(
                digests[i].get("introduction", "") + " " +
                digests[i].get("abstract", ""),
                _TASK_TERMS,
            )
            for i in range(len(titles))
        ]

        shared_methods: set[str] = method_vocabs[0].copy()
        for v in method_vocabs[1:]:
            shared_methods &= v

        shared_tasks: set[str] = task_vocabs[0].copy()
        for v in task_vocabs[1:]:
            shared_tasks &= v

        lines: list[str] = []
        if shared_methods:
            lines.append(
                "**Shared methodological approaches:** "
                + ", ".join(sorted(shared_methods))
                + "."
            )
        if shared_tasks:
            lines.append(
                "**Common task areas:** "
                + ", ".join(sorted(shared_tasks))
                + "."
            )

        # Shared dataset vocabulary (not full names, just overlapping words)
        all_dataset_names: list[set[str]] = [
            self._extract_dataset_names(digests[i].get("dataset", ""))
            for i in range(len(titles))
        ]
        shared_datasets: set[str] = all_dataset_names[0].copy()
        for v in all_dataset_names[1:]:
            shared_datasets &= v
        if shared_datasets:
            lines.append(
                "**Datasets referenced by all papers:** "
                + ", ".join(sorted(shared_datasets))
                + "."
            )

        if not lines:
            return (
                "No clearly shared methods, tasks, or datasets were detected "
                "from the section summaries. Review the dimension table above."
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Text analysis utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_vocabulary(text: str, term_list: list[str]) -> set[str]:
        """
        Return the subset of ``term_list`` terms present in ``text``
        (case-insensitive substring match).
        """
        text_lower = text.lower()
        return {term for term in term_list if term in text_lower}

    @staticmethod
    def _extract_dataset_names(text: str) -> set[str]:
        """
        Heuristically extract dataset names from a section text.

        Looks for patterns like:
          * All-caps acronyms: SQuAD, MNLI, CNN/DM
          * Title-cased multi-word phrases before "dataset" / "benchmark"
          * Quoted names
        """
        found: set[str] = set()

        # All-caps tokens (likely dataset/benchmark acronyms), 2–10 chars
        for m in re.finditer(r"\b[A-Z][A-Z0-9\-/]{1,9}\b", text):
            found.add(m.group())

        # Title-cased N-grams immediately before "dataset" or "benchmark"
        for m in re.finditer(
            r"([A-Z][a-zA-Z0-9]*(?:\s[A-Z][a-zA-Z0-9]*){0,3})\s+"
            r"(?:dataset|benchmark|corpus|collection)",
            text,
            re.IGNORECASE,
        ):
            found.add(m.group(1).strip())

        # Single-quoted or double-quoted names
        for m in re.finditer(r'["\']([A-Za-z0-9][^"\']{1,40})["\']', text):
            candidate = m.group(1).strip()
            if len(candidate.split()) <= 5:
                found.add(candidate)

        # Filter noise: remove very common English words and single letters
        noise = {"The", "A", "An", "In", "For", "We", "Our", "This", "These",
                 "I", "II", "III", "IV", "V", "NLP", "AI", "ML"}
        return found - noise

    @staticmethod
    def _extract_metric_terms(text: str) -> set[str]:
        """
        Extract common evaluation metric names from a results section.
        """
        metric_patterns = [
            r"\bROUGE[-\s]?\d?\b",
            r"\bBLEU\b",
            r"\bBERT(?:Score)?\b",
            r"\bF[-_]?1\b",
            r"\baccuracy\b",
            r"\bprecision\b",
            r"\brecall\b",
            r"\bMETEOR\b",
            r"\bperplexity\b",
            r"\bAUC\b",
            r"\bmAP\b",
        ]
        found: set[str] = set()
        for pat in metric_patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                found.add(m.group().strip())
        return found
