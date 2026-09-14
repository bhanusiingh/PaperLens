"""
src/evaluation/rouge_evaluator.py

ROUGE evaluation using the Hugging Face `evaluate` library.

Computes ROUGE-1, ROUGE-2, and ROUGE-L between generated summaries and
reference texts (e.g., the paper abstract from ccdv/arxiv-summarization).
"""

from __future__ import annotations

from dataclasses import dataclass
import evaluate


@dataclass
class ROUGEScores:
    """Container for ROUGE metric results."""
    rouge1: float
    rouge2: float
    rougeL: float

    def to_dict(self) -> dict[str, float]:
        return {
            "rouge1": round(self.rouge1, 4),
            "rouge2": round(self.rouge2, 4),
            "rougeL": round(self.rougeL, 4),
        }

    def __str__(self) -> str:
        return (
            f"ROUGE-1: {self.rouge1:.4f} | "
            f"ROUGE-2: {self.rouge2:.4f} | "
            f"ROUGE-L: {self.rougeL:.4f}"
        )


class ROUGEEvaluator:
    """
    Wraps the Hugging Face `evaluate` ROUGE metric for convenient use.

    Supports both single-pair and batch evaluation.

    Usage (single pair):
        evaluator = ROUGEEvaluator()
        scores = evaluator.score(prediction="...", reference="...")
        print(scores)

    Usage (batch):
        scores_list = evaluator.score_batch(predictions=[...], references=[...])
    """

    def __init__(self):
        # Load once; evaluate downloads/caches the metric automatically
        self._rouge = evaluate.load("rouge")

    def score(self, prediction: str, reference: str) -> ROUGEScores:
        """
        Compute ROUGE scores for a single prediction/reference pair.

        Args:
            prediction: Generated summary text.
            reference: Reference summary text (e.g., paper abstract).

        Returns:
            ROUGEScores dataclass with rouge1, rouge2, rougeL fields.
        """
        result = self._rouge.compute(
            predictions=[prediction],
            references=[reference],
            use_stemmer=True,
        )
        return ROUGEScores(
            rouge1=result["rouge1"],
            rouge2=result["rouge2"],
            rougeL=result["rougeL"],
        )

    def score_batch(
        self,
        predictions: list[str],
        references: list[str],
    ) -> list[ROUGEScores]:
        """
        Compute ROUGE scores for a batch of prediction/reference pairs.

        Args:
            predictions: List of generated summary texts.
            references: List of reference summary texts.

        Returns:
            List of ROUGEScores, one per pair.
        """
        if len(predictions) != len(references):
            raise ValueError(
                "predictions and references must have the same length."
            )
        return [
            self.score(pred, ref)
            for pred, ref in zip(predictions, references)
        ]

    def average(self, scores: list[ROUGEScores]) -> ROUGEScores:
        """
        Compute the mean ROUGE scores across a list of ROUGEScores.

        Args:
            scores: Non-empty list of ROUGEScores.

        Returns:
            A single ROUGEScores with averaged values.
        """
        if not scores:
            raise ValueError("Cannot average an empty list of scores.")
        n = len(scores)
        return ROUGEScores(
            rouge1=sum(s.rouge1 for s in scores) / n,
            rouge2=sum(s.rouge2 for s in scores) / n,
            rougeL=sum(s.rougeL for s in scores) / n,
        )
