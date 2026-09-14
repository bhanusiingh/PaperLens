"""
tests/test_rouge_evaluator.py

Unit tests for src/evaluation/rouge_evaluator.py (ROUGEEvaluator).
"""

import pytest
from src.evaluation.rouge_evaluator import ROUGEEvaluator, ROUGEScores


@pytest.fixture(scope="module")
def evaluator() -> ROUGEEvaluator:
    """Single evaluator instance shared across all tests in this module."""
    return ROUGEEvaluator()


class TestROUGEScores:

    def test_to_dict_keys(self):
        s = ROUGEScores(rouge1=0.5, rouge2=0.3, rougeL=0.45)
        d = s.to_dict()
        assert set(d.keys()) == {"rouge1", "rouge2", "rougeL"}

    def test_to_dict_rounded(self):
        s = ROUGEScores(rouge1=0.123456789, rouge2=0.0, rougeL=1.0)
        d = s.to_dict()
        assert d["rouge1"] == round(0.123456789, 4)

    def test_str_format(self):
        s = ROUGEScores(rouge1=0.5, rouge2=0.3, rougeL=0.45)
        text = str(s)
        assert "ROUGE-1" in text
        assert "ROUGE-2" in text
        assert "ROUGE-L" in text


class TestROUGEEvaluator:

    def test_score_returns_rouge_scores(self, evaluator):
        scores = evaluator.score("the cat sat on the mat", "the cat sat on a mat")
        assert isinstance(scores, ROUGEScores)

    def test_identical_strings_give_perfect_scores(self, evaluator):
        text = "the quick brown fox jumps over the lazy dog"
        scores = evaluator.score(text, text)
        assert scores.rouge1 == pytest.approx(1.0, abs=1e-4)
        assert scores.rougeL == pytest.approx(1.0, abs=1e-4)

    def test_completely_different_strings_give_low_scores(self, evaluator):
        scores = evaluator.score(
            "machine learning is great for image classification",
            "the sunset over the ocean was beautiful and peaceful",
        )
        assert scores.rouge1 < 0.2
        assert scores.rouge2 < 0.1

    def test_scores_in_range(self, evaluator):
        scores = evaluator.score(
            "transformers revolutionized natural language processing",
            "transformers changed the field of NLP significantly",
        )
        for val in [scores.rouge1, scores.rouge2, scores.rougeL]:
            assert 0.0 <= val <= 1.0

    def test_score_batch_returns_list(self, evaluator):
        preds = ["the cat sat", "a dog runs fast"]
        refs  = ["the cat sat on the mat", "a fast dog is running"]
        result = evaluator.score_batch(preds, refs)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(r, ROUGEScores) for r in result)

    def test_score_batch_length_mismatch_raises(self, evaluator):
        with pytest.raises(ValueError):
            evaluator.score_batch(["one"], ["one", "two"])

    def test_average_returns_rouge_scores(self, evaluator):
        scores_list = [
            ROUGEScores(0.4, 0.2, 0.35),
            ROUGEScores(0.6, 0.3, 0.55),
        ]
        avg = evaluator.average(scores_list)
        assert isinstance(avg, ROUGEScores)
        assert avg.rouge1 == pytest.approx(0.5, abs=1e-6)
        assert avg.rouge2 == pytest.approx(0.25, abs=1e-6)
        assert avg.rougeL == pytest.approx(0.45, abs=1e-6)

    def test_average_empty_list_raises(self, evaluator):
        with pytest.raises(ValueError):
            evaluator.average([])

    def test_rouge2_leq_rouge1(self, evaluator):
        """ROUGE-2 should never exceed ROUGE-1 (bigram ≤ unigram overlap)."""
        scores = evaluator.score(
            "this is a test of the summarization system output",
            "this is a test of the summarization quality and coverage",
        )
        assert scores.rouge2 <= scores.rouge1 + 1e-6
