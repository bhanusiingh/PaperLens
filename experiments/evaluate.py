"""
experiments/evaluate.py

Benchmark evaluation script for PaperLens.

Loads a configurable number of examples from ccdv/arxiv-summarization
(document configuration), generates summaries with both LED and the TF-IDF
baseline, computes ROUGE-1/2/L, and writes results to
results/evaluation_results.json.

Usage
-----
    python experiments/evaluate.py                # uses defaults from config.yaml
    python experiments/evaluate.py --num-samples 20
    python experiments/evaluate.py --num-samples 5 --split test
    python experiments/evaluate.py --tfidf-only   # skip LED (faster)

This script is separate from the Streamlit app and is intended for
reproducible benchmarking on the ccdv/arxiv-summarization dataset.
It does NOT require a PDF upload — it works directly on the dataset's
pre-extracted text fields.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure the project root is importable when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml
from datasets import load_dataset

from src.summarization.led_summarizer import LEDSummarizer
from src.summarization.tfidf_baseline import TFIDFSummarizer
from src.evaluation.rouge_evaluator import ROUGEEvaluator, ROUGEScores


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "config.yaml"
_RESULTS_DIR = _PROJECT_ROOT / "results"


def load_config() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LED and TF-IDF summarizers on ccdv/arxiv-summarization."
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=10,
        help="Number of dataset examples to evaluate (default: 10).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to use (default: test).",
    )
    parser.add_argument(
        "--tfidf-only",
        action="store_true",
        default=False,
        help="Skip LED inference (fast baseline-only run).",
    )
    parser.add_argument(
        "--tfidf-sentences",
        type=int,
        default=5,
        help="Number of sentences for TF-IDF baseline (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write JSON results (default: results/evaluation_results.json).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def truncate_for_display(text: str, max_chars: int = 120) -> str:
    return text[:max_chars] + "…" if len(text) > max_chars else text


def format_scores(scores: ROUGEScores) -> str:
    return (
        f"R1={scores.rouge1:.4f}  R2={scores.rouge2:.4f}  RL={scores.rougeL:.4f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config()

    dataset_name: str = cfg.get("dataset_name", "ccdv/arxiv-summarization")
    dataset_config: str = cfg.get("dataset_config", "document")
    input_field: str = cfg.get("dataset_input_field", "article")
    target_field: str = cfg.get("dataset_target_field", "abstract")

    output_path = Path(args.output) if args.output else (
        _RESULTS_DIR / "evaluation_results.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  PaperLens — Benchmark Evaluation")
    print(f"{'='*60}")
    print(f"  Dataset : {dataset_name} ({dataset_config})")
    print(f"  Split   : {args.split}")
    print(f"  Samples : {args.num_samples}")
    print(f"  LED     : {'SKIPPED' if args.tfidf_only else 'enabled'}")
    print(f"  TF-IDF  : enabled ({args.tfidf_sentences} sentences)")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print("Loading dataset…")
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=args.split,
    )
    n = min(args.num_samples, len(dataset))
    samples = dataset.select(range(n))
    print(f"  Loaded {n} examples from {args.split} split.\n")

    # ------------------------------------------------------------------
    # Instantiate models
    # ------------------------------------------------------------------
    tfidf_summarizer = TFIDFSummarizer(num_sentences=args.tfidf_sentences)
    led_summarizer: LEDSummarizer | None = None
    if not args.tfidf_only:
        print("Loading LED model (this may take a moment on first run)…")
        led_summarizer = LEDSummarizer()
        print("  LED model loaded.\n")

    evaluator = ROUGEEvaluator()

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    results: list[dict] = []
    led_scores_all: list[ROUGEScores] = []
    tfidf_scores_all: list[ROUGEScores] = []

    for idx, example in enumerate(samples):
        article: str = example[input_field]
        reference: str = example[target_field]

        print(f"[{idx + 1}/{n}] {truncate_for_display(article)}")

        row: dict = {
            "index": idx,
            "article_snippet": article[:200],
            "reference_snippet": reference[:200],
        }

        # TF-IDF
        t0 = time.perf_counter()
        tfidf_summary = tfidf_summarizer.summarize(article)
        tfidf_time = time.perf_counter() - t0
        tfidf_scores = evaluator.score(tfidf_summary, reference)
        tfidf_scores_all.append(tfidf_scores)
        row["tfidf"] = {
            "summary_snippet": tfidf_summary[:200],
            "scores": tfidf_scores.to_dict(),
            "inference_seconds": round(tfidf_time, 3),
        }
        print(f"  TF-IDF  : {format_scores(tfidf_scores)}  ({tfidf_time:.2f}s)")

        # LED
        if led_summarizer is not None:
            t0 = time.perf_counter()
            led_summary = led_summarizer.summarize(article)
            led_time = time.perf_counter() - t0
            led_scores = evaluator.score(led_summary, reference)
            led_scores_all.append(led_scores)
            row["led"] = {
                "summary_snippet": led_summary[:200],
                "scores": led_scores.to_dict(),
                "inference_seconds": round(led_time, 3),
            }
            print(f"  LED     : {format_scores(led_scores)}  ({led_time:.2f}s)")
        else:
            row["led"] = None

        results.append(row)
        print()

    # ------------------------------------------------------------------
    # Aggregate averages
    # ------------------------------------------------------------------
    print(f"{'='*60}")
    print("  Aggregate Results")
    print(f"{'='*60}")

    aggregate: dict = {"num_samples": n, "split": args.split}

    if tfidf_scores_all:
        avg_tfidf = evaluator.average(tfidf_scores_all)
        print(f"  TF-IDF avg : {format_scores(avg_tfidf)}")
        aggregate["tfidf_average"] = avg_tfidf.to_dict()

    if led_scores_all:
        avg_led = evaluator.average(led_scores_all)
        print(f"  LED avg    : {format_scores(avg_led)}")
        aggregate["led_average"] = avg_led.to_dict()

    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Write results
    # ------------------------------------------------------------------
    output_data = {
        "config": {
            "dataset": dataset_name,
            "dataset_config": dataset_config,
            "split": args.split,
            "num_samples": n,
            "tfidf_sentences": args.tfidf_sentences,
            "led_model": cfg.get("model_name", "allenai/led-large-16384-arxiv"),
            "tfidf_only": args.tfidf_only,
        },
        "aggregate": aggregate,
        "per_sample": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()
