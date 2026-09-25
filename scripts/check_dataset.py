"""
scripts/check_dataset.py
Dataset loading sanity check — does NOT run any model inference.
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import datasets as ds_lib
import huggingface_hub
import evaluate as ev_lib

print("=" * 60)
print("  DATASET SANITY CHECK")
print("=" * 60)

# ── 1. Package versions ──────────────────────────────────────────
print(f"\n[versions]")
print(f"  datasets        : {ds_lib.__version__}")
print(f"  huggingface_hub : {huggingface_hub.__version__}")
print(f"  evaluate        : {ev_lib.__version__}")

# ── 2. Load dataset (no trust_remote_code) ───────────────────────
print(f"\n[loading] ccdv/arxiv-summarization  config=document  split=test ...")
from datasets import load_dataset
dataset = load_dataset("ccdv/arxiv-summarization", "document", split="test")

# ── 3. Basic checks ──────────────────────────────────────────────
print(f"\n[dataset]")
print(f"  Type            : {type(dataset).__name__}")
print(f"  Num examples    : {len(dataset)}")
print(f"  Field names     : {dataset.column_names}")

assert len(dataset) > 0, "Dataset is empty!"
assert "article" in dataset.column_names, "Missing 'article' field!"
assert "abstract" in dataset.column_names, "Missing 'abstract' field!"

# ── 4. Sample check ──────────────────────────────────────────────
ex = dataset[0]
article_snippet  = ex["article"].strip()[:100]
abstract_snippet = ex["abstract"].strip()[:100]

print(f"\n[sample check — index 0]")
print(f"  article non-empty  : {len(ex['article'].strip()) > 0}")
print(f"  abstract non-empty : {len(ex['abstract'].strip()) > 0}")
print(f"  article  (first 100 chars): {article_snippet}")
print(f"  abstract (first 100 chars): {abstract_snippet}")

print(f"\nRESULT: PASS — dataset loaded successfully")
