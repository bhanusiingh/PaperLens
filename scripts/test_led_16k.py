"""
scripts/test_led_16k.py

Tests the >16K hierarchical chunking path of LEDSummarizer.
Uses the BERT paper (1810.04805) which previously produced 18,265 tokens.
Reports per-chunk validity, consolidation, timing, and VRAM.
"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

print("=" * 60)
print("  LED >16K HIERARCHICAL CHUNKING TEST")
print("=" * 60)

from src.pdf.extractor import PDFExtractor
from src.preprocessing.text_cleaner import TextCleaner
from src.summarization.led_summarizer import LEDSummarizer

PDF_PATH = "data/samples/1810.04805.pdf"

# ── 1. Extract + clean ──────────────────────────────────────────
print("\n[1] Extracting text from BERT paper...")
raw = PDFExtractor(PDF_PATH).extract_text()
clean = TextCleaner().clean(raw)
print(f"    Cleaned text length: {len(clean):,} chars")

# ── 2. Load model ───────────────────────────────────────────────
print("\n[2] Loading LED model...")
led = LEDSummarizer()
t_load_start = time.time()
tokenizer, model = led._load_model()
load_time = time.time() - t_load_start

print(f"    Load time  : {load_time:.2f}s")
print(f"    Device     : {led._device}")
print(f"    Dtype      : {next(model.parameters()).dtype}")
if torch.cuda.is_available():
    print(f"    VRAM used  : {torch.cuda.memory_allocated(0)/1024**2:.0f} MB")

# ── 3. Token count + chunk plan ─────────────────────────────────
print("\n[3] Tokenising full paper...")
old_max = tokenizer.model_max_length
tokenizer.model_max_length = int(1e9)
token_ids = tokenizer.encode(clean, add_special_tokens=False)
tokenizer.model_max_length = old_max

n_tokens = len(token_ids)
chunks = led._split_token_chunks(token_ids)

print(f"    Total input tokens : {n_tokens:,}")
print(f"    Chunk size limit   : {led.chunk_size:,}")
print(f"    Chunks created     : {len(chunks)}")
for i, ch in enumerate(chunks):
    print(f"      chunk[{i}]: {len(ch):,} tokens")

assert n_tokens > led.chunk_size, "Paper did not exceed chunk size — wrong file?"

# ── 4. Per-chunk inference ───────────────────────────────────────
print("\n[4] Running inference on each chunk...")
chunk_summaries = []
chunk_times = []

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats(0)

for i, chunk in enumerate(chunks):
    t_c = time.time()
    summary = led._summarize_tokens(chunk)
    elapsed = time.time() - t_c
    chunk_times.append(elapsed)

    non_empty = len(summary.strip()) > 0
    has_nan = "nan" in summary.lower()

    print(f"    chunk[{i}]: {elapsed:.1f}s | len={len(summary)} | "
          f"non-empty={non_empty} | nan={has_nan}")
    if not non_empty:
        print(f"    *** CHUNK {i} PRODUCED EMPTY OUTPUT — STOPPING ***")
        sys.exit(1)
    if has_nan:
        print(f"    *** CHUNK {i} PRODUCED NaN OUTPUT — STOPPING ***")
        sys.exit(1)

    chunk_summaries.append(summary)

if torch.cuda.is_available():
    peak_vram = torch.cuda.max_memory_allocated(0) / 1024**2
    curr_vram = torch.cuda.memory_allocated(0) / 1024**2
    print(f"\n    VRAM peak (chunks): {peak_vram:.0f} MB")
    print(f"    VRAM current      : {curr_vram:.0f} MB")

# ── 5. Consolidation pass ────────────────────────────────────────
print("\n[5] Consolidation pass...")
combined = " ".join(chunk_summaries)
combined_ids = tokenizer.encode(combined, add_special_tokens=False)
print(f"    Combined chunk summaries: {len(combined)} chars")
print(f"    Consolidation tokens    : {len(combined_ids):,}")

t_con = time.time()
if len(combined_ids) <= led.chunk_size:
    final_summary = led._summarize_tokens(combined_ids)
    con_time = time.time() - t_con
    print(f"    Consolidation time      : {con_time:.1f}s")
    print(f"    Consolidation non-empty : {len(final_summary.strip()) > 0}")
    print(f"    Consolidation nan       : {'nan' in final_summary.lower()}")
else:
    final_summary = combined
    con_time = 0
    print("    Consolidation skipped (combined too long — returned raw concat)")

# ── 6. Final report ──────────────────────────────────────────────
total_inf_time = sum(chunk_times) + con_time

if torch.cuda.is_available():
    peak_vram = torch.cuda.max_memory_allocated(0) / 1024**2
    print(f"\n    VRAM peak (total)       : {peak_vram:.0f} MB")

print("\n" + "=" * 60)
print("  FINAL RESULT")
print("=" * 60)
print(f"Total input tokens    : {n_tokens:,}")
print(f"Chunks processed      : {len(chunks)}")
print(f"Total inference time  : {total_inf_time:.1f}s")
print(f"Final summary length  : {len(final_summary)} chars")
print(f"Final summary non-empty: {len(final_summary.strip()) > 0}")
print(f"Final summary has NaN : {'nan' in final_summary.lower()}")
print("\n--- FIRST 300 CHARS OF FINAL SUMMARY ---")
print(final_summary[:300])
print("----------------------------------------")

passed = len(final_summary.strip()) > 0 and "nan" not in final_summary.lower()
print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
