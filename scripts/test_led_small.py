"""
scripts/test_led_small.py

Small controlled LED smoke test. Verifies:
- CUDA available + GPU name
- FP16 model loading
- correct device placement
- short input token count
- inference runtime
- non-empty, non-NaN output
- VRAM usage before/after inference
"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

print("=" * 60)
print("  LED SMOKE TEST")
print("=" * 60)
print(f"[device] CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[device] GPU name       : {torch.cuda.get_device_name(0)}")
    total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print(f"[device] VRAM total     : {total_mb:.0f} MB")

from src.summarization.led_summarizer import LEDSummarizer

SHORT_TEXT = (
    "We introduce BERT, a new language representation model which stands for "
    "Bidirectional Encoder Representations from Transformers. Unlike recent "
    "language representation models, BERT is designed to pre-train deep "
    "bidirectional representations from unlabeled text by jointly conditioning "
    "on both left and right context in all layers. As a result, the pre-trained "
    "BERT model can be fine-tuned with just one additional output layer to create "
    "state-of-the-art models for a wide range of tasks, such as question answering "
    "and language inference, without substantial task-specific architecture "
    "modifications. BERT is conceptually simple and empirically powerful. "
    "It obtains new state-of-the-art results on eleven natural language processing "
    "tasks, including pushing the GLUE score to 80.5 percent, MultiNLI accuracy "
    "to 86.7 percent, SQuAD v1.1 Test F1 to 93.2 and SQuAD v2.0 Test F1 to 83.1."
)

led = LEDSummarizer()

# -- Load model --
t0 = time.time()
tokenizer, model = led._load_model()
load_time = time.time() - t0

print(f"\n[load] Time             : {load_time:.2f}s")
print(f"[load] Device           : {led._device}")
print(f"[load] Model dtype      : {next(model.parameters()).dtype}")

if torch.cuda.is_available():
    vram_after_load = torch.cuda.memory_allocated(0) / 1024**2
    print(f"[vram] After model load : {vram_after_load:.0f} MB")

# -- Token count --
n_tokens = led.count_tokens(SHORT_TEXT)
print(f"\n[tokens] Input tokens   : {n_tokens}")
print(f"[tokens] Chunk size     : {led.chunk_size}")
print(f"[tokens] Will chunk?    : {n_tokens > led.chunk_size}")

# -- Inference --
print("\n[inference] Running...")
t_inf = time.time()
summary = led.summarize(SHORT_TEXT)
inf_time = time.time() - t_inf

if torch.cuda.is_available():
    vram_after_inf = torch.cuda.memory_allocated(0) / 1024**2
    print(f"[vram] After inference  : {vram_after_inf:.0f} MB")

print(f"[inference] Time        : {inf_time:.2f}s")
print(f"[inference] Output len  : {len(summary)} chars")
print(f"[inference] Non-empty   : {len(summary.strip()) > 0}")
print(f"[inference] Contains NaN: {'nan' in summary.lower()}")

print(f"\n--- SUMMARY OUTPUT ---")
print(summary)
print("----------------------")

passed = len(summary.strip()) > 0 and "nan" not in summary.lower()
print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
