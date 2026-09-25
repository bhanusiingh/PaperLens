"""scripts/audit_benchmark.py — Read-only audit of results/benchmark_20.json"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
d = json.load(open(ROOT / "results" / "benchmark_20.json"))

samples = d["per_sample"]
agg = d["aggregate"]

# ── 1. Sample count ──────────────────────────────────────────────
print(f"[1] Total samples in JSON: {len(samples)}")

# ── 2+3. Per-sample validity ─────────────────────────────────────
issues = []
for i, s in enumerate(samples):
    for method in ("tfidf", "led"):
        entry = s.get(method)
        if entry is None:
            issues.append(f"  sample[{i}]: {method} is None")
        else:
            sc = entry.get("scores", {})
            for k in ("rouge1", "rouge2", "rougeL"):
                if k not in sc:
                    issues.append(f"  sample[{i}]: {method} missing {k}")
    if not s.get("reference_snippet"):
        issues.append(f"  sample[{i}]: reference_snippet empty")
    if not s.get("article_snippet"):
        issues.append(f"  sample[{i}]: article_snippet empty")

all_present = "YES" if not issues else "NO"
print(f"[2+3] All TF-IDF/LED summaries and ROUGE scores present: {all_present}")
for iss in issues:
    print(iss)

# ── 4. Reference confirmation ────────────────────────────────────
print()
print("[4] Reference snippets for samples 0, 9, 19 (confirming human abstract):")
for i in (0, 9, 19):
    snippet = samples[i]["reference_snippet"][:120]
    print(f"  [{i:2d}] {snippet}")

# ── 5. Aggregate recalculation ───────────────────────────────────
tf_r1  = [s["tfidf"]["scores"]["rouge1"] for s in samples]
tf_r2  = [s["tfidf"]["scores"]["rouge2"] for s in samples]
tf_rl  = [s["tfidf"]["scores"]["rougeL"] for s in samples]
led_r1 = [s["led"]["scores"]["rouge1"]   for s in samples]
led_r2 = [s["led"]["scores"]["rouge2"]   for s in samples]
led_rl = [s["led"]["scores"]["rougeL"]   for s in samples]
n = len(samples)

calc_tf  = {k: round(v/n, 4) for k, v in zip(
    ("rouge1","rouge2","rougeL"), (sum(tf_r1), sum(tf_r2), sum(tf_rl)))}
calc_led = {k: round(v/n, 4) for k, v in zip(
    ("rouge1","rouge2","rougeL"), (sum(led_r1), sum(led_r2), sum(led_rl)))}

stored_tf  = agg.get("tfidf_average", {})
stored_led = agg.get("led_average",   {})

print()
print("[5] Aggregate recalculation:")
print(f"  TF-IDF  calc : {calc_tf}")
print(f"  TF-IDF  stored: {stored_tf}")
print(f"  LED     calc : {calc_led}")
print(f"  LED     stored: {stored_led}")

tf_ok  = all(abs(calc_tf[k]  - stored_tf.get(k, 0))  < 0.0002 for k in calc_tf)
led_ok = all(abs(calc_led[k] - stored_led.get(k, 0)) < 0.0002 for k in calc_led)
print(f"  TF-IDF aggregate matches: {tf_ok}")
print(f"  LED    aggregate matches: {led_ok}")

# ── 6. Sequential indices, no gaps/duplicates ────────────────────
indices = [s["index"] for s in samples]
print()
print(f"[6] Sample indices: {indices}")
print(f"    Sequential 0..19 : {indices == list(range(20))}")
print(f"    No duplicates    : {len(set(indices)) == len(indices)}")

# ── 7. Hierarchical path check ───────────────────────────────────
print()
print("[7] Hierarchical path (>16384 tokens) — checking from benchmark stdout records:")
print("    Token counts observed per paper (from benchmark stdout):")
token_counts = [
    7567, 12047, 3429, 4973, 6545, 4868, 3870, 6644,
    6658, 11882, 4122, 8404, 7383, 4786, 1343, 7389,
    6913, 16293, 1238, 249
]
over_16k = [i for i, t in enumerate(token_counts) if t > 16384]
print(f"    {list(zip(range(20), token_counts))}")
print(f"    Papers exceeding 16384 tokens: {over_16k}")
print(f"    Hierarchical path exercised in this run: {'YES' if over_16k else 'NO'}")
print(f"    NOTE: Paper[17] had 16,293 tokens — just UNDER the limit. Single-pass inference.")

# ── 8. Sampling order ────────────────────────────────────────────
print()
print("[8] Sampling: dataset.select(range(20)) on the 'test' split.")
print("    Deterministic, reproducible — always first 20 examples of the test split.")
print("    No shuffling is applied.")

# ── 9. Uncommitted changes ───────────────────────────────────────
print()
print("[9] Uncommitted source-code changes: see 'git status' output below.")
print("    Expected: only early_stopping removal in led_summarizer.py,")
print("    plus trust_remote_code removal in experiments/evaluate.py,")
print("    plus the new results/sanity_check.json and results/benchmark_20.json files.")

# ── Config dump ──────────────────────────────────────────────────
print()
print("[config stored in JSON]:")
print(json.dumps(d["config"], indent=4))
