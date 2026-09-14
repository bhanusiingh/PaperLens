# Experiments

This directory contains experiment scripts and logs.

Once evaluation runs are executed, results (ROUGE scores, qualitative notes,
comparison tables) should be stored here, organized by experiment name or date.

**Example structure:**

```
experiments/
├── run_01_led_baseline/
│   ├── config_snapshot.yaml
│   └── rouge_scores.json
└── run_02_tfidf_baseline/
    └── rouge_scores.json
```
