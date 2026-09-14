# Tests

Unit and integration tests for the PaperLens pipeline.

**Planned test coverage:**
- `test_extractor.py` — PDF text extraction
- `test_text_cleaner.py` — preprocessing transformations
- `test_detector.py` — section detection and normalization
- `test_tfidf_baseline.py` — TF-IDF summarizer correctness
- `test_rouge_evaluator.py` — ROUGE metric computation
- `test_paper_comparator.py` — comparison report structure

Run tests with:
```bash
pytest tests/
```
