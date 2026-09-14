# src/summarization/__init__.py
# Exports are imported lazily so that TFIDFSummarizer can be used without
# requiring the transformers/torch stack (e.g. in tests that don't touch LED).

def __getattr__(name):
    if name == "LEDSummarizer":
        from .led_summarizer import LEDSummarizer
        return LEDSummarizer
    if name == "TFIDFSummarizer":
        from .tfidf_baseline import TFIDFSummarizer
        return TFIDFSummarizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["LEDSummarizer", "TFIDFSummarizer"]
