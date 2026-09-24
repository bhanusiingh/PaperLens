"""
src/summarization/led_summarizer.py

Transformer-based summarization using allenai/led-large-16384-arxiv.

LED (Longformer Encoder-Decoder) supports up to 16 384 tokens natively via
its sparse local + global attention mechanism.  The strategy here is:

  1. If the tokenised text fits within ``chunk_size`` tokens, pass it
     directly to LED — no truncation, no chunking.
  2. If it exceeds ``chunk_size``, split into overlapping chunks, summarise
     each chunk, concatenate the chunk summaries, and run one final
     consolidating summarisation pass.  No content is silently discarded.

Heavy imports (torch, transformers) are deferred to first use so that
importing this module does not require PyTorch to be installed in environments
that only use TFIDFSummarizer (e.g., the test suite without GPU).
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # These imports are only for type checkers; at runtime they are deferred.
    import torch
    from transformers import AutoTokenizer, LEDForConditionalGeneration


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def _load_config(config_path: Path = _DEFAULT_CONFIG) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Hard model limit (tokens).
LED_MAX_INPUT: int = 16_384

# Overlap between consecutive chunks (tokens).
CHUNK_OVERLAP: int = 128


class LEDSummarizer:
    """
    Wraps allenai/led-large-16384-arxiv for abstractive summarization of
    long research paper text.

    Long-document strategy
    ----------------------
    - Texts that fit within ``chunk_size`` tokens are passed directly to the
      model.  LED's sparse attention handles up to 16 384 tokens, so most
      individual paper sections and even full short papers need no chunking.
    - Texts that exceed ``chunk_size`` are split into overlapping chunks of
      ``chunk_size`` tokens with ``CHUNK_OVERLAP`` tokens of overlap.  Each
      chunk is summarised independently.  The chunk summaries are then
      concatenated and, if small enough, fed back for a final consolidation
      pass.

    Usage
    -----
        summarizer = LEDSummarizer()
        summary = summarizer.summarize(paper_text)
        n_tokens = summarizer.count_tokens(paper_text)
    """

    MODEL_NAME: str = "allenai/led-large-16384-arxiv"

    def __init__(
        self,
        model_name: str | None = None,
        chunk_size: int | None = None,
        min_output_tokens: int = 64,
        max_output_tokens: int | None = None,
        device: str | None = None,
    ):
        """
        Args:
            model_name: HuggingFace model identifier.
            chunk_size: Number of input tokens per chunk.  Defaults to
                ``led_max_input_tokens`` from config (capped at LED_MAX_INPUT).
            min_output_tokens: Minimum generated summary length (tokens).
            max_output_tokens: Maximum generated summary length (tokens).
            device: ``'cuda'``, ``'cpu'``, or ``None`` (auto-detect).
        """
        cfg = _load_config()
        self.model_name = model_name or cfg.get("model_name", self.MODEL_NAME)

        raw_chunk = chunk_size or cfg.get("led_max_input_tokens", LED_MAX_INPUT)
        self.chunk_size: int = min(int(raw_chunk), LED_MAX_INPUT)

        self.max_output_tokens: int = max_output_tokens or int(
            cfg.get("max_summary_tokens", 512)
        )
        self.min_output_tokens: int = min_output_tokens

        # device resolution is deferred until first model load
        self._device_arg: str | None = device

        self._tokenizer = None
        self._model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarize(self, text: str) -> str:
        """
        Generate an abstractive summary for ``text``.

        Automatically chunks the input if it exceeds the model context window.

        Args:
            text: Cleaned paper text (full paper or a specific section).

        Returns:
            Generated summary as a plain string.
        """
        tokenizer, _ = self._load_model()
        
        # Suppress the max_length warning since we handle chunking manually
        old_max_len = tokenizer.model_max_length
        tokenizer.model_max_length = int(1e9)
        token_ids: list[int] = tokenizer.encode(text, add_special_tokens=False)
        tokenizer.model_max_length = old_max_len
        
        n_tokens: int = len(token_ids)

        if n_tokens <= self.chunk_size:
            return self._summarize_tokens(token_ids)

        return self._hierarchical_summarize(token_ids)

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in ``text`` (without special tokens)."""
        tokenizer, _ = self._load_model()
        return len(tokenizer.encode(text, add_special_tokens=False))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        """
        Lazy-load tokenizer and model on first call.

        torch and transformers are imported here so that this module can be
        imported in environments without PyTorch installed, as long as
        _load_model() is never called.
        """
        if self._tokenizer is None or self._model is None:
            import torch  # deferred import
            from transformers import AutoTokenizer, LEDForConditionalGeneration

            self._device: str = self._device_arg or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # FP16 on CUDA to halve VRAM usage (~3 GB vs ~6 GB); CPU must stay FP32.
            torch_dtype = torch.float16 if self._device == "cuda" else torch.float32
            self._model = LEDForConditionalGeneration.from_pretrained(
                self.model_name, torch_dtype=torch_dtype
            ).to(self._device)
            self._model.eval()
        return self._tokenizer, self._model

    def _summarize_tokens(self, token_ids: list[int]) -> str:
        """Summarise a token ID list that fits within ``chunk_size``."""
        import torch  # deferred import

        tokenizer, model = self._load_model()

        text_chunk: str = tokenizer.decode(token_ids, skip_special_tokens=True)

        encoded = tokenizer(
            text_chunk,
            return_tensors="pt",
            max_length=self.chunk_size,
            truncation=True,
            # No padding=max_length: dynamically pad to sequence length only.
            # padding=max_length on a 16K-context window bloats the KV-cache
            # and causes CUDA OOM / offloading on consumer GPUs.
        )
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded["attention_mask"].to(self._device)

        # LED global attention: first (BOS) token attends to all others.
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                min_length=self.min_output_tokens,
                max_new_tokens=self.max_output_tokens,
                num_beams=1,          # Greedy decoding: no beam KV-cache expansion.
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def _hierarchical_summarize(self, token_ids: list[int]) -> str:
        """
        Chunk-then-consolidate summarisation for texts exceeding the context
        window.
        """
        chunks: list[list[int]] = self._split_token_chunks(token_ids)
        chunk_summaries: list[str] = []

        for chunk in chunks:
            chunk_summaries.append(self._summarize_tokens(chunk))

        combined: str = " ".join(chunk_summaries)

        tokenizer, _ = self._load_model()
        combined_ids = tokenizer.encode(combined, add_special_tokens=False)
        if len(combined_ids) <= self.chunk_size:
            return self._summarize_tokens(combined_ids)

        # Combined summary too long — return directly.
        return combined

    def _split_token_chunks(self, token_ids: list[int]) -> list[list[int]]:
        """Split ``token_ids`` into overlapping chunks."""
        step: int = max(1, self.chunk_size - CHUNK_OVERLAP)
        chunks: list[list[int]] = []
        start: int = 0
        while start < len(token_ids):
            end: int = start + self.chunk_size
            chunks.append(token_ids[start:end])
            if end >= len(token_ids):
                break
            start += step
        return chunks
