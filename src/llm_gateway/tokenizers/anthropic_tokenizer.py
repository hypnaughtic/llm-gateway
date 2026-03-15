"""Token counter for Anthropic/Claude models using tiktoken BPE tokenizer."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class AnthropicTokenizer:
    """Token counter for Claude models using tiktoken's BPE tokenizer.

    Uses the ``cl100k_base`` encoding (closest match for Claude tokenization).
    Fully local — no API calls, no API key required.
    Falls back to calibrated heuristic (~4 chars/token) if tiktoken is unavailable.

    Works for all Claude models (Haiku, Sonnet, Opus).
    """

    HEURISTIC_CHARS_PER_TOKEN = 4.0

    def __init__(self) -> None:
        self._encoding: Any = None  # tiktoken.Encoding when available
        self._use_heuristic = False
        self._initialized = False

    def _init_encoding(self) -> None:
        """Lazy-init: try tiktoken, fall back to heuristic."""
        if self._initialized:
            return
        self._initialized = True

        try:
            import tiktoken

            self._encoding = tiktoken.get_encoding("cl100k_base")
            logger.debug("Anthropic tokenizer initialized via tiktoken (cl100k_base)")
        except (ImportError, Exception) as exc:
            self._use_heuristic = True
            logger.debug(
                "tiktoken not available — using heuristic tokenizer (chars / %.1f): %s",
                self.HEURISTIC_CHARS_PER_TOKEN,
                exc,
            )

    def count_tokens(self, text: str) -> int:
        """Count tokens using Claude's tokenizer (tiktoken with heuristic fallback)."""
        if not text:
            return 0
        self._init_encoding()

        if self._use_heuristic:
            return max(1, int(len(text) / self.HEURISTIC_CHARS_PER_TOKEN))

        try:
            tokens: list[int] = self._encoding.encode(text)
            return len(tokens)
        except Exception as exc:
            logger.warning("tiktoken encode failed, falling back to heuristic: %s", exc)
            return max(1, int(len(text) / self.HEURISTIC_CHARS_PER_TOKEN))

    @property
    def name(self) -> str:
        """Human-readable tokenizer name."""
        return "anthropic"
