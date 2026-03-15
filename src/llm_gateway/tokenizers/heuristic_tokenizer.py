"""Fallback tokenizer using chars-per-token heuristic."""

from __future__ import annotations


class HeuristicTokenizer:
    """Fallback tokenizer using chars-per-token heuristic.

    Default ratio of 4.0 chars/token is calibrated for English markdown.
    Accuracy: ~75-80% for typical LLM content.
    """

    def __init__(self, chars_per_token: float = 4.0) -> None:
        self._chars_per_token = chars_per_token

    def count_tokens(self, text: str) -> int:
        """Estimate token count from character count."""
        if not text:
            return 0
        return max(1, int(len(text) / self._chars_per_token))

    @property
    def name(self) -> str:
        """Human-readable tokenizer name."""
        return f"heuristic-{self._chars_per_token}"
