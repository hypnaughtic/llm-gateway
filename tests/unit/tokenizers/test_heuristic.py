"""Tests for HeuristicTokenizer."""

from __future__ import annotations

import pytest

from llm_gateway.tokenizers.heuristic_tokenizer import HeuristicTokenizer


@pytest.mark.unit
class TestHeuristicTokenizer:
    """Tests for the chars-per-token heuristic tokenizer."""

    def test_empty_string_returns_zero(self) -> None:
        """Empty string should return 0 tokens."""
        tok = HeuristicTokenizer()
        assert tok.count_tokens("") == 0

    def test_short_text_returns_at_least_one(self) -> None:
        """Short text (fewer chars than ratio) should return at least 1."""
        tok = HeuristicTokenizer()
        assert tok.count_tokens("Hi") >= 1

    def test_known_text_default_ratio(self) -> None:
        """Known text with default ratio of 4.0 chars/token."""
        tok = HeuristicTokenizer()
        # 20 chars / 4.0 = 5 tokens
        text = "a" * 20
        assert tok.count_tokens(text) == 5

    def test_known_text_exact_calculation(self) -> None:
        """Verify int(len/ratio) calculation for non-round numbers."""
        tok = HeuristicTokenizer()
        # 13 chars / 4.0 = 3.25 → int(3.25) = 3
        text = "a" * 13
        assert tok.count_tokens(text) == 3

    def test_custom_chars_per_token_ratio(self) -> None:
        """Custom ratio should change the calculation."""
        tok = HeuristicTokenizer(chars_per_token=2.0)
        # 10 chars / 2.0 = 5 tokens
        text = "a" * 10
        assert tok.count_tokens(text) == 5

    def test_gemini_calibrated_ratio(self) -> None:
        """Gemini-calibrated ratio of 3.5 chars/token."""
        tok = HeuristicTokenizer(chars_per_token=3.5)
        # 14 chars / 3.5 = 4 tokens
        text = "a" * 14
        assert tok.count_tokens(text) == 4

    def test_name_property_default(self) -> None:
        """Name includes the default ratio."""
        tok = HeuristicTokenizer()
        assert tok.name == "heuristic-4.0"

    def test_name_property_custom(self) -> None:
        """Name includes the custom ratio."""
        tok = HeuristicTokenizer(chars_per_token=3.5)
        assert tok.name == "heuristic-3.5"

    def test_single_char_returns_one(self) -> None:
        """Single character should return at least 1 token."""
        tok = HeuristicTokenizer()
        assert tok.count_tokens("x") == 1

    def test_long_text(self) -> None:
        """Long text should scale linearly."""
        tok = HeuristicTokenizer()
        text = "a" * 4000
        assert tok.count_tokens(text) == 1000
