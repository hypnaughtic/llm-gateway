"""Tests for AnthropicTokenizer (uses tiktoken for local BPE tokenization)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from llm_gateway.tokenizers.anthropic_tokenizer import AnthropicTokenizer


@pytest.mark.unit
class TestAnthropicTokenizer:
    """Tests for the Anthropic tokenizer using tiktoken."""

    def test_empty_string_returns_zero(self) -> None:
        """Empty string should return 0 tokens."""
        tok = AnthropicTokenizer()
        assert tok.count_tokens("") == 0

    def test_nonempty_text_returns_positive(self) -> None:
        """Non-empty text should return a positive int."""
        tok = AnthropicTokenizer()
        result = tok.count_tokens("Hello, world!")
        assert isinstance(result, int)
        assert result > 0

    def test_consistent_count(self) -> None:
        """Same text should return the same count on repeated calls."""
        tok = AnthropicTokenizer()
        text = "The quick brown fox jumps over the lazy dog."
        first = tok.count_tokens(text)
        second = tok.count_tokens(text)
        assert first == second

    def test_longer_text_more_tokens(self) -> None:
        """Longer text should produce more tokens."""
        tok = AnthropicTokenizer()
        short = tok.count_tokens("Hello")
        long = tok.count_tokens("Hello, this is a much longer piece of text with many words.")
        assert long > short

    def test_lazy_initialization(self) -> None:
        """Encoding should not be initialized until first count_tokens call."""
        tok = AnthropicTokenizer()
        assert tok._initialized is False
        tok.count_tokens("test")
        assert tok._initialized is True

    def test_encoding_reused_across_calls(self) -> None:
        """Encoding should be reused (not recreated) on subsequent calls."""
        tok = AnthropicTokenizer()
        tok.count_tokens("first")
        encoding_ref = tok._encoding
        tok.count_tokens("second")
        assert tok._encoding is encoding_ref

    def test_name_property(self) -> None:
        """Name property should return 'anthropic'."""
        tok = AnthropicTokenizer()
        assert tok.name == "anthropic"

    def test_single_token_text(self) -> None:
        """A single common word should be 1-2 tokens."""
        tok = AnthropicTokenizer()
        result = tok.count_tokens("hello")
        assert 1 <= result <= 2

    def test_whitespace_only(self) -> None:
        """Whitespace-only text should return positive count (not empty)."""
        tok = AnthropicTokenizer()
        result = tok.count_tokens("   ")
        assert result > 0


@pytest.mark.unit
class TestAnthropicTokenizerFallback:
    """Tests for graceful fallback when tiktoken is unavailable."""

    def test_import_error_falls_back_to_heuristic(self) -> None:
        """When tiktoken import fails, should fall back to heuristic."""
        tok = AnthropicTokenizer()

        mock_tiktoken = MagicMock()
        mock_tiktoken.get_encoding.side_effect = ImportError("no tiktoken")
        with patch.dict(sys.modules, {"tiktoken": mock_tiktoken}):
            tok._init_encoding()
            assert tok._use_heuristic is True

    def test_heuristic_calculation(self) -> None:
        """Heuristic should match max(1, int(len(text) / 4.0))."""
        tok = AnthropicTokenizer()
        tok._initialized = True
        tok._use_heuristic = True

        text = "a" * 40  # 40 / 4.0 = 10
        assert tok.count_tokens(text) == 10

    def test_heuristic_short_text_at_least_one(self) -> None:
        """Short text should return at least 1 token via heuristic."""
        tok = AnthropicTokenizer()
        tok._initialized = True
        tok._use_heuristic = True

        assert tok.count_tokens("Hi") >= 1

    def test_encode_exception_falls_back(self) -> None:
        """When tiktoken.encode raises, should fall back to heuristic."""
        tok = AnthropicTokenizer()
        tok._initialized = True
        tok._use_heuristic = False

        mock_encoding = MagicMock()
        mock_encoding.encode.side_effect = RuntimeError("encode error")
        tok._encoding = mock_encoding

        mock_tiktoken = MagicMock()
        mock_tiktoken.Encoding = type(mock_encoding)
        with patch.dict(sys.modules, {"tiktoken": mock_tiktoken}):
            result = tok.count_tokens("a" * 40)
            assert result == 10  # Falls back to 40 / 4.0 = 10

    def test_init_encoding_idempotent(self) -> None:
        """Calling _init_encoding twice should not re-initialize."""
        tok = AnthropicTokenizer()
        tok._initialized = True
        tok._use_heuristic = True
        tok._init_encoding()  # Should be a no-op
        assert tok._use_heuristic is True
