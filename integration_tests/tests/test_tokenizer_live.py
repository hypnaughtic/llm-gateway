"""Live integration tests for tokenizer accuracy.

These tests require:
- GEMINI_API_KEY for Gemini exact counts
- anthropic/tiktoken packages for Anthropic exact counts
"""

from __future__ import annotations

import pytest

from llm_gateway import count_tokens
from llm_gateway.tokenizer import build_tokenizer
from llm_gateway.tokenizers.anthropic_tokenizer import AnthropicTokenizer
from llm_gateway.tokenizers.gemini_tokenizer import GeminiTokenizer

SAMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "This sentence is used to test tokenization across multiple providers."
)


@pytest.mark.live
class TestAnthropicTokenizerLive:
    """Live tests for Anthropic tokenizer (local, no API key needed)."""

    def test_standalone_count_tokens(self) -> None:
        """count_tokens(text, provider='anthropic') returns consistent exact counts."""
        result = count_tokens(SAMPLE_TEXT, provider="anthropic")
        assert isinstance(result, int)
        assert result > 0
        # Same text should always give same result
        assert count_tokens(SAMPLE_TEXT, provider="anthropic") == result

    def test_build_tokenizer_returns_anthropic(self) -> None:
        """build_tokenizer('anthropic') returns an AnthropicTokenizer."""
        tok = build_tokenizer("anthropic")
        assert isinstance(tok, AnthropicTokenizer)
        assert tok.name == "anthropic"

    def test_empty_string(self) -> None:
        """Empty string returns 0."""
        assert count_tokens("", provider="anthropic") == 0


@pytest.mark.live
class TestGeminiTokenizerLive:
    """Live tests for Gemini tokenizer (requires GEMINI_API_KEY)."""

    def test_gemini_exact_count(self) -> None:
        """With GEMINI_API_KEY set, GeminiTokenizer returns exact positive count."""
        tok = GeminiTokenizer()
        result = tok.count_tokens(SAMPLE_TEXT)
        assert isinstance(result, int)
        assert result > 0

    def test_gemini_consistency(self) -> None:
        """Same text returns same count on repeated calls."""
        tok = GeminiTokenizer()
        first = tok.count_tokens(SAMPLE_TEXT)
        second = tok.count_tokens(SAMPLE_TEXT)
        assert first == second


@pytest.mark.live
class TestCrossProviderComparison:
    """Compare token counts across providers."""

    def test_cross_provider_order_of_magnitude(self) -> None:
        """Anthropic and Gemini counts should be in the same order of magnitude.

        Different vocabularies mean different exact counts, but they should be
        within 2x of each other for typical English text.
        """
        anthropic_count = count_tokens(SAMPLE_TEXT, provider="anthropic")
        gemini_tok = GeminiTokenizer()
        gemini_count = gemini_tok.count_tokens(SAMPLE_TEXT)

        assert anthropic_count > 0
        assert gemini_count > 0

        ratio = max(anthropic_count, gemini_count) / min(anthropic_count, gemini_count)
        assert ratio < 2.0, (
            f"Token counts differ by more than 2x: "
            f"anthropic={anthropic_count}, gemini={gemini_count}"
        )
