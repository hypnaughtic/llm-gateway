"""Tests for the tokenizer registry (build_tokenizer, count_tokens, register_tokenizer)."""

from __future__ import annotations

import pytest

from llm_gateway.tokenizer import (
    build_tokenizer,
    count_tokens,
    register_tokenizer,
)
from llm_gateway.tokenizers.anthropic_tokenizer import AnthropicTokenizer
from llm_gateway.tokenizers.heuristic_tokenizer import HeuristicTokenizer


@pytest.fixture(autouse=True)
def _clean_registry() -> None:  # type: ignore[misc]
    """Reset tokenizer registry state before each test."""
    import llm_gateway.tokenizer as mod

    saved_tokenizers = dict(mod._TOKENIZERS)
    saved_cache = dict(mod._TOKENIZER_CACHE)
    saved_registered = mod._tokenizer_builtins_registered

    yield

    mod._TOKENIZERS.clear()
    mod._TOKENIZERS.update(saved_tokenizers)
    mod._TOKENIZER_CACHE.clear()
    mod._TOKENIZER_CACHE.update(saved_cache)
    mod._tokenizer_builtins_registered = saved_registered


@pytest.mark.unit
class TestBuildTokenizer:
    """Tests for build_tokenizer()."""

    def test_anthropic_returns_anthropic_tokenizer(self) -> None:
        """build_tokenizer('anthropic') should return an AnthropicTokenizer."""
        tok = build_tokenizer("anthropic")
        assert isinstance(tok, AnthropicTokenizer)

    def test_fake_returns_heuristic_tokenizer(self) -> None:
        """build_tokenizer('fake') should return a HeuristicTokenizer."""
        tok = build_tokenizer("fake")
        assert isinstance(tok, HeuristicTokenizer)

    def test_unknown_provider_returns_heuristic(self) -> None:
        """Unknown provider should fall back to HeuristicTokenizer (no error)."""
        tok = build_tokenizer("unknown_provider_xyz")
        assert isinstance(tok, HeuristicTokenizer)

    def test_caching_returns_same_instance(self) -> None:
        """Calling build_tokenizer twice for same provider returns same instance."""
        tok1 = build_tokenizer("anthropic")
        tok2 = build_tokenizer("anthropic")
        assert tok1 is tok2

    def test_local_claude_returns_anthropic_tokenizer(self) -> None:
        """local_claude should use the same AnthropicTokenizer."""
        tok = build_tokenizer("local_claude")
        assert isinstance(tok, AnthropicTokenizer)


@pytest.mark.unit
class TestCountTokens:
    """Tests for the standalone count_tokens() function."""

    def test_anthropic_returns_positive(self) -> None:
        """count_tokens with anthropic provider returns positive int."""
        result = count_tokens("Hello, world!", provider="anthropic")
        assert isinstance(result, int)
        assert result > 0

    def test_default_provider_is_anthropic(self) -> None:
        """Default provider should be anthropic."""
        result = count_tokens("Hello, world!")
        assert result > 0

    def test_empty_string_returns_zero(self) -> None:
        """Empty string should return 0 for any provider."""
        assert count_tokens("", provider="anthropic") == 0
        assert count_tokens("", provider="fake") == 0

    def test_fake_provider_heuristic(self) -> None:
        """Fake provider should use heuristic (chars/4)."""
        # 40 chars / 4.0 = 10
        assert count_tokens("a" * 40, provider="fake") == 10


@pytest.mark.unit
class TestRegisterTokenizer:
    """Tests for register_tokenizer()."""

    def test_custom_factory(self) -> None:
        """register_tokenizer with a custom factory should work."""
        import llm_gateway.tokenizer as mod

        # Clear cache for our custom provider
        mod._TOKENIZER_CACHE.pop("custom_test", None)

        custom = HeuristicTokenizer(chars_per_token=2.0)
        register_tokenizer("custom_test", lambda: custom)

        tok = build_tokenizer("custom_test")
        assert tok is custom
        # 10 chars / 2.0 = 5
        assert tok.count_tokens("a" * 10) == 5

    def test_factory_failure_falls_back(self) -> None:
        """If factory raises, should fall back to HeuristicTokenizer."""
        import llm_gateway.tokenizer as mod

        mod._TOKENIZER_CACHE.pop("broken_test", None)

        def broken_factory() -> HeuristicTokenizer:
            raise RuntimeError("factory error")

        register_tokenizer("broken_test", broken_factory)
        tok = build_tokenizer("broken_test")
        assert isinstance(tok, HeuristicTokenizer)
