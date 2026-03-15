"""Tokenizer Protocol, registry, and standalone count_tokens() function."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from llm_gateway.tokenizers.heuristic_tokenizer import HeuristicTokenizer

logger = logging.getLogger(__name__)


@runtime_checkable
class Tokenizer(Protocol):
    """Protocol for token counting.

    Implementations should be local where possible. Some providers (e.g., Gemini)
    may require an API call for exact counts, with a heuristic fallback for offline use.
    """

    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text."""
        ...

    @property
    def name(self) -> str:
        """Human-readable tokenizer name."""
        ...


# Registry (factories) + cache (singleton instances)
_TOKENIZERS: dict[str, Callable[[], Tokenizer]] = {}
_TOKENIZER_CACHE: dict[str, Tokenizer] = {}
_tokenizer_builtins_registered = False


def register_tokenizer(provider: str, factory: Callable[[], Tokenizer]) -> None:
    """Register a tokenizer factory for a provider name."""
    _TOKENIZERS[provider] = factory


def build_tokenizer(provider: str) -> Tokenizer:
    """Build or return cached tokenizer for the given provider. Falls back to heuristic."""
    if provider in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[provider]

    _ensure_tokenizer_builtins_registered()
    factory = _TOKENIZERS.get(provider)
    if factory is None:
        logger.warning("No tokenizer for provider '%s', using heuristic", provider)
        tokenizer: Tokenizer = HeuristicTokenizer()
    else:
        try:
            tokenizer = factory()
        except Exception:
            logger.warning("Failed to build tokenizer for '%s', using heuristic", provider)
            tokenizer = HeuristicTokenizer()

    _TOKENIZER_CACHE[provider] = tokenizer
    return tokenizer


def count_tokens(text: str, provider: str = "anthropic") -> int:
    """Convenience function: count tokens for a provider without instantiating a client.

    Uses cached tokenizer instances — safe to call repeatedly without overhead.

    Usage:
        from llm_gateway import count_tokens
        n = count_tokens("Hello, world!", provider="anthropic")
    """
    tokenizer = build_tokenizer(provider)
    return tokenizer.count_tokens(text)


def _ensure_tokenizer_builtins_registered() -> None:
    """Lazily register built-in tokenizers on first use."""
    global _tokenizer_builtins_registered
    if _tokenizer_builtins_registered:
        return
    _tokenizer_builtins_registered = True

    # Anthropic (also used by local_claude)
    try:
        from llm_gateway.tokenizers.anthropic_tokenizer import AnthropicTokenizer

        factory: Callable[[], Tokenizer] = AnthropicTokenizer
        register_tokenizer("anthropic", factory)
        register_tokenizer("local_claude", factory)
    except ImportError:
        pass

    # Gemini
    try:
        from llm_gateway.tokenizers.gemini_tokenizer import GeminiTokenizer

        register_tokenizer("gemini", GeminiTokenizer)
    except ImportError:
        pass

    # Fake (always available)
    register_tokenizer("fake", HeuristicTokenizer)
