"""Tests for LLMClient.count_tokens() delegation."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from llm_gateway.client import LLMClient
from llm_gateway.testing import FakeLLMProvider


class _TestModel(BaseModel):
    answer: str


@pytest.mark.unit
class TestClientCountTokens:
    """Tests for LLMClient.count_tokens() method."""

    def test_delegates_to_provider(self) -> None:
        """Client should delegate count_tokens to the underlying provider."""
        fake = FakeLLMProvider()
        client = LLMClient(provider_instance=fake)

        result = client.count_tokens("Hello, world!")
        assert isinstance(result, int)
        assert result > 0

    def test_empty_string_returns_zero(self) -> None:
        """Empty string should return 0 tokens."""
        fake = FakeLLMProvider()
        client = LLMClient(provider_instance=fake)

        assert client.count_tokens("") == 0

    def test_nonempty_text_returns_positive(self) -> None:
        """Non-empty text should return a positive int."""
        fake = FakeLLMProvider()
        client = LLMClient(provider_instance=fake)

        result = client.count_tokens("The quick brown fox")
        assert result > 0

    def test_consistent_across_calls(self) -> None:
        """Same text should return the same count across multiple calls."""
        fake = FakeLLMProvider()
        client = LLMClient(provider_instance=fake)

        text = "This is a test sentence for token counting."
        first = client.count_tokens(text)
        second = client.count_tokens(text)
        assert first == second

    def test_heuristic_calculation(self) -> None:
        """FakeLLMProvider uses HeuristicTokenizer (chars/4)."""
        fake = FakeLLMProvider()
        client = LLMClient(provider_instance=fake)

        # 40 chars / 4.0 = 10 tokens
        text = "a" * 40
        assert client.count_tokens(text) == 10
