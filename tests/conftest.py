"""Shared test fixtures for llm-gateway."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

import pytest

from llm_gateway.config import GatewayConfig
from llm_gateway.cost import build_token_usage
from llm_gateway.types import LLMMessage, LLMResponse

T = TypeVar("T")


class FakeLLMProvider:
    """In-memory fake provider for testing.

    Pre-configure responses with `set_response()`, then pass as
    `LLMClient(provider_instance=fake)`.
    """

    def __init__(self) -> None:
        self._responses: dict[type, object] = {}
        self._call_count: int = 0
        self._last_messages: Sequence[LLMMessage] = []
        self._last_model: str = ""

    def set_response(self, response_model: type[T], response: T) -> None:
        """Pre-configure a response for a given model type."""
        self._responses[response_model] = response

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        response_model: type[T],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        image_files: Sequence[str] | None = None,
    ) -> LLMResponse[T]:
        """Return pre-configured response."""
        self._call_count += 1
        self._last_messages = messages
        self._last_model = model

        content = self._responses.get(response_model)
        if content is None:
            msg = f"No fake response configured for {response_model.__name__}"
            raise ValueError(msg)

        usage = build_token_usage(model, 100, 50)
        return LLMResponse(
            content=content,  # type: ignore[arg-type]
            usage=usage,
            model=model,
            provider="fake",
            latency_ms=1.0,
        )

    async def close(self) -> None:
        """No-op."""


@pytest.fixture
def fake_provider() -> FakeLLMProvider:
    """Return a fresh FakeLLMProvider."""
    return FakeLLMProvider()


@pytest.fixture
def test_config(monkeypatch: pytest.MonkeyPatch) -> GatewayConfig:
    """Return a GatewayConfig with test defaults (no real API key needed)."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_API_KEY", "test-key-fake")
    monkeypatch.setenv("LLM_TRACE_ENABLED", "false")
    return GatewayConfig()
