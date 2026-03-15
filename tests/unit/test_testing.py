"""Tests for llm_gateway.testing — shipped FakeLLMProvider."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import BaseModel

from llm_gateway.config import GatewayConfig
from llm_gateway.exceptions import ResponseValidationError
from llm_gateway.registry import build_provider, list_providers
from llm_gateway.testing import FakeCall, FakeLLMProvider
from llm_gateway.types import LLMMessage, LLMResponse

# ── Test models ──────────────────────────────────────────────────


class Greeting(BaseModel):
    """Simple model for testing."""

    text: str


class MathAnswer(BaseModel):
    """Another model for multi-model tests."""

    value: int


# ── FakeLLMProvider ──────────────────────────────────────────────


@pytest.mark.unit
class TestFakeLLMProvider:
    """Tests for FakeLLMProvider."""

    async def test_set_response_returns_preconfigured(self) -> None:
        """set_response() pre-configures response; complete() returns it."""
        fake = FakeLLMProvider()
        expected = Greeting(text="hello")
        fake.set_response(Greeting, expected)

        resp = await fake.complete(
            messages=[{"role": "user", "content": "hi"}],
            response_model=Greeting,
            model="test-model",
        )

        assert resp.content == expected

    async def test_response_factory_called(self) -> None:
        """response_factory receives (response_model, messages) and returns T."""
        captured: list[tuple[type, Sequence[LLMMessage]]] = []

        def factory(
            model_cls: type[Greeting],
            messages: Sequence[LLMMessage],
        ) -> Greeting:
            captured.append((model_cls, messages))
            return Greeting(text="from factory")

        fake = FakeLLMProvider(response_factory=factory)
        msgs: list[LLMMessage] = [{"role": "user", "content": "hi"}]
        resp = await fake.complete(
            messages=msgs,
            response_model=Greeting,
            model="test-model",
        )

        assert resp.content.text == "from factory"
        assert len(captured) == 1
        assert captured[0][0] is Greeting

    async def test_set_response_takes_precedence_over_factory(self) -> None:
        """Pre-configured response wins over factory when both set."""
        preconfigured = Greeting(text="preconfigured")

        def factory(
            model_cls: type[Greeting],
            messages: Sequence[LLMMessage],
        ) -> Greeting:
            return Greeting(text="factory")

        fake = FakeLLMProvider(response_factory=factory)
        fake.set_response(Greeting, preconfigured)

        resp = await fake.complete(
            messages=[{"role": "user", "content": "hi"}],
            response_model=Greeting,
            model="test-model",
        )

        assert resp.content.text == "preconfigured"

    async def test_returns_proper_llm_response(self) -> None:
        """complete() returns LLMResponse[T] with correct content, usage, model, provider."""
        fake = FakeLLMProvider()
        fake.set_response(Greeting, Greeting(text="hi"))

        resp = await fake.complete(
            messages=[{"role": "user", "content": "test"}],
            response_model=Greeting,
            model="claude-haiku-4-5-20251001",
        )

        assert isinstance(resp, LLMResponse)
        assert resp.model == "claude-haiku-4-5-20251001"
        assert resp.provider == "fake"
        assert resp.usage.input_tokens > 0
        assert resp.usage.output_tokens > 0
        assert resp.latency_ms == 0.0

    async def test_token_usage_defaults(self) -> None:
        """TokenUsage uses configurable default_input_tokens/default_output_tokens."""
        fake = FakeLLMProvider()
        fake.set_response(Greeting, Greeting(text="hi"))

        resp = await fake.complete(
            messages=[{"role": "user", "content": "test"}],
            response_model=Greeting,
            model="test-model",
        )

        assert resp.usage.input_tokens == 100
        assert resp.usage.output_tokens == 50

    async def test_custom_token_counts(self) -> None:
        """Constructor accepts custom default token counts."""
        fake = FakeLLMProvider(default_input_tokens=500, default_output_tokens=200)
        fake.set_response(Greeting, Greeting(text="hi"))

        resp = await fake.complete(
            messages=[{"role": "user", "content": "test"}],
            response_model=Greeting,
            model="test-model",
        )

        assert resp.usage.input_tokens == 500
        assert resp.usage.output_tokens == 200

    async def test_calls_recorded(self) -> None:
        """Each complete() call is recorded in .calls list with full context."""
        fake = FakeLLMProvider()
        fake.set_response(Greeting, Greeting(text="hi"))

        msgs: list[LLMMessage] = [{"role": "user", "content": "hello"}]
        await fake.complete(
            messages=msgs,
            response_model=Greeting,
            model="test-model",
        )

        assert len(fake.calls) == 1
        call = fake.calls[0]
        assert call.response_model is Greeting
        assert call.model == "test-model"
        assert call.response == Greeting(text="hi")

    async def test_call_count_property(self) -> None:
        """call_count reflects number of complete() calls."""
        fake = FakeLLMProvider()
        fake.set_response(Greeting, Greeting(text="hi"))

        assert fake.call_count == 0

        for _ in range(3):
            await fake.complete(
                messages=[{"role": "user", "content": "hi"}],
                response_model=Greeting,
                model="m",
            )

        assert fake.call_count == 3

    async def test_close_is_noop(self) -> None:
        """close() completes without error."""
        fake = FakeLLMProvider()
        await fake.close()  # should not raise

    async def test_no_response_configured_raises(self) -> None:
        """complete() with unknown response_model and no factory raises."""
        fake = FakeLLMProvider()

        with pytest.raises(ResponseValidationError, match="No fake response configured"):
            await fake.complete(
                messages=[{"role": "user", "content": "hi"}],
                response_model=Greeting,
                model="m",
            )

    async def test_multiple_models_independent(self) -> None:
        """set_response for Model A doesn't affect Model B."""
        fake = FakeLLMProvider()
        fake.set_response(Greeting, Greeting(text="hello"))

        # Greeting works
        resp = await fake.complete(
            messages=[{"role": "user", "content": "hi"}],
            response_model=Greeting,
            model="m",
        )
        assert resp.content.text == "hello"

        # MathAnswer not configured — should raise
        with pytest.raises(ResponseValidationError):
            await fake.complete(
                messages=[{"role": "user", "content": "hi"}],
                response_model=MathAnswer,
                model="m",
            )

    async def test_from_config_creates_instance(self) -> None:
        """from_config() creates FakeLLMProvider from GatewayConfig."""
        config = GatewayConfig(provider="fake", trace_enabled=False)
        provider = FakeLLMProvider.from_config(config)

        assert isinstance(provider, FakeLLMProvider)
        assert provider.call_count == 0
        assert provider.calls == []

    async def test_provider_field_in_response(self) -> None:
        """LLMResponse.provider is 'fake'."""
        fake = FakeLLMProvider()
        fake.set_response(Greeting, Greeting(text="hi"))

        resp = await fake.complete(
            messages=[{"role": "user", "content": "hi"}],
            response_model=Greeting,
            model="m",
        )

        assert resp.provider == "fake"

    async def test_cost_calculated_for_known_model(self) -> None:
        """TokenUsage has non-zero cost when a known model is used."""
        fake = FakeLLMProvider()
        fake.set_response(Greeting, Greeting(text="hi"))

        resp = await fake.complete(
            messages=[{"role": "user", "content": "test"}],
            response_model=Greeting,
            model="claude-haiku-4-5-20251001",
        )

        assert resp.usage.total_cost_usd > 0.0


# ── FakeCall ─────────────────────────────────────────────────────


@pytest.mark.unit
class TestFakeCall:
    """Tests for FakeCall dataclass."""

    def test_fields(self) -> None:
        """FakeCall stores messages, response_model, model, response."""
        msgs: list[LLMMessage] = [{"role": "user", "content": "hi"}]
        call = FakeCall(
            messages=msgs,
            response_model=Greeting,
            model="test-model",
            response=Greeting(text="hi"),
        )

        assert call.messages == msgs
        assert call.response_model is Greeting
        assert call.model == "test-model"
        assert call.response == Greeting(text="hi")


# ── Registry Integration ─────────────────────────────────────────


@pytest.mark.unit
class TestFakeProviderRegistry:
    """Tests for fake provider in the registry."""

    def test_fake_provider_registered(self) -> None:
        """'fake' appears in list_providers() after builtins registered."""
        providers = list_providers()
        assert "fake" in providers

    def test_build_fake_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """build_provider(config) with provider='fake' returns FakeLLMProvider."""
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config = GatewayConfig(provider="fake", trace_enabled=False)
        provider = build_provider(config)

        assert isinstance(provider, FakeLLMProvider)


@pytest.mark.unit
class TestFakeLLMProviderCountTokens:
    """Tests for FakeLLMProvider.count_tokens()."""

    def test_count_tokens_returns_positive(self) -> None:
        """count_tokens should return a positive int for non-empty text."""
        fake = FakeLLMProvider()
        result = fake.count_tokens("Hello, world!")
        assert isinstance(result, int)
        assert result > 0

    def test_count_tokens_empty_string(self) -> None:
        """count_tokens should return 0 for empty string."""
        fake = FakeLLMProvider()
        assert fake.count_tokens("") == 0

    def test_count_tokens_heuristic(self) -> None:
        """FakeLLMProvider uses heuristic (chars/4)."""
        fake = FakeLLMProvider()
        # 40 chars / 4.0 = 10
        assert fake.count_tokens("a" * 40) == 10

    def test_count_tokens_lazy_init(self) -> None:
        """Tokenizer should be lazily initialized."""
        fake = FakeLLMProvider()
        assert fake._tokenizer is None
        fake.count_tokens("test")
        assert fake._tokenizer is not None
