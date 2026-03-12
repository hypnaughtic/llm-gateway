"""Dry-run integration tests — all LLM calls are mocked.

These tests validate that llm-gateway works correctly as an installed
package dependency: imports resolve, the client wires up, structured
responses are returned, cost tracking accumulates, etc.

No real LLM calls are made. Run with: pytest -v
"""

from __future__ import annotations

import pytest

from llm_gateway import (
    CostLimitExceededError,
    CostTracker,
    GatewayConfig,
    GatewayError,
    LLMClient,
    LLMProvider,
    LLMResponse,
    ProviderError,
    ProviderNotFoundError,
    TokenUsage,
    calculate_cost,
    list_providers,
    register_pricing,
    register_provider,
)
from llm_gateway.types import LLMMessage

from .conftest import FakeLLMProvider
from .response_models import (
    CapitalCity,
    FactAnswer,
    Greeting,
    MathAnswer,
    SentimentResult,
    SummaryResult,
    TranslationResult,
)

# ─── Package Import Validation ───────────────────────────────────


@pytest.mark.dry_run
class TestPackageInstallation:
    """Verify that the package installed via git dependency is fully functional."""

    def test_core_imports_resolve(self) -> None:
        """All public API symbols are importable."""
        assert LLMClient is not None
        assert GatewayConfig is not None
        assert LLMResponse is not None
        assert TokenUsage is not None
        assert CostTracker is not None
        assert LLMProvider is not None

    def test_exception_imports_resolve(self) -> None:
        """Exception hierarchy is importable."""
        assert issubclass(ProviderError, GatewayError)
        assert issubclass(ProviderNotFoundError, GatewayError)
        assert issubclass(CostLimitExceededError, GatewayError)

    def test_registry_functions_resolve(self) -> None:
        """Registry functions are importable and callable."""
        providers = list_providers()
        assert isinstance(providers, list)

    def test_cost_functions_resolve(self) -> None:
        """Cost functions are importable and callable."""
        input_cost, output_cost = calculate_cost("claude-haiku-4-5-20251001", 1000, 500)
        assert isinstance(input_cost, float)
        assert isinstance(output_cost, float)

    def test_config_loads_with_defaults(self) -> None:
        """GatewayConfig loads with defaults (no env vars required)."""
        config = GatewayConfig(provider="local_claude")
        assert config.provider == "local_claude"
        assert config.max_tokens == 4096
        assert config.temperature == 0.0

    def test_token_usage_is_frozen(self) -> None:
        """TokenUsage is immutable."""
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        with pytest.raises(AttributeError):
            usage.input_tokens = 200  # type: ignore[misc]


# ─── Single-Question Dry-Run Tests ──────────────────────────────


@pytest.mark.dry_run
class TestSingleQuestionDryRun:
    """Test individual LLM calls with mocked provider."""

    @pytest.mark.asyncio
    async def test_factual_question(
        self, dry_run_client: LLMClient, fake_provider: FakeLLMProvider
    ) -> None:
        """Ask a factual question and get a structured FactAnswer."""
        resp = await dry_run_client.complete(
            messages=[{"role": "user", "content": "What is the speed of light?"}],
            response_model=FactAnswer,
        )
        assert isinstance(resp, LLMResponse)
        assert isinstance(resp.content, FactAnswer)
        assert isinstance(resp.content.answer, str)
        assert len(resp.content.answer) > 0
        assert resp.provider == "fake"

    @pytest.mark.asyncio
    async def test_geography_question(
        self, dry_run_client: LLMClient, fake_provider: FakeLLMProvider
    ) -> None:
        """Ask a geography question and get a structured CapitalCity."""
        resp = await dry_run_client.complete(
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            response_model=CapitalCity,
        )
        assert isinstance(resp.content, CapitalCity)
        assert isinstance(resp.content.country, str)
        assert isinstance(resp.content.capital, str)

    @pytest.mark.asyncio
    async def test_sentiment_analysis(
        self, dry_run_client: LLMClient, fake_provider: FakeLLMProvider
    ) -> None:
        """Analyze sentiment of a text passage."""
        resp = await dry_run_client.complete(
            messages=[
                {
                    "role": "user",
                    "content": "Analyze the sentiment: 'I absolutely love this product!'",
                }
            ],
            response_model=SentimentResult,
        )
        assert isinstance(resp.content, SentimentResult)
        assert isinstance(resp.content.sentiment, str)
        assert isinstance(resp.content.text, str)

    @pytest.mark.asyncio
    async def test_translation(
        self, dry_run_client: LLMClient, fake_provider: FakeLLMProvider
    ) -> None:
        """Translate text between languages."""
        resp = await dry_run_client.complete(
            messages=[
                {
                    "role": "user",
                    "content": "Translate 'Hello world' from English to French.",
                }
            ],
            response_model=TranslationResult,
        )
        assert isinstance(resp.content, TranslationResult)
        assert isinstance(resp.content.translation, str)

    @pytest.mark.asyncio
    async def test_summarization(
        self, dry_run_client: LLMClient, fake_provider: FakeLLMProvider
    ) -> None:
        """Summarize a passage of text."""
        resp = await dry_run_client.complete(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Summarize: The Python programming language was created by "
                        "Guido van Rossum and first released in 1991. It emphasizes "
                        "code readability and supports multiple programming paradigms."
                    ),
                }
            ],
            response_model=SummaryResult,
        )
        assert isinstance(resp.content, SummaryResult)
        assert isinstance(resp.content.summary, str)

    @pytest.mark.asyncio
    async def test_math_question(
        self, dry_run_client: LLMClient, fake_provider: FakeLLMProvider
    ) -> None:
        """Solve a math problem with explanation."""
        resp = await dry_run_client.complete(
            messages=[{"role": "user", "content": "What is 17 * 23?"}],
            response_model=MathAnswer,
        )
        assert isinstance(resp.content, MathAnswer)
        assert isinstance(resp.content.answer, str)
        assert isinstance(resp.content.explanation, str)


# ─── Multi-Call / Session Tests ──────────────────────────────────


@pytest.mark.dry_run
class TestMultiCallSession:
    """Test multi-call sessions: cost tracking, call counts, context manager."""

    @pytest.mark.asyncio
    async def test_cost_accumulates_across_calls(
        self, dry_run_client: LLMClient, fake_provider: FakeLLMProvider
    ) -> None:
        """Cost and token counts accumulate across multiple calls."""
        for _ in range(3):
            await dry_run_client.complete(
                messages=[{"role": "user", "content": "Hello"}],
                response_model=Greeting,
            )

        assert dry_run_client.call_count == 3
        assert dry_run_client.total_tokens > 0

    @pytest.mark.asyncio
    async def test_cost_summary_keys(
        self, dry_run_client: LLMClient, fake_provider: FakeLLMProvider
    ) -> None:
        """cost_summary() returns dict with all expected keys."""
        await dry_run_client.complete(
            messages=[{"role": "user", "content": "Hi"}],
            response_model=Greeting,
        )
        summary = dry_run_client.cost_summary()
        assert "total_input_tokens" in summary
        assert "total_output_tokens" in summary
        assert "total_tokens" in summary
        assert "total_cost_usd" in summary
        assert "call_count" in summary
        assert summary["call_count"] == 1

    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self, fake_provider: FakeLLMProvider) -> None:
        """Async context manager properly closes the provider."""
        config = GatewayConfig(
            provider="fake", model="test", trace_enabled=False, log_format="console"
        )
        async with LLMClient(config=config, provider_instance=fake_provider) as client:
            await client.complete(
                messages=[{"role": "user", "content": "test"}],
                response_model=Greeting,
            )
            assert client.call_count == 1
        # After exiting, client is closed — no error

    @pytest.mark.asyncio
    async def test_cost_limit_enforcement(self, fake_provider: FakeLLMProvider) -> None:
        """CostLimitExceededError is raised when cost limit is breached."""
        config = GatewayConfig(
            provider="fake",
            model="claude-haiku-4-5-20251001",  # has known pricing
            cost_limit_usd=0.0001,  # extremely low limit
            trace_enabled=False,
            log_format="console",
        )
        client = LLMClient(config=config, provider_instance=fake_provider)

        with pytest.raises(CostLimitExceededError):
            await client.complete(
                messages=[{"role": "user", "content": "test"}],
                response_model=Greeting,
                model="claude-haiku-4-5-20251001",
            )

    @pytest.mark.asyncio
    async def test_provider_call_log_captures_messages(
        self, dry_run_client: LLMClient, fake_provider: FakeLLMProvider
    ) -> None:
        """The fake provider logs each call's messages and parameters."""
        await dry_run_client.complete(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
            ],
            response_model=MathAnswer,
        )

        assert len(fake_provider.call_log) == 1
        logged = fake_provider.call_log[0]
        assert logged["response_model"] == "MathAnswer"
        assert len(logged["messages"]) == 2
        assert logged["messages"][1]["content"] == "What is 2+2?"


# ─── Response Structure Validation ───────────────────────────────


@pytest.mark.dry_run
class TestResponseStructure:
    """Validate the LLMResponse wrapper returned by every call."""

    @pytest.mark.asyncio
    async def test_response_has_usage(
        self, dry_run_client: LLMClient, fake_provider: FakeLLMProvider
    ) -> None:
        """Every response includes TokenUsage with token counts."""
        resp = await dry_run_client.complete(
            messages=[{"role": "user", "content": "test"}],
            response_model=Greeting,
        )
        assert isinstance(resp.usage, TokenUsage)
        assert resp.usage.input_tokens > 0
        assert resp.usage.output_tokens > 0
        assert resp.usage.total_tokens == resp.usage.input_tokens + resp.usage.output_tokens

    @pytest.mark.asyncio
    async def test_response_has_model_and_provider(
        self, dry_run_client: LLMClient, fake_provider: FakeLLMProvider
    ) -> None:
        """Response includes model and provider metadata."""
        resp = await dry_run_client.complete(
            messages=[{"role": "user", "content": "test"}],
            response_model=Greeting,
        )
        assert isinstance(resp.model, str)
        assert isinstance(resp.provider, str)
        assert len(resp.model) > 0
        assert len(resp.provider) > 0

    @pytest.mark.asyncio
    async def test_response_has_latency(
        self, dry_run_client: LLMClient, fake_provider: FakeLLMProvider
    ) -> None:
        """Response includes latency_ms measurement."""
        resp = await dry_run_client.complete(
            messages=[{"role": "user", "content": "test"}],
            response_model=Greeting,
        )
        assert isinstance(resp.latency_ms, float)
        assert resp.latency_ms >= 0


# ─── Custom Provider Registration ────────────────────────────────


@pytest.mark.dry_run
class TestCustomProviderRegistration:
    """Test the extensibility story: registering a custom provider from consumer code."""

    @pytest.mark.asyncio
    async def test_register_and_use_custom_provider(self) -> None:
        """Consumer can register a custom provider and use it via config."""
        from collections.abc import Sequence
        from typing import TypeVar

        _T = TypeVar("_T")

        class EchoProvider:
            async def complete(
                self,
                messages: Sequence[LLMMessage],
                response_model: type[_T],
                model: str,
                max_tokens: int = 4096,
                temperature: float = 0.0,
                image_files: Sequence[str] | None = None,
            ) -> LLMResponse[_T]:
                content = response_model.model_validate(  # type: ignore[union-attr]
                    {"greeting": "echo_hello"}
                )
                return LLMResponse(
                    content=content,
                    usage=TokenUsage(input_tokens=10, output_tokens=5),
                    model=model,
                    provider="echo_test",
                )

            async def close(self) -> None:
                pass

        register_provider("echo_integration_test", lambda config: EchoProvider())

        config = GatewayConfig(
            provider="echo_integration_test",
            trace_enabled=False,
            log_format="console",
        )
        async with LLMClient(config=config) as client:
            resp = await client.complete(
                messages=[{"role": "user", "content": "hello"}],
                response_model=Greeting,
            )
            assert resp.content.greeting == "echo_hello"
            assert resp.provider == "echo_test"

    def test_pricing_registration_from_consumer(self) -> None:
        """Consumer can register pricing for custom models."""
        register_pricing("my-private-model", input_per_1m=5.0, output_per_1m=25.0)
        input_cost, output_cost = calculate_cost("my-private-model", 1_000_000, 1_000_000)
        assert input_cost == pytest.approx(5.0)
        assert output_cost == pytest.approx(25.0)
