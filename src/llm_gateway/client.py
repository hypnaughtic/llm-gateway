"""LLMClient — the single class consumers import and use."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, TypeVar

from llm_gateway.config import GatewayConfig
from llm_gateway.cost import CostTracker
from llm_gateway.observability.logging import configure_logging
from llm_gateway.observability.tracing import configure_tracing, traced_llm_call
from llm_gateway.providers.base import LLMProvider
from llm_gateway.registry import build_provider
from llm_gateway.types import LLMMessage, LLMResponse

logger = logging.getLogger(__name__)

T = TypeVar("T")


class LLMClient:
    """Unified LLM client with config-driven provider selection.

    This is the ONE class consumers should import. Provider switching
    happens entirely via environment variables — zero code changes.

    Usage:
        # Reads LLM_* env vars automatically
        llm = LLMClient()

        # Or with explicit config
        llm = LLMClient(config=GatewayConfig(provider="local_claude"))

        # Or with injected provider (for testing)
        llm = LLMClient(provider_instance=my_mock_provider)

        # Make a call
        resp = await llm.complete(
            messages=[{"role": "user", "content": "Hello"}],
            response_model=MyModel,
        )
        print(resp.content)         # MyModel instance
        print(resp.usage.total_cost_usd)  # Cost in USD
    """

    def __init__(
        self,
        config: GatewayConfig | None = None,
        provider_instance: LLMProvider | None = None,
    ) -> None:
        self._config = config or GatewayConfig()
        self._provider = provider_instance or build_provider(self._config)
        self._cost_tracker = CostTracker(
            cost_limit_usd=self._config.cost_limit_usd,
            cost_warn_usd=self._config.cost_warn_usd,
        )
        self._closed = False

        # Auto-configure observability
        configure_logging(
            level=self._config.log_level,
            fmt=self._config.log_format,
        )
        if self._config.trace_enabled:
            configure_tracing(
                exporter=self._config.trace_exporter,
                endpoint=self._config.trace_endpoint,
                service_name=self._config.trace_service_name,
            )

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        response_model: type[T],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        image_files: Sequence[str] | None = None,
    ) -> LLMResponse[T]:
        """Send messages to the configured LLM and return a structured response.

        Args:
            messages: Conversation messages.
            response_model: Pydantic model for structured output validation.
            model: Override the default model from config.
            max_tokens: Override the default max_tokens from config.
            temperature: Override the default temperature from config.
            image_files: Optional list of local file paths to images for
                multimodal evaluation. Passed through to the provider.

        Returns:
            LLMResponse[T] with validated content, token usage, and cost.

        Raises:
            CostLimitExceededError: If cumulative cost exceeds the limit.
            ProviderError: If the underlying provider raises an error.
            ResponseValidationError: If the response cannot be validated.
        """
        effective_model = model or self._config.model
        effective_max_tokens = max_tokens or self._config.max_tokens
        effective_temperature = (
            temperature if temperature is not None else self._config.temperature
        )

        async with traced_llm_call(
            model=effective_model,
            provider=self._config.provider,
        ) as span_data:
            response = await self._provider.complete(
                messages=messages,
                response_model=response_model,
                model=effective_model,
                max_tokens=effective_max_tokens,
                temperature=effective_temperature,
                image_files=image_files,
            )
            span_data["response"] = response

        # Track cost
        self._cost_tracker.record(response.usage)

        logger.info(
            "LLM call completed",
            extra={
                "provider": response.provider,
                "model": response.model,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cost_usd": response.usage.total_cost_usd,
                "latency_ms": round(response.latency_ms, 1),
                "cumulative_cost_usd": self._cost_tracker.total_cost_usd,
            },
        )

        return response

    @property
    def total_cost_usd(self) -> float:
        """Cumulative cost across all calls on this client instance."""
        return self._cost_tracker.total_cost_usd

    @property
    def total_tokens(self) -> int:
        """Cumulative tokens across all calls on this client instance."""
        return self._cost_tracker.total_tokens

    @property
    def call_count(self) -> int:
        """Number of LLM calls made on this client instance."""
        return self._cost_tracker.call_count

    def cost_summary(self) -> dict[str, Any]:
        """Return a summary dict of cost/token usage."""
        return self._cost_tracker.summary()

    async def close(self) -> None:
        """Clean up provider resources."""
        if not self._closed:
            await self._provider.close()
            self._closed = True

    async def __aenter__(self) -> LLMClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Async context manager exit — closes provider."""
        await self.close()
