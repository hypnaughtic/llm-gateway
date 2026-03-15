"""Anthropic provider — wraps AsyncAnthropic + instructor for structured output."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TypeVar

from tenacity import retry, stop_after_attempt, wait_exponential

from llm_gateway.cost import build_token_usage
from llm_gateway.exceptions import ProviderError
from llm_gateway.tokenizers.anthropic_tokenizer import AnthropicTokenizer
from llm_gateway.types import LLMMessage, LLMResponse, TokenUsage

try:
    import instructor
    from anthropic import AsyncAnthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

if not HAS_ANTHROPIC:
    msg = (
        "Anthropic provider requires 'anthropic' and 'instructor' packages. "
        "Install with: pip install 'llm-gateway[anthropic]'"
    )
    raise ImportError(msg)

from llm_gateway.config import GatewayConfig

T = TypeVar("T")


class AnthropicProvider:
    """LLM provider backed by the Anthropic API via instructor."""

    DEFAULT_MODEL = "claude-sonnet-4-5-20250514"

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        max_retries: int = 3,
        timeout_seconds: int = 120,
    ) -> None:
        self._client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            timeout=float(timeout_seconds),
        )
        self._instructor = instructor.from_anthropic(self._client)
        self._max_retries = max_retries
        self._tokenizer: AnthropicTokenizer | None = None

    @classmethod
    def from_config(cls, config: GatewayConfig) -> AnthropicProvider:
        """Factory method for the provider registry."""
        return cls(
            api_key=config.get_api_key(),
            base_url=config.base_url,
            max_retries=config.max_retries,
            timeout_seconds=config.timeout_seconds,
        )

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        response_model: type[T],
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        image_files: Sequence[str] | None = None,
    ) -> LLMResponse[T]:
        """Call Anthropic API and return structured response with usage.

        Note: image_files is accepted for protocol compatibility but not yet
        implemented for the Anthropic provider. Future versions may use the
        Messages API vision capability.
        """
        effective_model = model or self.DEFAULT_MODEL
        start = time.monotonic()

        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        )
        async def _do_call() -> T:
            result: T = await self._instructor.messages.create(  # type: ignore[type-var]
                model=effective_model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=list(messages),  # type: ignore[arg-type]
                response_model=response_model,
            )
            return result

        try:
            result = await _do_call()
        except Exception as exc:
            raise ProviderError("anthropic", exc) from exc

        latency_ms = (time.monotonic() - start) * 1000
        usage = self._extract_usage(result, effective_model)

        return LLMResponse(
            content=result,
            usage=usage,
            model=effective_model,
            provider="anthropic",
            latency_ms=latency_ms,
        )

    @staticmethod
    def _extract_usage(result: object, model: str) -> TokenUsage:
        """Extract token usage from instructor's _raw_response."""
        raw = getattr(result, "_raw_response", None)
        if raw is None:
            return build_token_usage(model, 0, 0)

        usage = getattr(raw, "usage", None)
        if usage is None:
            return build_token_usage(model, 0, 0)

        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        return build_token_usage(model, input_tokens, output_tokens)

    def count_tokens(self, text: str) -> int:
        """Count tokens using the Anthropic/Claude tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = AnthropicTokenizer()
        return self._tokenizer.count_tokens(text)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()
