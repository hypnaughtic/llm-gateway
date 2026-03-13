"""Gemini provider — wraps google-genai + instructor for structured output."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import TypeVar

from tenacity import retry, stop_after_attempt, wait_exponential

from llm_gateway.cost import build_token_usage
from llm_gateway.exceptions import ProviderError
from llm_gateway.types import LLMMessage, LLMResponse, TokenUsage

try:
    import instructor
    from google import genai
    from google.genai.types import GenerateContentConfig

    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

if not HAS_GEMINI:
    msg = (
        "Gemini provider requires 'google-genai' and 'instructor' packages. "
        "Install with: pip install 'llm-gateway[gemini]'"
    )
    raise ImportError(msg)

from llm_gateway.config import GatewayConfig

T = TypeVar("T")


class GeminiProvider:
    """LLM provider backed by the Gemini API via google-genai and instructor."""

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(
        self,
        api_key: str,
        max_retries: int = 3,
        timeout_seconds: int = 120,
    ) -> None:
        self._genai_client = genai.Client(api_key=api_key)
        self._instructor = instructor.from_genai(
            client=self._genai_client,
            mode=instructor.Mode.GENAI_TOOLS,
        )
        self._max_retries = max_retries
        self._timeout_seconds = timeout_seconds

    @classmethod
    def from_config(cls, config: GatewayConfig) -> GeminiProvider:
        """Factory method for the provider registry."""
        return cls(
            api_key=config.get_api_key(),
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
        """Call Gemini API and return structured response with usage.

        Note: image_files is accepted for protocol compatibility but not yet
        implemented for the Gemini provider.
        """
        effective_model = model or self.DEFAULT_MODEL
        start = time.monotonic()

        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        )
        def _do_call() -> T:
            instructor_messages = []
            for m in messages:
                if isinstance(m, dict):
                    instructor_messages.append(m)
                else:
                    instructor_messages.append({"role": m.role, "content": m.content})

            result: T = self._instructor.chat.completions.create(
                model=effective_model,
                messages=instructor_messages,
                response_model=response_model,
                config=GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            return result

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(_do_call),
                timeout=float(self._timeout_seconds),
            )
        except TimeoutError as exc:
            raise ProviderError("gemini", exc) from exc
        except Exception as exc:
            raise ProviderError("gemini", exc) from exc

        latency_ms = (time.monotonic() - start) * 1000
        usage = self._extract_usage(result, effective_model)

        return LLMResponse(
            content=result,
            usage=usage,
            model=effective_model,
            provider="gemini",
            latency_ms=latency_ms,
        )

    @staticmethod
    def _extract_usage(result: object, model: str) -> TokenUsage:
        """Extract token usage from instructor's _raw_response.usage_metadata."""
        raw = getattr(result, "_raw_response", None)
        if raw is None:
            return build_token_usage(model, 0, 0)

        usage_metadata = getattr(raw, "usage_metadata", None)
        if usage_metadata is None:
            return build_token_usage(model, 0, 0)

        input_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0
        return build_token_usage(model, input_tokens, output_tokens)

    async def close(self) -> None:
        """Close resources (no-op for Gemini sync client)."""
