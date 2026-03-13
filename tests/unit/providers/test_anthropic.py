"""Tests for AnthropicProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from llm_gateway.types import LLMResponse


class _TestModel(BaseModel):
    answer: str


@pytest.mark.unit
class TestAnthropicProvider:
    @pytest.mark.asyncio
    async def test_complete_returns_llm_response(self) -> None:
        """AnthropicProvider.complete() wraps instructor result in LLMResponse."""
        from llm_gateway.providers.anthropic import AnthropicProvider

        with (
            patch("llm_gateway.providers.anthropic.AsyncAnthropic"),
            patch("llm_gateway.providers.anthropic.instructor") as mock_instructor,
        ):
            provider = AnthropicProvider(api_key="test-key")

            expected = _TestModel(answer="hello")
            # Attach fake _raw_response for token extraction
            raw = MagicMock()
            raw.usage.input_tokens = 100
            raw.usage.output_tokens = 50
            expected._raw_response = raw  # type: ignore[attr-defined]

            mock_instructor.from_anthropic.return_value.messages.create = AsyncMock(
                return_value=expected
            )

            resp = await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                response_model=_TestModel,
                model="claude-haiku-4-5-20251001",
            )

            assert isinstance(resp, LLMResponse)
            assert resp.content.answer == "hello"
            assert resp.usage.input_tokens == 100
            assert resp.usage.output_tokens == 50
            assert resp.provider == "anthropic"

    @pytest.mark.asyncio
    async def test_missing_raw_response(self) -> None:
        """Gracefully handles missing _raw_response (usage = 0)."""
        from llm_gateway.providers.anthropic import AnthropicProvider

        with (
            patch("llm_gateway.providers.anthropic.AsyncAnthropic"),
            patch("llm_gateway.providers.anthropic.instructor") as mock_instructor,
        ):
            provider = AnthropicProvider(api_key="test-key")
            expected = _TestModel(answer="ok")
            # No _raw_response attached

            mock_instructor.from_anthropic.return_value.messages.create = AsyncMock(
                return_value=expected
            )

            resp = await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                response_model=_TestModel,
                model="claude-haiku-4-5-20251001",
            )

            assert resp.usage.input_tokens == 0
            assert resp.usage.output_tokens == 0

    @pytest.mark.asyncio
    async def test_from_config(self) -> None:
        """from_config factory creates a valid provider."""
        from llm_gateway.config import GatewayConfig
        from llm_gateway.providers.anthropic import AnthropicProvider

        with (
            patch("llm_gateway.providers.anthropic.AsyncAnthropic"),
            patch("llm_gateway.providers.anthropic.instructor"),
        ):
            config = GatewayConfig(
                provider="anthropic",
                api_key="test-key",  # type: ignore[arg-type]
            )
            provider = AnthropicProvider.from_config(config)
            assert isinstance(provider, AnthropicProvider)

    @pytest.mark.asyncio
    async def test_complete_wraps_exception_in_provider_error(self) -> None:
        """complete() wraps SDK exceptions in ProviderError."""
        from llm_gateway.exceptions import ProviderError
        from llm_gateway.providers.anthropic import AnthropicProvider

        with (
            patch("llm_gateway.providers.anthropic.AsyncAnthropic"),
            patch("llm_gateway.providers.anthropic.instructor") as mock_instructor,
        ):
            provider = AnthropicProvider(api_key="test-key", max_retries=1)

            original_exc = RuntimeError("API connection failed")
            mock_instructor.from_anthropic.return_value.messages.create = AsyncMock(
                side_effect=original_exc
            )

            with pytest.raises(ProviderError) as exc_info:
                await provider.complete(
                    messages=[{"role": "user", "content": "test"}],
                    response_model=_TestModel,
                    model="claude-haiku-4-5-20251001",
                )

            assert exc_info.value.provider == "anthropic"
            assert exc_info.value.original is original_exc
            assert exc_info.value.__cause__ is original_exc

    @pytest.mark.asyncio
    async def test_extract_usage_raw_response_without_usage(self) -> None:
        """_extract_usage returns zero tokens when usage attr is None."""
        from llm_gateway.providers.anthropic import AnthropicProvider

        with (
            patch("llm_gateway.providers.anthropic.AsyncAnthropic"),
            patch("llm_gateway.providers.anthropic.instructor") as mock_instructor,
        ):
            provider = AnthropicProvider(api_key="test-key")

            expected = _TestModel(answer="ok")
            # _raw_response exists but has no usage attribute
            raw = MagicMock(spec=[])  # spec=[] means no attributes
            expected._raw_response = raw  # type: ignore[attr-defined]

            mock_instructor.from_anthropic.return_value.messages.create = AsyncMock(
                return_value=expected
            )

            resp = await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                response_model=_TestModel,
                model="claude-haiku-4-5-20251001",
            )

            assert resp.usage.input_tokens == 0
            assert resp.usage.output_tokens == 0

    @pytest.mark.asyncio
    async def test_close_delegates_to_underlying_client(self) -> None:
        """close() calls close on the underlying AsyncAnthropic client."""
        from llm_gateway.providers.anthropic import AnthropicProvider

        with (
            patch("llm_gateway.providers.anthropic.AsyncAnthropic") as mock_cls,
            patch("llm_gateway.providers.anthropic.instructor"),
        ):
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client

            provider = AnthropicProvider(api_key="test-key")
            await provider.close()

            mock_client.close.assert_awaited_once()
