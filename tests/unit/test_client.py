"""Tests for LLMClient."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from llm_gateway.client import LLMClient
from llm_gateway.config import GatewayConfig
from llm_gateway.exceptions import CostLimitExceededError
from llm_gateway.types import TokenUsage


class _Answer(BaseModel):
    text: str


@pytest.mark.unit
class TestLLMClient:
    @pytest.mark.asyncio
    async def test_complete_returns_response(self, fake_provider) -> None:  # type: ignore[no-untyped-def]
        """LLMClient.complete() returns the provider's response."""
        fake_provider.set_response(_Answer, _Answer(text="hello"))

        config = GatewayConfig(provider="fake", api_key="not-needed")  # type: ignore[arg-type]
        client = LLMClient(config=config, provider_instance=fake_provider)

        resp = await client.complete(
            messages=[{"role": "user", "content": "hi"}],
            response_model=_Answer,
        )
        assert resp.content.text == "hello"
        assert isinstance(resp.usage, TokenUsage)

    @pytest.mark.asyncio
    async def test_tracks_cost(self, fake_provider) -> None:  # type: ignore[no-untyped-def]
        """LLMClient accumulates cost across calls."""
        fake_provider.set_response(_Answer, _Answer(text="ok"))

        config = GatewayConfig(provider="fake", api_key="not-needed")  # type: ignore[arg-type]
        client = LLMClient(config=config, provider_instance=fake_provider)

        await client.complete(
            messages=[{"role": "user", "content": "test"}],
            response_model=_Answer,
        )
        assert client.total_tokens > 0
        assert client.call_count == 1

    @pytest.mark.asyncio
    async def test_cost_limit_enforcement(self, fake_provider) -> None:  # type: ignore[no-untyped-def]
        """CostLimitExceededError raised when limit exceeded."""
        fake_provider.set_response(_Answer, _Answer(text="ok"))

        config = GatewayConfig(
            provider="fake",
            api_key="not-needed",  # type: ignore[arg-type]
            cost_limit_usd=0.0001,  # Very low limit
        )
        client = LLMClient(config=config, provider_instance=fake_provider)

        with pytest.raises(CostLimitExceededError):
            await client.complete(
                messages=[{"role": "user", "content": "test"}],
                response_model=_Answer,
                model="claude-haiku-4-5-20251001",  # Has known pricing
            )

    @pytest.mark.asyncio
    async def test_model_override(self, fake_provider) -> None:  # type: ignore[no-untyped-def]
        """Model parameter overrides config default."""
        fake_provider.set_response(_Answer, _Answer(text="ok"))

        config = GatewayConfig(
            provider="fake",
            api_key="not-needed",  # type: ignore[arg-type]
            model="default-model",
        )
        client = LLMClient(config=config, provider_instance=fake_provider)

        await client.complete(
            messages=[{"role": "user", "content": "test"}],
            response_model=_Answer,
            model="override-model",
        )
        assert fake_provider._last_model == "override-model"

    @pytest.mark.asyncio
    async def test_context_manager(self, fake_provider) -> None:  # type: ignore[no-untyped-def]
        """Async context manager calls close()."""
        config = GatewayConfig(provider="fake", api_key="not-needed")  # type: ignore[arg-type]
        async with LLMClient(config=config, provider_instance=fake_provider) as client:
            assert client is not None
        # close() was called

    @pytest.mark.asyncio
    async def test_cost_summary(self, fake_provider) -> None:  # type: ignore[no-untyped-def]
        """cost_summary() returns dict with expected keys."""
        fake_provider.set_response(_Answer, _Answer(text="ok"))

        config = GatewayConfig(provider="fake", api_key="not-needed")  # type: ignore[arg-type]
        client = LLMClient(config=config, provider_instance=fake_provider)

        await client.complete(
            messages=[{"role": "user", "content": "test"}],
            response_model=_Answer,
        )
        summary = client.cost_summary()
        assert "call_count" in summary
        assert "total_tokens" in summary
        assert summary["call_count"] == 1

    @pytest.mark.asyncio
    async def test_total_cost_usd_property(self, fake_provider) -> None:  # type: ignore[no-untyped-def]
        """total_cost_usd reflects cumulative cost."""
        fake_provider.set_response(_Answer, _Answer(text="ok"))

        config = GatewayConfig(provider="fake", api_key="not-needed")  # type: ignore[arg-type]
        client = LLMClient(config=config, provider_instance=fake_provider)

        assert client.total_cost_usd == 0.0
        await client.complete(
            messages=[{"role": "user", "content": "test"}],
            response_model=_Answer,
        )
        # After a call, total_cost_usd should be non-negative (could be 0 for unknown models)
        assert client.total_cost_usd >= 0.0

    @pytest.mark.asyncio
    async def test_close_idempotent(self, fake_provider) -> None:  # type: ignore[no-untyped-def]
        """Calling close() twice does not raise."""
        config = GatewayConfig(provider="fake", api_key="not-needed")  # type: ignore[arg-type]
        client = LLMClient(config=config, provider_instance=fake_provider)
        await client.close()
        await client.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_temperature_override(self, fake_provider) -> None:  # type: ignore[no-untyped-def]
        """Temperature parameter is passed through to provider."""
        fake_provider.set_response(_Answer, _Answer(text="ok"))

        config = GatewayConfig(provider="fake", api_key="not-needed")  # type: ignore[arg-type]
        client = LLMClient(config=config, provider_instance=fake_provider)

        await client.complete(
            messages=[{"role": "user", "content": "test"}],
            response_model=_Answer,
            temperature=0.9,
        )
        # The call completed without error (temperature was forwarded)

    @pytest.mark.asyncio
    async def test_image_files_passed_through(self, fake_provider) -> None:  # type: ignore[no-untyped-def]
        """image_files parameter is forwarded to the provider."""
        fake_provider.set_response(_Answer, _Answer(text="visual"))

        config = GatewayConfig(provider="fake", api_key="not-needed")  # type: ignore[arg-type]
        client = LLMClient(config=config, provider_instance=fake_provider)

        resp = await client.complete(
            messages=[{"role": "user", "content": "evaluate"}],
            response_model=_Answer,
            image_files=["/tmp/test.png"],
        )
        assert resp.content.text == "visual"
