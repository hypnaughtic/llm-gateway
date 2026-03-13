"""Tests for GeminiProvider — fully mocked, no google-genai install needed."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from llm_gateway.exceptions import ProviderError
from llm_gateway.types import LLMResponse


class _TestModel(BaseModel):
    answer: str


# ---------------------------------------------------------------------------
# Module-level mock setup: mock google.genai and instructor BEFORE importing
# GeminiProvider so the module-level try/except succeeds.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_gemini_deps(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Inject mock google.genai and instructor into sys.modules.

    This allows importing GeminiProvider without having google-genai
    actually installed (or working on the current Python version).
    """
    mock_genai = MagicMock()
    mock_genai_types = MagicMock()
    mock_google = MagicMock()
    mock_instructor = MagicMock()

    # Ensure the module is reloaded each time with fresh mocks
    for mod_name in list(sys.modules):
        if "llm_gateway.providers.gemini" in mod_name:
            del sys.modules[mod_name]

    monkeypatch.setitem(sys.modules, "google", mock_google)
    monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", mock_genai_types)
    monkeypatch.setitem(sys.modules, "instructor", mock_instructor)

    # Wire up the mock so `from google import genai` works
    mock_google.genai = mock_genai
    mock_genai.types = mock_genai_types

    return mock_instructor


def _import_gemini_provider() -> Any:
    """Import (or re-import) GeminiProvider with mocked dependencies."""
    if "llm_gateway.providers.gemini" in sys.modules:
        del sys.modules["llm_gateway.providers.gemini"]
    from llm_gateway.providers.gemini import GeminiProvider

    return GeminiProvider


def _make_mock_result(value: str, *, tokens: bool = True) -> _TestModel:
    """Create a mock result with optional _raw_response for token extraction."""
    obj = _TestModel(answer=value)
    if tokens:
        raw = MagicMock()
        raw.usage_metadata.prompt_token_count = 120
        raw.usage_metadata.candidates_token_count = 45
        obj._raw_response = raw  # type: ignore[attr-defined]
    return obj


@pytest.mark.unit
class TestGeminiProviderComplete:
    """Tests for GeminiProvider.complete() with mocked google-genai + instructor."""

    @pytest.mark.asyncio
    async def test_complete_returns_llm_response(self) -> None:
        """complete() wraps instructor result in LLMResponse."""
        GeminiProvider = _import_gemini_provider()
        provider = GeminiProvider(api_key="test-key")

        expected = _make_mock_result("hello from gemini")
        provider._instructor.chat.completions.create = MagicMock(return_value=expected)

        resp = await provider.complete(
            messages=[{"role": "user", "content": "test"}],
            response_model=_TestModel,
            model="gemini-2.5-flash",
        )

        assert isinstance(resp, LLMResponse)
        assert resp.content.answer == "hello from gemini"
        assert resp.provider == "gemini"
        assert resp.model == "gemini-2.5-flash"
        assert resp.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_token_extraction_from_raw_response(self) -> None:
        """Token counts are correctly extracted from _raw_response.usage_metadata."""
        GeminiProvider = _import_gemini_provider()
        provider = GeminiProvider(api_key="test-key")

        expected = _TestModel(answer="tokens test")
        raw = MagicMock()
        raw.usage_metadata.prompt_token_count = 200
        raw.usage_metadata.candidates_token_count = 80
        expected._raw_response = raw  # type: ignore[attr-defined]

        provider._instructor.chat.completions.create = MagicMock(return_value=expected)

        resp = await provider.complete(
            messages=[{"role": "user", "content": "count my tokens"}],
            response_model=_TestModel,
        )

        assert resp.usage.input_tokens == 200
        assert resp.usage.output_tokens == 80
        assert resp.usage.total_tokens == 280

    @pytest.mark.asyncio
    async def test_missing_raw_response_returns_zero_usage(self) -> None:
        """Gracefully handles missing _raw_response — usage falls back to zeros."""
        GeminiProvider = _import_gemini_provider()
        provider = GeminiProvider(api_key="test-key")

        expected = _TestModel(answer="no raw")
        provider._instructor.chat.completions.create = MagicMock(return_value=expected)

        resp = await provider.complete(
            messages=[{"role": "user", "content": "test"}],
            response_model=_TestModel,
        )

        assert resp.usage.input_tokens == 0
        assert resp.usage.output_tokens == 0
        assert resp.usage.total_cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_missing_usage_metadata_returns_zero_usage(self) -> None:
        """Handles _raw_response present but usage_metadata missing."""
        GeminiProvider = _import_gemini_provider()
        provider = GeminiProvider(api_key="test-key")

        expected = _TestModel(answer="partial raw")
        raw = MagicMock(spec=[])  # No usage_metadata attribute
        expected._raw_response = raw  # type: ignore[attr-defined]

        provider._instructor.chat.completions.create = MagicMock(return_value=expected)

        resp = await provider.complete(
            messages=[{"role": "user", "content": "test"}],
            response_model=_TestModel,
        )

        assert resp.usage.input_tokens == 0
        assert resp.usage.output_tokens == 0

    @pytest.mark.asyncio
    async def test_cost_calculated_from_pricing_registry(self) -> None:
        """Usage cost is calculated using the Gemini pricing registry."""
        GeminiProvider = _import_gemini_provider()
        provider = GeminiProvider(api_key="test-key")

        expected = _TestModel(answer="priced")
        raw = MagicMock()
        raw.usage_metadata.prompt_token_count = 1_000_000
        raw.usage_metadata.candidates_token_count = 1_000_000
        expected._raw_response = raw  # type: ignore[attr-defined]

        provider._instructor.chat.completions.create = MagicMock(return_value=expected)

        resp = await provider.complete(
            messages=[{"role": "user", "content": "test"}],
            response_model=_TestModel,
            model="gemini-2.5-flash",
        )

        assert resp.usage.input_cost_usd == pytest.approx(0.15)
        assert resp.usage.output_cost_usd == pytest.approx(0.60)

    @pytest.mark.asyncio
    async def test_complete_converts_llm_message_dicts(self) -> None:
        """Dict-style messages are passed through to instructor."""
        GeminiProvider = _import_gemini_provider()
        provider = GeminiProvider(api_key="test-key")

        expected = _make_mock_result("dict msg")
        mock_create = MagicMock(return_value=expected)
        provider._instructor.chat.completions.create = mock_create

        await provider.complete(
            messages=[{"role": "user", "content": "hello"}],
            response_model=_TestModel,
        )

        call_kwargs = mock_create.call_args
        msgs = call_kwargs.kwargs["messages"]
        assert msgs == [{"role": "user", "content": "hello"}]

    @pytest.mark.asyncio
    async def test_image_files_accepted_for_protocol_compat(self) -> None:
        """image_files param is accepted without error (protocol compatibility)."""
        GeminiProvider = _import_gemini_provider()
        provider = GeminiProvider(api_key="test-key")

        expected = _make_mock_result("image compat")
        provider._instructor.chat.completions.create = MagicMock(return_value=expected)

        resp = await provider.complete(
            messages=[{"role": "user", "content": "test"}],
            response_model=_TestModel,
            image_files=["test.png"],
        )

        assert resp.content.answer == "image compat"


@pytest.mark.unit
class TestGeminiProviderErrorHandling:
    """Tests for error wrapping and retry logic."""

    @pytest.mark.asyncio
    async def test_provider_error_on_sdk_exception(self) -> None:
        """SDK exceptions are wrapped in ProviderError('gemini', ...)."""
        GeminiProvider = _import_gemini_provider()
        provider = GeminiProvider(api_key="test-key", max_retries=1)

        provider._instructor.chat.completions.create = MagicMock(
            side_effect=RuntimeError("API quota exceeded")
        )

        with pytest.raises(ProviderError) as exc_info:
            await provider.complete(
                messages=[{"role": "user", "content": "fail"}],
                response_model=_TestModel,
            )

        assert exc_info.value.provider == "gemini"
        assert "API quota exceeded" in str(exc_info.value.original)

    @pytest.mark.asyncio
    async def test_retry_on_transient_failure(self) -> None:
        """Retries transient failures before succeeding."""
        GeminiProvider = _import_gemini_provider()
        provider = GeminiProvider(api_key="test-key", max_retries=3)

        success = _make_mock_result("recovered")
        mock_create = MagicMock(
            side_effect=[
                RuntimeError("transient 1"),
                RuntimeError("transient 2"),
                success,
            ]
        )
        provider._instructor.chat.completions.create = mock_create

        resp = await provider.complete(
            messages=[{"role": "user", "content": "retry me"}],
            response_model=_TestModel,
        )

        assert resp.content.answer == "recovered"
        assert mock_create.call_count == 3

    @pytest.mark.asyncio
    async def test_timeout_wraps_as_provider_error(self) -> None:
        """TimeoutError from asyncio.wait_for is wrapped in ProviderError."""
        GeminiProvider = _import_gemini_provider()
        provider = GeminiProvider(api_key="test-key", timeout_seconds=1)

        with patch("llm_gateway.providers.gemini.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = MagicMock()
            mock_asyncio.wait_for = MagicMock(side_effect=TimeoutError("timed out"))

            with pytest.raises(ProviderError) as exc_info:
                await provider.complete(
                    messages=[{"role": "user", "content": "slow"}],
                    response_model=_TestModel,
                )

            assert exc_info.value.provider == "gemini"

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_raises_provider_error(self) -> None:
        """When all retries are exhausted, ProviderError is raised."""
        GeminiProvider = _import_gemini_provider()
        provider = GeminiProvider(api_key="test-key", max_retries=2)

        provider._instructor.chat.completions.create = MagicMock(
            side_effect=RuntimeError("persistent failure")
        )

        with pytest.raises(ProviderError):
            await provider.complete(
                messages=[{"role": "user", "content": "fail always"}],
                response_model=_TestModel,
            )


@pytest.mark.unit
class TestGeminiProviderDefaults:
    """Tests for default values and configuration."""

    def test_default_model(self) -> None:
        """DEFAULT_MODEL is gemini-2.5-flash."""
        GeminiProvider = _import_gemini_provider()
        assert GeminiProvider.DEFAULT_MODEL == "gemini-2.5-flash"

    @pytest.mark.asyncio
    async def test_complete_uses_default_model_when_none(self) -> None:
        """When model=None, uses DEFAULT_MODEL."""
        GeminiProvider = _import_gemini_provider()
        provider = GeminiProvider(api_key="test-key")

        expected = _make_mock_result("default model")
        provider._instructor.chat.completions.create = MagicMock(return_value=expected)

        resp = await provider.complete(
            messages=[{"role": "user", "content": "test"}],
            response_model=_TestModel,
            model=None,
        )

        assert resp.model == "gemini-2.5-flash"

    @pytest.mark.asyncio
    async def test_close_is_noop(self) -> None:
        """close() completes without error."""
        GeminiProvider = _import_gemini_provider()
        provider = GeminiProvider(api_key="test-key")
        await provider.close()  # Should not raise


@pytest.mark.unit
class TestGeminiFromConfig:
    """Tests for GeminiProvider.from_config() factory."""

    def test_from_config_with_explicit_api_key(self) -> None:
        """from_config works when api_key is set in config."""
        from llm_gateway.config import GatewayConfig

        GeminiProvider = _import_gemini_provider()
        config = GatewayConfig(
            provider="gemini",
            api_key="test-gemini-key",  # type: ignore[arg-type]
        )
        provider = GeminiProvider.from_config(config)
        assert isinstance(provider, GeminiProvider)

    def test_from_config_falls_back_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """from_config falls back to GEMINI_API_KEY env var via config resolution."""
        monkeypatch.setenv("GEMINI_API_KEY", "env-gemini-key")
        monkeypatch.delenv("LLM_API_KEY", raising=False)

        from llm_gateway.config import GatewayConfig

        GeminiProvider = _import_gemini_provider()
        config = GatewayConfig(provider="gemini")
        provider = GeminiProvider.from_config(config)
        assert isinstance(provider, GeminiProvider)

    def test_from_config_no_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """from_config raises ValueError when no API key is available."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)

        from llm_gateway.config import GatewayConfig

        GeminiProvider = _import_gemini_provider()
        config = GatewayConfig(provider="gemini")
        with pytest.raises(ValueError, match="No API key configured"):
            GeminiProvider.from_config(config)

    def test_from_config_respects_max_retries(self) -> None:
        """from_config passes max_retries from config."""
        from llm_gateway.config import GatewayConfig

        GeminiProvider = _import_gemini_provider()
        config = GatewayConfig(
            provider="gemini",
            api_key="test-key",  # type: ignore[arg-type]
            max_retries=5,
        )
        provider = GeminiProvider.from_config(config)
        assert provider._max_retries == 5

    def test_from_config_respects_timeout(self) -> None:
        """from_config passes timeout_seconds from config."""
        from llm_gateway.config import GatewayConfig

        GeminiProvider = _import_gemini_provider()
        config = GatewayConfig(
            provider="gemini",
            api_key="test-key",  # type: ignore[arg-type]
            timeout_seconds=60,
        )
        provider = GeminiProvider.from_config(config)
        assert provider._timeout_seconds == 60


@pytest.mark.unit
class TestGeminiConfigResolution:
    """Tests for GatewayConfig API key resolution for Gemini provider."""

    def test_config_resolves_gemini_api_key_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GatewayConfig._resolve_api_key picks up GEMINI_API_KEY for provider=gemini."""
        monkeypatch.setenv("GEMINI_API_KEY", "resolved-key")
        monkeypatch.delenv("LLM_API_KEY", raising=False)

        from llm_gateway.config import GatewayConfig

        config = GatewayConfig(provider="gemini")
        assert config.api_key is not None
        assert config.api_key.get_secret_value() == "resolved-key"

    def test_config_llm_api_key_takes_precedence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLM_API_KEY takes precedence over GEMINI_API_KEY."""
        monkeypatch.setenv("LLM_API_KEY", "primary-key")
        monkeypatch.setenv("GEMINI_API_KEY", "fallback-key")

        from llm_gateway.config import GatewayConfig

        config = GatewayConfig(provider="gemini")
        assert config.api_key is not None
        assert config.api_key.get_secret_value() == "primary-key"

    def test_get_api_key_raises_when_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_api_key() raises ValueError when no key is configured."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)

        from llm_gateway.config import GatewayConfig

        config = GatewayConfig(provider="gemini")
        with pytest.raises(ValueError, match="No API key configured"):
            config.get_api_key()


@pytest.mark.unit
class TestGeminiRegistry:
    """Tests for Gemini registration in the provider registry."""

    def test_registry_includes_gemini(self) -> None:
        """list_providers() includes 'gemini' when google-genai is installed."""
        # Force re-registration with mocked deps
        import llm_gateway.registry as reg

        reg._builtins_registered = False

        providers = reg.list_providers()
        assert "gemini" in providers

    def test_build_provider_gemini(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """build_provider returns a GeminiProvider instance."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-registry-key")
        monkeypatch.delenv("LLM_API_KEY", raising=False)

        import llm_gateway.registry as reg

        reg._builtins_registered = False

        from llm_gateway.config import GatewayConfig

        GeminiProvider = _import_gemini_provider()
        config = GatewayConfig(provider="gemini")
        provider = reg.build_provider(config)
        assert isinstance(provider, GeminiProvider)


@pytest.mark.unit
class TestGeminiCostIntegration:
    """Tests for Gemini-specific pricing in the cost module."""

    def test_gemini_25_flash_pricing(self) -> None:
        """gemini-2.5-flash pricing is registered."""
        from llm_gateway.cost import get_pricing

        pricing = get_pricing("gemini-2.5-flash")
        assert pricing is not None
        assert pricing["input"] == 0.15
        assert pricing["output"] == 0.60

    def test_gemini_20_flash_pricing(self) -> None:
        """gemini-2.0-flash pricing is registered."""
        from llm_gateway.cost import get_pricing

        pricing = get_pricing("gemini-2.0-flash")
        assert pricing is not None
        assert pricing["input"] == 0.10
        assert pricing["output"] == 0.40

    def test_gemini_15_flash_pricing(self) -> None:
        """gemini-1.5-flash pricing is registered."""
        from llm_gateway.cost import get_pricing

        pricing = get_pricing("gemini-1.5-flash")
        assert pricing is not None
        assert pricing["input"] == 0.075
        assert pricing["output"] == 0.30

    def test_gemini_15_pro_pricing(self) -> None:
        """gemini-1.5-pro pricing is registered."""
        from llm_gateway.cost import get_pricing

        pricing = get_pricing("gemini-1.5-pro")
        assert pricing is not None
        assert pricing["input"] == 1.25
        assert pricing["output"] == 5.00

    def test_gemini_25_pro_pricing(self) -> None:
        """gemini-2.5-pro pricing is registered."""
        from llm_gateway.cost import get_pricing

        pricing = get_pricing("gemini-2.5-pro")
        assert pricing is not None
        assert pricing["input"] == 1.25
        assert pricing["output"] == 10.00

    def test_gemini_cost_calculation(self) -> None:
        """calculate_cost returns correct values for Gemini model."""
        from llm_gateway.cost import calculate_cost

        input_cost, output_cost = calculate_cost("gemini-2.5-flash", 1_000_000, 1_000_000)
        assert input_cost == pytest.approx(0.15)
        assert output_cost == pytest.approx(0.60)

    def test_gemini_build_token_usage(self) -> None:
        """build_token_usage returns TokenUsage with Gemini pricing."""
        from llm_gateway.cost import build_token_usage

        usage = build_token_usage("gemini-2.5-flash", 500_000, 100_000)
        assert usage.input_tokens == 500_000
        assert usage.output_tokens == 100_000
        assert usage.input_cost_usd == pytest.approx(0.075)
        assert usage.output_cost_usd == pytest.approx(0.06)
        assert usage.total_cost_usd == pytest.approx(0.135)

    def test_cost_tracker_with_gemini(self) -> None:
        """CostTracker correctly sums multiple Gemini calls."""
        from llm_gateway.cost import CostTracker, build_token_usage

        tracker = CostTracker()
        usage1 = build_token_usage("gemini-2.5-flash", 100_000, 50_000)
        usage2 = build_token_usage("gemini-2.5-flash", 200_000, 100_000)

        tracker.record(usage1)
        tracker.record(usage2)

        assert tracker.call_count == 2
        assert tracker.total_tokens == 450_000
        assert tracker.total_cost_usd == pytest.approx(
            usage1.total_cost_usd + usage2.total_cost_usd
        )


@pytest.mark.unit
class TestExtractUsageEdgeCases:
    """Edge case tests for _extract_usage static method."""

    def test_none_prompt_token_count(self) -> None:
        """Handles None prompt_token_count gracefully."""
        GeminiProvider = _import_gemini_provider()

        result = _TestModel(answer="none tokens")
        raw = MagicMock()
        raw.usage_metadata.prompt_token_count = None
        raw.usage_metadata.candidates_token_count = None
        result._raw_response = raw  # type: ignore[attr-defined]

        usage = GeminiProvider._extract_usage(result, "gemini-2.5-flash")
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0

    def test_zero_token_counts(self) -> None:
        """Zero token counts produce zero-cost usage."""
        GeminiProvider = _import_gemini_provider()

        result = _TestModel(answer="zero")
        raw = MagicMock()
        raw.usage_metadata.prompt_token_count = 0
        raw.usage_metadata.candidates_token_count = 0
        result._raw_response = raw  # type: ignore[attr-defined]

        usage = GeminiProvider._extract_usage(result, "gemini-2.5-flash")
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_cost_usd == 0.0
