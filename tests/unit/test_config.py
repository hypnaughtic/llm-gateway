"""Tests for GatewayConfig."""

from __future__ import annotations

import pytest

from llm_gateway.config import GatewayConfig


@pytest.mark.unit
class TestGatewayConfig:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default config loads without errors."""
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config = GatewayConfig()
        assert config.provider == "anthropic"
        assert config.max_tokens == 4096

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLM_ prefixed env vars override defaults."""
        monkeypatch.setenv("LLM_PROVIDER", "local_claude")
        monkeypatch.setenv("LLM_MODEL", "custom-model")
        monkeypatch.setenv("LLM_MAX_TOKENS", "2048")
        config = GatewayConfig()
        assert config.provider == "local_claude"
        assert config.model == "custom-model"
        assert config.max_tokens == 2048

    def test_api_key_fallback_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Falls back to ANTHROPIC_API_KEY when LLM_API_KEY is not set."""
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        config = GatewayConfig()
        assert config.get_api_key() == "sk-ant-test"

    def test_api_key_fallback_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Falls back to OPENAI_API_KEY for openai provider."""
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        config = GatewayConfig()
        assert config.get_api_key() == "sk-openai-test"

    def test_get_api_key_raises_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ValueError when no API key is configured."""
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config = GatewayConfig()
        with pytest.raises(ValueError, match="No API key"):
            config.get_api_key()

    def test_image_api_key_direct(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLM_IMAGE_API_KEY resolves directly."""
        monkeypatch.setenv("LLM_IMAGE_API_KEY", "img-key-direct")
        monkeypatch.setenv("LLM_IMAGE_PROVIDER", "gemini_image")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        config = GatewayConfig()
        assert config.get_image_api_key() == "img-key-direct"

    def test_image_api_key_fallback_gemini(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GEMINI_API_KEY resolves for gemini_image provider."""
        monkeypatch.delenv("LLM_IMAGE_API_KEY", raising=False)
        monkeypatch.setenv("LLM_IMAGE_PROVIDER", "gemini_image")
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-123")
        config = GatewayConfig()
        assert config.get_image_api_key() == "gemini-key-123"

    def test_image_api_key_fallback_google(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GOOGLE_API_KEY as secondary fallback for gemini_image."""
        monkeypatch.delenv("LLM_IMAGE_API_KEY", raising=False)
        monkeypatch.setenv("LLM_IMAGE_PROVIDER", "gemini_image")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "google-key-456")
        config = GatewayConfig()
        assert config.get_image_api_key() == "google-key-456"

    def test_image_api_key_fallback_openai_image(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OPENAI_API_KEY resolves for openai_image provider."""
        monkeypatch.delenv("LLM_IMAGE_API_KEY", raising=False)
        monkeypatch.setenv("LLM_IMAGE_PROVIDER", "openai_image")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-img")
        config = GatewayConfig()
        assert config.get_image_api_key() == "sk-openai-img"

    def test_get_image_api_key_raises_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ValueError when no image API key is configured."""
        monkeypatch.delenv("LLM_IMAGE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("LLM_IMAGE_PROVIDER", "gemini_image")
        config = GatewayConfig()
        with pytest.raises(ValueError, match="No image API key"):
            config.get_image_api_key()

    def test_cost_guardrails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_COST_LIMIT_USD", "10.0")
        monkeypatch.setenv("LLM_COST_WARN_USD", "5.0")
        config = GatewayConfig()
        assert config.cost_limit_usd == 10.0
        assert config.cost_warn_usd == 5.0
