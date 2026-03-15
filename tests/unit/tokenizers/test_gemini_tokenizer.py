"""Tests for GeminiTokenizer (heuristic fallback path only — SDK is mocked)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from llm_gateway.tokenizers.gemini_tokenizer import GeminiTokenizer


@pytest.mark.unit
class TestGeminiTokenizerHeuristic:
    """Tests for the Gemini tokenizer when google-genai SDK is unavailable."""

    def test_empty_string_returns_zero(self) -> None:
        """Empty string should return 0 tokens."""
        tok = GeminiTokenizer()
        tok._use_heuristic = True
        tok._initialized = True
        assert tok.count_tokens("") == 0

    def test_heuristic_calculation(self) -> None:
        """Heuristic should match max(1, int(len(text) / 3.5))."""
        tok = GeminiTokenizer()
        tok._use_heuristic = True
        tok._initialized = True

        text = "a" * 35  # 35 / 3.5 = 10
        assert tok.count_tokens(text) == 10

    def test_heuristic_short_text_at_least_one(self) -> None:
        """Short text should return at least 1 token via heuristic."""
        tok = GeminiTokenizer()
        tok._use_heuristic = True
        tok._initialized = True

        assert tok.count_tokens("Hi") >= 1

    def test_heuristic_non_round_result(self) -> None:
        """Non-round division should truncate (int, not round)."""
        tok = GeminiTokenizer()
        tok._use_heuristic = True
        tok._initialized = True

        # 10 / 3.5 = 2.857... → int = 2
        text = "a" * 10
        assert tok.count_tokens(text) == 2

    def test_name_property(self) -> None:
        """Name property should return 'gemini'."""
        tok = GeminiTokenizer()
        assert tok.name == "gemini"

    def test_default_model(self) -> None:
        """Default model should be gemini-2.5-flash."""
        tok = GeminiTokenizer()
        assert tok._model_name == "gemini-2.5-flash"

    def test_custom_model(self) -> None:
        """Custom model should be stored."""
        tok = GeminiTokenizer(model="gemini-2.5-pro")
        assert tok._model_name == "gemini-2.5-pro"


@pytest.mark.unit
class TestGeminiTokenizerSDKFallback:
    """Tests for graceful fallback when SDK import fails or API call fails."""

    def test_sdk_count_tokens_exception_falls_back(self) -> None:
        """When SDK count_tokens raises, should fall back to heuristic."""
        tok = GeminiTokenizer()
        tok._initialized = True
        tok._use_heuristic = False

        mock_client = MagicMock()
        mock_client.models.count_tokens.side_effect = RuntimeError("API error")
        tok._client = mock_client

        # Mock the google.genai import inside count_tokens
        mock_genai = MagicMock()
        mock_genai.Client = MagicMock
        with patch.dict(sys.modules, {"google": MagicMock(genai=mock_genai)}):
            result = tok.count_tokens("a" * 35)
            assert result == 10  # Falls back to 35 / 3.5 = 10

    def test_sdk_success_returns_exact_count(self) -> None:
        """When SDK works, should return exact token count from API."""
        tok = GeminiTokenizer()
        tok._initialized = True
        tok._use_heuristic = False

        mock_result = MagicMock()
        mock_result.total_tokens = 42

        mock_client = MagicMock()
        mock_client.models.count_tokens.return_value = mock_result
        tok._client = mock_client

        mock_genai = MagicMock()
        mock_genai.Client = MagicMock
        with patch.dict(sys.modules, {"google": MagicMock(genai=mock_genai)}):
            result = tok.count_tokens("some text")
            assert result == 42

    def test_init_client_sdk_failure(self) -> None:
        """_init_client should set _use_heuristic=True on SDK failure."""
        tok = GeminiTokenizer()
        assert tok._initialized is False

        mock_google = MagicMock()
        mock_google.genai.Client.side_effect = RuntimeError("no API key")
        with patch.dict(sys.modules, {"google": mock_google, "google.genai": mock_google.genai}):
            tok._init_client()
            assert tok._initialized is True
            assert tok._use_heuristic is True

    def test_init_client_success(self) -> None:
        """_init_client should set client when SDK is available."""
        tok = GeminiTokenizer()

        mock_client_instance = MagicMock()
        mock_google = MagicMock()
        mock_google.genai.Client.return_value = mock_client_instance
        with patch.dict(sys.modules, {"google": mock_google, "google.genai": mock_google.genai}):
            tok._init_client()
            assert tok._initialized is True
            assert tok._use_heuristic is False
            assert tok._client is mock_client_instance

    def test_init_client_idempotent(self) -> None:
        """Calling _init_client twice should not re-initialize."""
        tok = GeminiTokenizer()
        tok._initialized = True
        tok._use_heuristic = True
        tok._init_client()  # Should be a no-op
        assert tok._use_heuristic is True

    def test_empty_string_skips_sdk(self) -> None:
        """Empty string should return 0 without calling SDK."""
        tok = GeminiTokenizer()
        tok._initialized = True
        tok._use_heuristic = False
        mock_client = MagicMock()
        tok._client = mock_client
        assert tok.count_tokens("") == 0
        mock_client.models.count_tokens.assert_not_called()
