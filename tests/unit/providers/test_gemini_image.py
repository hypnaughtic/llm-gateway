"""Unit tests for GeminiImageProvider."""

from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_gateway.exceptions import ProviderError
from llm_gateway.types import ImageGenerationResponse


def _make_generated_image(image_bytes: bytes) -> SimpleNamespace:
    """Create a fake generated image object matching google-genai response shape."""
    return SimpleNamespace(image=SimpleNamespace(image_bytes=image_bytes))


def _make_generate_result(images: list[SimpleNamespace]) -> SimpleNamespace:
    """Create a fake generate_images result."""
    return SimpleNamespace(generated_images=images)


@pytest.mark.unit
class TestGeminiImageProvider:
    """Tests for GeminiImageProvider with mocked google-genai Client."""

    def _make_provider(self, mock_client: MagicMock) -> Any:
        """Create a GeminiImageProvider with an injected mock client."""
        with patch(
            "llm_gateway.providers.gemini_image.GeminiImageProvider.__init__", lambda s, **kw: None
        ):
            from llm_gateway.providers.gemini_image import GeminiImageProvider

            provider = GeminiImageProvider.__new__(GeminiImageProvider)
            provider._client = mock_client
            provider._max_retries = 3
            provider._timeout_seconds = 120
            return provider

    def _make_mock_client(self, result: SimpleNamespace) -> MagicMock:
        """Create a mock google-genai Client with async generate_images."""
        mock = MagicMock()
        mock.aio.models.generate_images = AsyncMock(return_value=result)
        return mock

    @pytest.mark.asyncio
    async def test_single_image_response(self) -> None:
        """Single image is returned with base64 encoding."""
        raw_bytes = b"\x89PNG\r\n\x1a\nfake-image-data"
        result = _make_generate_result([_make_generated_image(raw_bytes)])
        mock_client = self._make_mock_client(result)
        provider = self._make_provider(mock_client)

        resp = await provider.generate_image(prompt="a sunset")

        assert isinstance(resp, ImageGenerationResponse)
        assert len(resp.images) == 1
        assert resp.images[0].b64_json == base64.b64encode(raw_bytes).decode("ascii")
        assert resp.images[0].url is None
        assert resp.provider == "gemini_image"
        assert resp.model == "imagen-4.0-generate-001"
        assert resp.latency_ms > 0

    @pytest.mark.asyncio
    async def test_multiple_images(self) -> None:
        """Multiple images are all returned."""
        imgs = [
            _make_generated_image(b"image-data-0"),
            _make_generated_image(b"image-data-1"),
            _make_generated_image(b"image-data-2"),
        ]
        result = _make_generate_result(imgs)
        mock_client = self._make_mock_client(result)
        provider = self._make_provider(mock_client)

        resp = await provider.generate_image(prompt="cats", num_images=3)

        assert len(resp.images) == 3
        for i, img in enumerate(resp.images):
            expected = base64.b64encode(f"image-data-{i}".encode()).decode("ascii")
            assert img.b64_json == expected

    @pytest.mark.asyncio
    async def test_custom_model(self) -> None:
        """Custom model is passed through to the API."""
        result = _make_generate_result([_make_generated_image(b"data")])
        mock_client = self._make_mock_client(result)
        provider = self._make_provider(mock_client)

        resp = await provider.generate_image(
            prompt="test",
            model="imagen-3.0-fast-generate-001",
        )

        assert resp.model == "imagen-3.0-fast-generate-001"
        call_kwargs = mock_client.aio.models.generate_images.call_args
        assert call_kwargs.kwargs["model"] == "imagen-3.0-fast-generate-001"

    @pytest.mark.asyncio
    async def test_default_model(self) -> None:
        """Default model is imagen-3.0-generate-002."""
        from llm_gateway.providers.gemini_image import GeminiImageProvider

        assert GeminiImageProvider.DEFAULT_MODEL == "imagen-4.0-generate-001"

    @pytest.mark.asyncio
    async def test_error_wrapping(self) -> None:
        """Provider errors are wrapped in ProviderError."""
        mock_client = MagicMock()
        mock_client.aio.models.generate_images = AsyncMock(
            side_effect=RuntimeError("API quota exceeded")
        )
        provider = self._make_provider(mock_client)

        with pytest.raises(ProviderError, match="gemini_image"):
            await provider.generate_image(prompt="test")

    @pytest.mark.asyncio
    async def test_empty_result(self) -> None:
        """Empty generated_images returns empty list."""
        result = SimpleNamespace(generated_images=None)
        mock_client = self._make_mock_client(result)
        provider = self._make_provider(mock_client)

        resp = await provider.generate_image(prompt="test")

        assert resp.images == []

    @pytest.mark.asyncio
    async def test_cost_tracking(self) -> None:
        """Usage includes cost from pricing registry."""
        result = _make_generate_result([_make_generated_image(b"data")])
        mock_client = self._make_mock_client(result)
        provider = self._make_provider(mock_client)

        resp = await provider.generate_image(prompt="test")

        assert resp.usage.total_cost_usd == pytest.approx(0.04)
