"""E2E image generation tests — calls real provider APIs when --run-live is used.

Dry-run mode (default): uses FakeImageProvider, no API key needed.
Live mode: uses the provider specified by --image-provider (default: gemini_image).

Run examples:
    # Dry-run (no API calls)
    pytest tests/test_image_e2e.py -v

    # Live with Gemini
    pytest --run-live -m live tests/test_image_e2e.py -v

    # Live with custom prompt
    pytest --run-live -m live tests/test_image_e2e.py -v \
        --image-prompt "a cyberpunk city at night"

    # Live with OpenAI
    pytest --run-live -m live tests/test_image_e2e.py -v \
        --image-provider openai_image
"""

from __future__ import annotations

import base64
import logging
from collections.abc import Callable
from pathlib import Path

import pytest

from llm_gateway import GatewayConfig, ImageClient
from llm_gateway.types import ImageData

logger = logging.getLogger(__name__)


# ─── Dry-Run Tests (always run) ──────────────────────────────────


@pytest.mark.dry_run
class TestImageDryRun:
    """Image generation tests with FakeImageProvider — no real API calls."""

    @pytest.mark.asyncio
    async def test_generate_image_dry_run(self) -> None:
        """Dry-run image generation returns valid response."""
        config = GatewayConfig(
            image_provider="fake_image",
            trace_enabled=False,
            log_format="console",
        )
        async with ImageClient(config=config) as client:
            resp = await client.generate_image(prompt="a cat wearing a hat")

        assert len(resp.images) >= 1
        assert resp.provider == "fake_image"
        assert resp.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_generate_multiple_images_dry_run(self) -> None:
        """Dry-run generates requested number of images."""
        config = GatewayConfig(
            image_provider="fake_image",
            trace_enabled=False,
            log_format="console",
        )
        async with ImageClient(config=config) as client:
            resp = await client.generate_image(prompt="cats", num_images=3)

        assert len(resp.images) == 3

    @pytest.mark.asyncio
    async def test_cost_tracking_dry_run(self) -> None:
        """Cost tracking works in dry-run mode."""
        config = GatewayConfig(
            image_provider="fake_image",
            trace_enabled=False,
            log_format="console",
        )
        async with ImageClient(config=config) as client:
            await client.generate_image(prompt="test")
            assert client.call_count == 1
            summary = client.cost_summary()
            assert summary["call_count"] == 1


# ─── Live Tests (require --run-live) ─────────────────────────────


def _save_image(image: ImageData, path: Path) -> None:
    """Save an ImageData to disk as PNG."""
    if image.b64_json:
        path.write_bytes(base64.b64decode(image.b64_json))
        logger.info("Saved image to %s (%d bytes)", path, path.stat().st_size)
    elif image.url:
        logger.info("Image URL (not saved): %s", image.url)


@pytest.mark.live
@pytest.mark.image
class TestImageGenerationE2E:
    """E2E image generation tests — calls real provider APIs."""

    @pytest.mark.asyncio
    async def test_generate_single_image(
        self,
        make_live_image_client: Callable[[], ImageClient],
        image_prompt: str,
        test_output_dir: Path,
    ) -> None:
        """Generate a single image, verify response, save to disk."""
        async with make_live_image_client() as client:
            resp = await client.generate_image(prompt=image_prompt)

        assert len(resp.images) >= 1
        assert resp.images[0].b64_json is not None or resp.images[0].url is not None
        assert resp.provider in ("gemini_image", "openai_image")
        assert resp.latency_ms > 0
        assert resp.usage.total_cost_usd >= 0

        _save_image(resp.images[0], test_output_dir / "single_image.png")

    @pytest.mark.asyncio
    async def test_generate_multiple_images(
        self,
        make_live_image_client: Callable[[], ImageClient],
        image_prompt: str,
        test_output_dir: Path,
    ) -> None:
        """Generate 2 images, save both."""
        async with make_live_image_client() as client:
            resp = await client.generate_image(prompt=image_prompt, num_images=2)

        assert len(resp.images) >= 2
        for i, img in enumerate(resp.images):
            _save_image(img, test_output_dir / f"multi_image_{i}.png")

    @pytest.mark.asyncio
    async def test_custom_prompt(
        self,
        make_live_image_client: Callable[[], ImageClient],
        test_output_dir: Path,
    ) -> None:
        """Test with a specific prompt to verify prompt handling."""
        async with make_live_image_client() as client:
            resp = await client.generate_image(prompt="a red cube on a white background")

        assert len(resp.images) >= 1
        _save_image(resp.images[0], test_output_dir / "custom_prompt.png")

    @pytest.mark.asyncio
    async def test_cost_tracking(
        self,
        make_live_image_client: Callable[[], ImageClient],
        image_prompt: str,
    ) -> None:
        """Verify cost tracking works end-to-end."""
        async with make_live_image_client() as client:
            await client.generate_image(prompt=image_prompt)
            assert client.call_count == 1
            assert client.total_cost_usd >= 0
            summary = client.cost_summary()
            assert summary["call_count"] == 1
