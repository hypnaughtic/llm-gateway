"""Google Gemini (Imagen) image generation provider."""

from __future__ import annotations

import base64
import time
from typing import TYPE_CHECKING, Any

from tenacity import retry, stop_after_attempt, wait_exponential

from llm_gateway.cost import build_image_usage
from llm_gateway.exceptions import ProviderError
from llm_gateway.types import ImageData, ImageGenerationResponse

if TYPE_CHECKING:
    from google import genai

    from llm_gateway.config import GatewayConfig


class GeminiImageProvider:
    """Image generation via Google's Imagen API (google-genai SDK)."""

    DEFAULT_MODEL = "imagen-4.0-generate-001"

    def __init__(
        self,
        api_key: str,
        max_retries: int = 3,
        timeout_seconds: int = 120,
    ) -> None:
        try:
            from google import genai as _genai
        except ImportError as exc:
            msg = "google-genai package required: pip install 'llm-gateway[gemini]'"
            raise ImportError(msg) from exc

        self._client: genai.Client = _genai.Client(api_key=api_key)
        self._max_retries = max_retries
        self._timeout_seconds = timeout_seconds

    @classmethod
    def from_config(cls, config: GatewayConfig) -> GeminiImageProvider:
        """Factory for provider registry."""
        return cls(
            api_key=config.get_image_api_key(),
            max_retries=config.max_retries,
            timeout_seconds=config.timeout_seconds,
        )

    async def generate_image(
        self,
        prompt: str,
        model: str | None = None,
        width: int | None = None,
        height: int | None = None,
        num_images: int = 1,
        quality: str = "standard",
    ) -> ImageGenerationResponse:
        """Generate images via Google Imagen API.

        Args:
            prompt: Text description of the desired image.
            model: Model identifier (imagen-3.0-generate-002, etc.).
            width: Image width in pixels (not used by Imagen — ignored).
            height: Image height in pixels (not used by Imagen — ignored).
            num_images: Number of images to generate.
            quality: Quality tier (passed to cost calculation).

        Returns:
            ImageGenerationResponse with generated image data and cost.
        """
        from google.genai import types as genai_types

        effective_model = model or self.DEFAULT_MODEL

        start = time.monotonic()

        config = genai_types.GenerateImagesConfig(
            number_of_images=num_images,
        )

        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        )
        async def _do_call() -> Any:
            return await self._client.aio.models.generate_images(
                model=effective_model,
                prompt=prompt,
                config=config,
            )

        try:
            result = await _do_call()
        except Exception as exc:
            raise ProviderError("gemini_image", exc) from exc

        latency_ms = (time.monotonic() - start) * 1000

        images: list[ImageData] = []
        if result.generated_images:
            for img in result.generated_images:
                b64 = base64.b64encode(img.image.image_bytes).decode("ascii")
                images.append(
                    ImageData(
                        b64_json=b64,
                        revised_prompt="",
                    )
                )

        usage = build_image_usage(
            model=effective_model,
            quality=quality,
            size="auto",
            num_images=len(images),
        )

        return ImageGenerationResponse(
            images=images,
            usage=usage,
            model=effective_model,
            provider="gemini_image",
            latency_ms=latency_ms,
        )

    async def close(self) -> None:
        """No-op — google-genai Client has no close method."""
