"""Testing utilities shipped with llm-gateway.

Provides ``FakeLLMProvider`` and ``FakeImageProvider`` for consumers to
use in their test suites without reimplementing the provider Protocols.

Usage::

    from llm_gateway import LLMClient, FakeLLMProvider
    from pydantic import BaseModel

    class Answer(BaseModel):
        text: str

    fake = FakeLLMProvider()
    fake.set_response(Answer, Answer(text="42"))

    async with LLMClient(provider_instance=fake) as client:
        resp = await client.complete(
            messages=[{"role": "user", "content": "What is 6*7?"}],
            response_model=Answer,
        )
        assert resp.content.text == "42"
        assert fake.call_count == 1

Image generation::

    from llm_gateway import ImageClient, FakeImageProvider

    fake_img = FakeImageProvider()
    async with ImageClient(provider_instance=fake_img) as client:
        resp = await client.generate_image("a cat")
        assert len(resp.images) == 1
        assert fake_img.call_count == 1
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from llm_gateway.cost import build_token_usage
from llm_gateway.exceptions import ResponseValidationError
from llm_gateway.types import (
    ImageData,
    ImageGenerationResponse,
    ImageTokenUsage,
    LLMMessage,
    LLMResponse,
)

if TYPE_CHECKING:
    from llm_gateway.config import GatewayConfig

T = TypeVar("T")


@dataclass
class FakeCall:
    """Record of a single ``FakeLLMProvider.complete()`` invocation."""

    messages: Sequence[LLMMessage]
    response_model: type
    model: str
    response: object


class FakeLLMProvider:
    """Fake LLM provider for testing. Implements the ``LLMProvider`` Protocol.

    Two modes:

    1. **Pre-configured**: call ``set_response(ModelClass, instance)`` before
       ``complete()``.
    2. **Dynamic**: pass a ``response_factory`` callable to the constructor.

    Resolution order in ``complete()``:

    1. Pre-configured via ``set_response()`` (exact type match)
    2. ``response_factory`` callable (if provided)
    3. Raise ``ResponseValidationError`` (no response configured)
    """

    def __init__(
        self,
        response_factory: Callable[[type[T], Sequence[LLMMessage]], T] | None = None,
        default_input_tokens: int = 100,
        default_output_tokens: int = 50,
    ) -> None:
        self._responses: dict[type, object] = {}
        self._response_factory = response_factory
        self._default_input_tokens = default_input_tokens
        self._default_output_tokens = default_output_tokens
        self.calls: list[FakeCall] = []

    def set_response(self, response_model: type[T], response: T) -> None:
        """Pre-configure a response for a specific ``response_model`` class."""
        self._responses[response_model] = response

    DEFAULT_MODEL = "fake-model"

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        response_model: type[T],
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        image_files: Sequence[str] | None = None,
    ) -> LLMResponse[T]:
        """Return pre-configured or factory-built response.

        Args:
            messages: Conversation messages.
            response_model: Pydantic model class for structured output.
            model: Model identifier. ``None`` uses ``DEFAULT_MODEL``.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature.
            image_files: Accepted for protocol compatibility; ignored by fake.

        Returns:
            ``LLMResponse[T]`` with configurable ``TokenUsage``.

        Raises:
            ResponseValidationError: If no response is configured and no factory
                is provided.
        """
        effective_model = model or self.DEFAULT_MODEL
        content: T | None = None

        # 1. Pre-configured response (exact type match)
        preconfigured = self._responses.get(response_model)
        if preconfigured is not None:
            content = preconfigured  # type: ignore[assignment]

        # 2. response_factory callable
        if content is None and self._response_factory is not None:
            content = self._response_factory(response_model, messages)  # type: ignore[assignment,arg-type]

        # 3. No response available
        if content is None:
            raise ResponseValidationError(
                model_name=response_model.__name__,
                reason="No fake response configured. "
                "Use set_response() or pass a response_factory.",
            )

        usage = build_token_usage(
            effective_model,
            self._default_input_tokens,
            self._default_output_tokens,
        )

        response = LLMResponse(
            content=content,
            usage=usage,
            model=effective_model,
            provider="fake",
            latency_ms=0.0,
        )

        self.calls.append(
            FakeCall(
                messages=messages,
                response_model=response_model,
                model=effective_model,
                response=content,
            )
        )

        return response

    @property
    def call_count(self) -> int:
        """Number of ``complete()`` calls recorded."""
        return len(self.calls)

    async def close(self) -> None:
        """No-op cleanup."""

    @classmethod
    def from_config(cls, config: GatewayConfig) -> FakeLLMProvider:
        """Factory for provider registry. Creates an empty ``FakeLLMProvider``."""
        return cls()


# ── Fake Image Provider ─────────────────────────────────────────


@dataclass
class FakeImageCall:
    """Record of a single ``FakeImageProvider.generate_image()`` invocation."""

    prompt: str
    model: str
    num_images: int
    quality: str
    width: int | None
    height: int | None


class FakeImageProvider:
    """Fake image provider for testing. Implements ``ImageGenerationProvider`` Protocol.

    By default, returns a placeholder ImageData with a fake URL.
    Use ``set_response()`` to configure custom responses.
    """

    DEFAULT_MODEL = "fake-image-model"

    def __init__(
        self,
        default_cost_usd: float = 0.0,
    ) -> None:
        self._custom_response: ImageGenerationResponse | None = None
        self._default_cost_usd = default_cost_usd
        self.calls: list[FakeImageCall] = []

    def set_response(self, response: ImageGenerationResponse) -> None:
        """Pre-configure a custom response for all generate_image calls."""
        self._custom_response = response

    async def generate_image(
        self,
        prompt: str,
        model: str | None = None,
        width: int | None = None,
        height: int | None = None,
        num_images: int = 1,
        quality: str = "standard",
    ) -> ImageGenerationResponse:
        """Return pre-configured or default placeholder response.

        Args:
            prompt: Text description (recorded in calls).
            model: Model identifier.
            width: Image width.
            height: Image height.
            num_images: Number of images to generate.
            quality: Quality tier.

        Returns:
            ImageGenerationResponse with fake image data.
        """
        effective_model = model or self.DEFAULT_MODEL

        self.calls.append(
            FakeImageCall(
                prompt=prompt,
                model=effective_model,
                num_images=num_images,
                quality=quality,
                width=width,
                height=height,
            )
        )

        if self._custom_response is not None:
            return self._custom_response

        images = [
            ImageData(
                url=f"https://fake-image-provider.test/image_{i}.png",
                revised_prompt=prompt,
            )
            for i in range(num_images)
        ]

        return ImageGenerationResponse(
            images=images,
            usage=ImageTokenUsage(
                prompt_tokens=len(prompt.split()),
                total_cost_usd=self._default_cost_usd,
            ),
            model=effective_model,
            provider="fake_image",
            latency_ms=0.0,
        )

    @property
    def call_count(self) -> int:
        """Number of ``generate_image()`` calls recorded."""
        return len(self.calls)

    async def close(self) -> None:
        """No-op cleanup."""

    @classmethod
    def from_config(cls, config: GatewayConfig) -> FakeImageProvider:
        """Factory for provider registry."""
        return cls()
