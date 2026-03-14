"""Token pricing registry and cost tracking."""

from __future__ import annotations

import logging
from typing import Any

from llm_gateway.exceptions import CostLimitExceededError
from llm_gateway.types import ImageTokenUsage, TokenUsage

logger = logging.getLogger(__name__)

# ── Pricing Registry (USD per 1 million tokens) ────────────────
_PRICING: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-5-20250514": {"input": 3.00, "output": 15.00},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    # OpenAI (examples — update as needed)
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    # Gemini 3.1
    "gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00},
    "gemini-3.1-flash-lite-preview": {"input": 0.25, "output": 1.50},
    # Gemini 3
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
    # Gemini 2.5
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    # Gemini 2.0
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    # Gemini 1.5
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash-8b": {"input": 0.0375, "output": 0.15},
}


def register_pricing(model: str, input_per_1m: float, output_per_1m: float) -> None:
    """Register or update pricing for a model.

    Args:
        model: Model identifier string.
        input_per_1m: Cost in USD per 1M input tokens.
        output_per_1m: Cost in USD per 1M output tokens.
    """
    _PRICING[model] = {"input": input_per_1m, "output": output_per_1m}


def get_pricing(model: str) -> dict[str, float] | None:
    """Return pricing dict for a model, or None if unknown."""
    return _PRICING.get(model)


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> tuple[float, float]:
    """Calculate USD cost for a given token count.

    Returns:
        Tuple of (input_cost_usd, output_cost_usd). Both 0.0 if model unknown.
    """
    pricing = _PRICING.get(model)
    if pricing is None:
        return 0.0, 0.0
    input_cost = input_tokens * pricing["input"] / 1_000_000
    output_cost = output_tokens * pricing["output"] / 1_000_000
    return input_cost, output_cost


def build_token_usage(model: str, input_tokens: int, output_tokens: int) -> TokenUsage:
    """Build a TokenUsage with cost calculated from the pricing registry."""
    input_cost, output_cost = calculate_cost(model, input_tokens, output_tokens)
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
    )


class CostTracker:
    """Accumulates token usage and cost across multiple LLM calls.

    Supports cost guardrails (warn and hard limit).
    """

    def __init__(
        self,
        cost_limit_usd: float | None = None,
        cost_warn_usd: float | None = None,
    ) -> None:
        self._cost_limit = cost_limit_usd
        self._cost_warn = cost_warn_usd
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cost_usd: float = 0.0
        self._call_count: int = 0
        self._warned: bool = False

    def record(self, usage: TokenUsage) -> None:
        """Record a single LLM call's usage and check guardrails."""
        self._total_input_tokens += usage.input_tokens
        self._total_output_tokens += usage.output_tokens
        self._total_cost_usd += usage.total_cost_usd
        self._call_count += 1

        self._check_guardrails()

    def _check_guardrails(self) -> None:
        """Enforce cost warning and hard limit."""
        if self._cost_warn and not self._warned and self._total_cost_usd >= self._cost_warn:
            self._warned = True
            logger.warning(
                "LLM cost warning threshold reached: $%.4f >= $%.4f",
                self._total_cost_usd,
                self._cost_warn,
            )

        if self._cost_limit and self._total_cost_usd >= self._cost_limit:
            raise CostLimitExceededError(self._total_cost_usd, self._cost_limit)

    @property
    def total_cost_usd(self) -> float:
        """Cumulative cost in USD."""
        return self._total_cost_usd

    @property
    def total_tokens(self) -> int:
        """Cumulative total tokens."""
        return self._total_input_tokens + self._total_output_tokens

    @property
    def call_count(self) -> int:
        """Number of LLM calls recorded."""
        return self._call_count

    def summary(self) -> dict[str, Any]:
        """Return a summary dict suitable for logging or span attributes."""
        return {
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self._total_cost_usd, 6),
            "call_count": self._call_count,
        }

    def reset(self) -> None:
        """Reset all accumulators."""
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost_usd = 0.0
        self._call_count = 0
        self._warned = False


# ── Image Pricing Registry ─────────────────────────────────────

# model → quality → size → cost_per_image_usd
_IMAGE_PRICING: dict[str, dict[str, dict[str, float]]] = {
    "gpt-image-1": {
        "low": {"1024x1024": 0.011, "1024x1536": 0.016, "1536x1024": 0.016, "auto": 0.011},
        "medium": {"1024x1024": 0.042, "1024x1536": 0.063, "1536x1024": 0.063, "auto": 0.042},
        "high": {"1024x1024": 0.167, "1024x1536": 0.190, "1536x1024": 0.190, "auto": 0.167},
    },
    "dall-e-3": {
        "standard": {"1024x1024": 0.040, "1024x1792": 0.080, "1792x1024": 0.080, "auto": 0.040},
        "hd": {"1024x1024": 0.080, "1024x1792": 0.120, "1792x1024": 0.120, "auto": 0.080},
    },
    "dall-e-2": {
        "standard": {"256x256": 0.016, "512x512": 0.018, "1024x1024": 0.020, "auto": 0.020},
    },
    # Google Imagen
    "imagen-3.0-generate-002": {
        "standard": {"auto": 0.04, "1024x1024": 0.04},
    },
    "imagen-3.0-fast-generate-001": {
        "standard": {"auto": 0.02, "1024x1024": 0.02},
    },
    "imagen-4.0-generate-001": {
        "standard": {"auto": 0.04, "1024x1024": 0.04},
    },
    "imagen-4.0-ultra-generate-001": {
        "standard": {"auto": 0.08, "1024x1024": 0.08},
    },
    "imagen-4.0-fast-generate-001": {
        "standard": {"auto": 0.02, "1024x1024": 0.02},
    },
}


def register_image_pricing(
    model: str,
    quality: str,
    size: str,
    cost_per_image_usd: float,
) -> None:
    """Register or update pricing for an image model.

    Args:
        model: Model identifier (e.g. "gpt-image-1").
        quality: Quality tier (e.g. "standard", "hd", "low", "medium", "high").
        size: Image size (e.g. "1024x1024").
        cost_per_image_usd: Cost per generated image in USD.
    """
    if model not in _IMAGE_PRICING:
        _IMAGE_PRICING[model] = {}
    if quality not in _IMAGE_PRICING[model]:
        _IMAGE_PRICING[model][quality] = {}
    _IMAGE_PRICING[model][quality][size] = cost_per_image_usd


def calculate_image_cost(
    model: str,
    quality: str = "standard",
    size: str = "auto",
    num_images: int = 1,
) -> float:
    """Calculate cost for image generation.

    Returns:
        Total cost in USD. 0.0 if model/quality/size unknown.
    """
    model_pricing = _IMAGE_PRICING.get(model)
    if model_pricing is None:
        return 0.0
    quality_pricing = model_pricing.get(quality)
    if quality_pricing is None:
        return 0.0
    per_image = quality_pricing.get(size, quality_pricing.get("auto", 0.0))
    return per_image * num_images


def build_image_usage(
    model: str,
    quality: str = "standard",
    size: str = "auto",
    num_images: int = 1,
    prompt_tokens: int = 0,
) -> ImageTokenUsage:
    """Build an ImageTokenUsage with cost from the pricing registry."""
    cost = calculate_image_cost(model, quality, size, num_images)
    return ImageTokenUsage(prompt_tokens=prompt_tokens, total_cost_usd=cost)


class ImageCostTracker:
    """Accumulates image generation costs across multiple calls.

    Supports cost guardrails (warn and hard limit).
    """

    def __init__(
        self,
        cost_limit_usd: float | None = None,
        cost_warn_usd: float | None = None,
    ) -> None:
        self._cost_limit = cost_limit_usd
        self._cost_warn = cost_warn_usd
        self._total_cost_usd: float = 0.0
        self._total_images: int = 0
        self._call_count: int = 0
        self._warned: bool = False

    def record(self, usage: ImageTokenUsage) -> None:
        """Record a single image generation call's usage and check guardrails."""
        self._total_cost_usd += usage.total_cost_usd
        self._call_count += 1
        self._check_guardrails()

    def record_images(self, count: int) -> None:
        """Track total images generated."""
        self._total_images += count

    def _check_guardrails(self) -> None:
        """Enforce cost warning and hard limit."""
        if self._cost_warn and not self._warned and self._total_cost_usd >= self._cost_warn:
            self._warned = True
            logger.warning(
                "Image cost warning threshold reached: $%.4f >= $%.4f",
                self._total_cost_usd,
                self._cost_warn,
            )

        if self._cost_limit and self._total_cost_usd >= self._cost_limit:
            raise CostLimitExceededError(self._total_cost_usd, self._cost_limit)

    @property
    def total_cost_usd(self) -> float:
        """Cumulative cost in USD."""
        return self._total_cost_usd

    @property
    def total_images(self) -> int:
        """Cumulative images generated."""
        return self._total_images

    @property
    def call_count(self) -> int:
        """Number of image generation calls recorded."""
        return self._call_count

    def summary(self) -> dict[str, Any]:
        """Return a summary dict suitable for logging or span attributes."""
        return {
            "total_cost_usd": round(self._total_cost_usd, 6),
            "total_images": self._total_images,
            "call_count": self._call_count,
        }

    def reset(self) -> None:
        """Reset all accumulators."""
        self._total_cost_usd = 0.0
        self._total_images = 0
        self._call_count = 0
        self._warned = False
