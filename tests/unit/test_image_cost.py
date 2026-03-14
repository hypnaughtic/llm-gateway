"""Unit tests for image cost tracking."""

from __future__ import annotations

import pytest

from llm_gateway.cost import (
    ImageCostTracker,
    build_image_usage,
    calculate_image_cost,
    register_image_pricing,
)
from llm_gateway.exceptions import CostLimitExceededError
from llm_gateway.types import ImageTokenUsage


@pytest.mark.unit
class TestImagePricing:
    """Tests for image pricing registry."""

    def test_known_model_cost(self) -> None:
        """Known model returns correct cost."""
        cost = calculate_image_cost("gpt-image-1", quality="low", size="1024x1024")
        assert cost == pytest.approx(0.011)

    def test_known_model_hd_cost(self) -> None:
        """HD quality costs more."""
        cost = calculate_image_cost("dall-e-3", quality="hd", size="1024x1024")
        assert cost == pytest.approx(0.080)

    def test_multiple_images(self) -> None:
        """Multiple images multiply cost."""
        cost = calculate_image_cost("gpt-image-1", quality="low", size="1024x1024", num_images=3)
        assert cost == pytest.approx(0.033)

    def test_unknown_model_returns_zero(self) -> None:
        """Unknown model returns 0.0."""
        cost = calculate_image_cost("unknown-model")
        assert cost == 0.0

    def test_unknown_quality_returns_zero(self) -> None:
        """Unknown quality for known model returns 0.0."""
        cost = calculate_image_cost("gpt-image-1", quality="ultra")
        assert cost == 0.0

    def test_auto_size_fallback(self) -> None:
        """Auto size falls back correctly."""
        cost = calculate_image_cost("gpt-image-1", quality="low", size="auto")
        assert cost == pytest.approx(0.011)

    def test_register_custom_pricing(self) -> None:
        """Custom pricing can be registered."""
        register_image_pricing("custom-img", "standard", "512x512", 0.05)
        cost = calculate_image_cost("custom-img", quality="standard", size="512x512")
        assert cost == pytest.approx(0.05)

    def test_imagen_standard_cost(self) -> None:
        """Imagen 3.0 generate returns $0.04."""
        cost = calculate_image_cost("imagen-3.0-generate-002", quality="standard", size="auto")
        assert cost == pytest.approx(0.04)

    def test_imagen_fast_cost(self) -> None:
        """Imagen 3.0 fast returns $0.02."""
        cost = calculate_image_cost(
            "imagen-3.0-fast-generate-001", quality="standard", size="auto"
        )
        assert cost == pytest.approx(0.02)

    def test_build_image_usage(self) -> None:
        """build_image_usage creates correct ImageTokenUsage."""
        usage = build_image_usage("gpt-image-1", quality="low", size="1024x1024", num_images=2)
        assert isinstance(usage, ImageTokenUsage)
        assert usage.total_cost_usd == pytest.approx(0.022)


@pytest.mark.unit
class TestImageCostTracker:
    """Tests for ImageCostTracker."""

    def test_record_accumulates(self) -> None:
        """Costs accumulate across records."""
        tracker = ImageCostTracker()
        tracker.record(ImageTokenUsage(total_cost_usd=0.04))
        tracker.record(ImageTokenUsage(total_cost_usd=0.08))
        assert tracker.total_cost_usd == pytest.approx(0.12)
        assert tracker.call_count == 2

    def test_cost_limit_raises(self) -> None:
        """Exceeding cost limit raises CostLimitExceededError."""
        tracker = ImageCostTracker(cost_limit_usd=0.10)
        tracker.record(ImageTokenUsage(total_cost_usd=0.05))
        with pytest.raises(CostLimitExceededError):
            tracker.record(ImageTokenUsage(total_cost_usd=0.06))

    def test_summary(self) -> None:
        """Summary returns structured dict."""
        tracker = ImageCostTracker()
        tracker.record(ImageTokenUsage(total_cost_usd=0.04))
        s = tracker.summary()
        assert s["call_count"] == 1
        assert s["total_cost_usd"] == pytest.approx(0.04)

    def test_reset(self) -> None:
        """Reset clears all accumulators."""
        tracker = ImageCostTracker()
        tracker.record(ImageTokenUsage(total_cost_usd=0.04))
        tracker.reset()
        assert tracker.total_cost_usd == 0.0
        assert tracker.call_count == 0

    def test_record_images(self) -> None:
        """Image count tracking works."""
        tracker = ImageCostTracker()
        tracker.record_images(3)
        tracker.record_images(2)
        assert tracker.total_images == 5
