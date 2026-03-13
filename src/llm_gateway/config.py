"""Gateway configuration via environment variables."""

from __future__ import annotations

import os

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings


class GatewayConfig(BaseSettings):
    """LLM Gateway configuration.

    All fields are read from environment variables with the ``LLM_`` prefix.
    Example: ``LLM_PROVIDER=anthropic`` sets ``provider="anthropic"``.
    """

    model_config = {"env_prefix": "LLM_", "env_file": ".env", "extra": "ignore"}

    # ── Provider ────────────────────────────────────────────────
    provider: str = Field(
        default="anthropic",
        description="Provider name: 'anthropic', 'local_claude', 'openai', etc.",
    )
    model: str | None = Field(
        default=None,
        description=(
            "Model identifier passed to the provider. "
            "When unset, each provider uses its own default."
        ),
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="API key. Falls back to provider-specific env vars if unset.",
    )
    base_url: str | None = Field(
        default=None,
        description="Optional base URL override for the provider API.",
    )

    # ── Image Provider ─────────────────────────────────────────
    image_provider: str = Field(
        default="fake_image",
        description="Image provider name: 'openai_image', 'fake_image', etc.",
    )
    image_model: str | None = Field(
        default=None,
        description="Image model identifier. When unset, each provider uses its default.",
    )

    # ── Request defaults ────────────────────────────────────────
    max_tokens: int = Field(default=4096, ge=1)
    max_retries: int = Field(default=3, ge=0)
    timeout_seconds: int = Field(default=120, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # ── Cost guardrails ─────────────────────────────────────────
    cost_limit_usd: float | None = Field(
        default=None,
        description="Max cumulative cost (USD) per LLMClient instance. None = no limit.",
    )
    cost_warn_usd: float | None = Field(
        default=None,
        description="Emit warning when cumulative cost exceeds this (USD).",
    )

    # ── Observability ───────────────────────────────────────────
    trace_enabled: bool = Field(default=False)
    trace_exporter: str = Field(
        default="none",
        description="Trace exporter: 'none', 'console', 'otlp'.",
    )
    trace_endpoint: str = Field(default="http://localhost:4317")
    trace_service_name: str = Field(default="llm-gateway")

    log_level: str = Field(default="INFO")
    log_format: str = Field(
        default="json",
        description="Log format: 'json' or 'console'.",
    )

    @model_validator(mode="after")
    def _resolve_api_key(self) -> GatewayConfig:
        """Fall back to provider-specific env vars if LLM_API_KEY is unset."""
        if self.api_key is not None:
            return self

        fallback_map: dict[str, str] = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }
        env_var = fallback_map.get(self.provider)
        if env_var:
            value = os.environ.get(env_var)
            if value:
                self.api_key = SecretStr(value)

        return self

    def get_api_key(self) -> str:
        """Return the resolved API key as a plain string.

        Raises:
            ValueError: If no API key is configured for a provider that needs one.
        """
        if self.api_key is None:
            msg = (
                f"No API key configured for provider '{self.provider}'. "
                f"Set LLM_API_KEY or the provider-specific env var."
            )
            raise ValueError(msg)
        return self.api_key.get_secret_value()
