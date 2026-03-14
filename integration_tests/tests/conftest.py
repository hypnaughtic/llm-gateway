"""Shared fixtures and CLI options for integration tests.

Provides:
- --run-live CLI flag to enable live LLM tests
- FakeLLMProvider for dry-run mode (fully mocked, no real calls)
- Automatic provider selection: fake for dry-run, local_claude for live
- LiveSessionStats for aggregating token usage and cost across live tests
- Live log streaming (DEBUG level) when --run-live is active
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import pytest
from pydantic import BaseModel

# ── All imports come from the installed llm-gateway package ──
from llm_gateway import GatewayConfig, ImageClient, LLMClient, LLMResponse
from llm_gateway.cost import build_token_usage
from llm_gateway.types import LLMMessage

T = TypeVar("T")


# ─── Live Session Stats ─────────────────────────────────────────


@dataclass
class LiveSessionStats:
    """Accumulates stats across all live integration tests."""

    cli_sessions_opened: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0

    def record_client(self, client: LLMClient) -> None:
        """Record stats from a completed client session."""
        summary = client.cost_summary()
        self.cli_sessions_opened += summary["call_count"]
        self.total_input_tokens += summary["total_input_tokens"]
        self.total_output_tokens += summary["total_output_tokens"]
        self.total_cost_usd += summary["total_cost_usd"]


# Module-level ref so pytest_terminal_summary can access it
_live_stats: LiveSessionStats | None = None


# ─── CLI Option ──────────────────────────────────────────────────


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add --run-live flag and image test options."""
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Run live integration tests that call real LLM providers.",
    )
    parser.addoption(
        "--image-prompt",
        default="a serene mountain lake at sunset, digital art",
        help="Prompt to use for image generation e2e tests.",
    )
    parser.addoption(
        "--image-provider",
        default="gemini_image",
        help="Image provider for e2e tests: 'gemini_image' or 'openai_image'.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Adjust markers and enable live log streaming when --run-live is given."""
    if config.getoption("--run-live", default=False):
        # Remove the default '-m dry_run' so all tests (or the user's -m) run
        config.option.markexpr = config.option.markexpr or ""
        if config.option.markexpr == "dry_run":
            config.option.markexpr = ""
        # Stream logs at DEBUG level during live test runs
        config.option.log_cli_level = "DEBUG"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip live tests unless --run-live is passed."""
    if config.getoption("--run-live"):
        return
    skip_live = pytest.mark.skip(reason="Need --run-live to execute live LLM tests")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: pytest.Config) -> None:
    """Print live test suite summary with CLI sessions, tokens, and cost."""
    if _live_stats is None or _live_stats.cli_sessions_opened == 0:
        return

    total_tokens = _live_stats.total_input_tokens + _live_stats.total_output_tokens

    terminalreporter.write_sep("=", "Live Test Suite Summary")
    terminalreporter.write_line(f"  Claude CLI calls      : {_live_stats.cli_sessions_opened}")
    terminalreporter.write_line(
        f"  Total tokens (est.)   : {total_tokens:,}"
        f"  ({_live_stats.total_input_tokens:,} input"
        f" + {_live_stats.total_output_tokens:,} output)"
    )
    terminalreporter.write_line(f"  Cost estimation       : ${_live_stats.total_cost_usd:.6f}")
    if _live_stats.total_cost_usd == 0.0:
        terminalreporter.write_line("                          (local_claude — no API fees)")
    terminalreporter.write_sep("=", "")


# ─── Fake Provider (dry-run mode) ───────────────────────────────


class FakeLLMProvider:
    """Deterministic mock provider for dry-run integration tests.

    Inspects the response_model's fields and returns plausible canned data.
    This simulates what a real LLM would return — validated Pydantic models
    with realistic token usage — without making any real calls.
    """

    # Canned answers keyed by field name patterns
    _CANNED: ClassVar[dict[str, Any]] = {
        "capital": "Paris",
        "country": "France",
        "city": "Paris",
        "greeting": "Hello! How can I help you today?",
        "message": "Hello from the LLM gateway!",
        "text": "This is a response from the LLM.",
        "answer": "42",
        "summary": "A concise summary of the topic.",
        "sentiment": "positive",
        "language": "English",
        "translation": "Bonjour le monde",
        "explanation": "This is a detailed explanation.",
        "name": "Claude",
        "color": "blue",
    }

    def __init__(self) -> None:
        self.call_log: list[dict[str, Any]] = []

    def _build_fake_content(self, response_model: type[T]) -> T:
        """Build a valid instance of response_model using canned values."""
        if not issubclass(response_model, BaseModel):
            msg = f"response_model must be a BaseModel subclass, got {response_model}"
            raise TypeError(msg)

        fields = response_model.model_fields
        data: dict[str, Any] = {}
        for field_name, field_info in fields.items():
            annotation = field_info.annotation
            # Match by field name first
            if field_name.lower() in self._CANNED:
                data[field_name] = self._CANNED[field_name.lower()]
            elif annotation is str:
                data[field_name] = f"fake_{field_name}_value"
            elif annotation is int:
                data[field_name] = 42
            elif annotation is float:
                data[field_name] = 0.95
            elif annotation is bool:
                data[field_name] = True
            elif annotation is list or (
                hasattr(annotation, "__origin__") and annotation.__origin__ is list
            ):
                data[field_name] = []
            else:
                data[field_name] = f"fake_{field_name}"

        return response_model.model_validate(data)

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        response_model: type[T],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        image_files: Sequence[str] | None = None,
    ) -> LLMResponse[T]:
        """Return a deterministic fake response and log the call."""
        content = self._build_fake_content(response_model)

        self.call_log.append(
            {
                "messages": list(messages),
                "response_model": response_model.__name__,
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )

        usage = build_token_usage(model, input_tokens=150, output_tokens=60)
        return LLMResponse(
            content=content,
            usage=usage,
            model=model,
            provider="fake",
            latency_ms=2.5,
        )

    async def close(self) -> None:
        """No-op."""


# ─── Fixtures ────────────────────────────────────────────────────


@pytest.fixture(scope="session", autouse=True)
def _configure_live_logging(request: pytest.FixtureRequest) -> None:
    """Prevent gateway logging from clobbering pytest's live-log handler.

    ``configure_logging()`` calls ``root.handlers.clear()`` which removes
    pytest's ``_LiveLoggingStreamHandler``.  We mark it as already configured
    so ``LLMClient.__init__`` skips the call, then set the root logger to
    DEBUG so provider-level log records reach pytest's handler.
    """
    if not request.config.getoption("--run-live", default=False):
        return
    import logging as stdlib_logging

    from llm_gateway.observability import logging as gw_logging

    gw_logging._CONFIGURED = True
    stdlib_logging.getLogger().setLevel(stdlib_logging.DEBUG)


@pytest.fixture(scope="session")
def live_session_stats() -> LiveSessionStats:
    """Session-scoped stats accumulator for live test runs."""
    global _live_stats
    _live_stats = LiveSessionStats()
    return _live_stats


@pytest.fixture
def make_live_client(
    live_session_stats: LiveSessionStats,
) -> Iterator[Any]:
    """Factory fixture that creates live LLMClients and auto-records stats.

    Usage in tests::

        async def test_foo(self, make_live_client):
            async with make_live_client() as client:
                resp = await client.complete(...)

    On fixture teardown, each client's cost/token stats are recorded
    into the session-wide LiveSessionStats accumulator.
    """
    created: list[LLMClient] = []

    def _factory(timeout: int = 180) -> LLMClient:
        config = GatewayConfig(
            provider="local_claude",
            timeout_seconds=timeout,
            trace_enabled=False,
            log_format="console",
            log_level="DEBUG",
        )
        client = LLMClient(config=config)
        created.append(client)
        return client

    yield _factory

    for client in created:
        live_session_stats.record_client(client)


@pytest.fixture
def fake_provider() -> FakeLLMProvider:
    """Fresh FakeLLMProvider for dry-run tests."""
    return FakeLLMProvider()


@pytest.fixture
def dry_run_client(fake_provider: FakeLLMProvider) -> LLMClient:
    """LLMClient wired to the fake provider (no real LLM calls)."""
    config = GatewayConfig(
        provider="fake",
        model="dry-run-model",
        trace_enabled=False,
        log_format="console",
    )
    return LLMClient(config=config, provider_instance=fake_provider)


@pytest.fixture
def live_client() -> LLMClient | None:
    """LLMClient wired to the real local_claude provider.

    Returns None if claude CLI is not available (tests should skip).
    """
    if not shutil.which("claude"):
        return None
    config = GatewayConfig(
        provider="local_claude",
        timeout_seconds=180,
        trace_enabled=False,
        log_format="console",
        log_level="DEBUG",
    )
    return LLMClient(config=config)


# ─── Image Test Fixtures ──────────────────────────────────────


@dataclass
class LiveImageSessionStats:
    """Accumulates stats across all live image integration tests."""

    total_calls: int = 0
    total_images: int = 0
    total_cost_usd: float = 0.0

    def record_client(self, client: ImageClient) -> None:
        """Record stats from a completed image client session."""
        summary = client.cost_summary()
        self.total_calls += summary["call_count"]
        self.total_cost_usd += summary["total_cost_usd"]


_live_image_stats: LiveImageSessionStats | None = None


@pytest.fixture(scope="session")
def live_image_session_stats() -> LiveImageSessionStats:
    """Session-scoped stats accumulator for live image test runs."""
    global _live_image_stats
    _live_image_stats = LiveImageSessionStats()
    return _live_image_stats


@pytest.fixture
def image_prompt(request: pytest.FixtureRequest) -> str:
    """Return the image prompt from CLI option."""
    return str(request.config.getoption("--image-prompt"))


@pytest.fixture
def test_output_dir() -> Path:
    """Create and return the test outputs directory."""
    out = Path(__file__).parent.parent / "test_outputs"
    out.mkdir(exist_ok=True)
    return out


@pytest.fixture
def make_live_image_client(
    request: pytest.FixtureRequest,
    live_image_session_stats: LiveImageSessionStats,
) -> Iterator[Callable[[], ImageClient]]:
    """Factory fixture that creates live ImageClients and auto-records stats.

    Reads API key from environment (set via .env or export).
    Provider is selected via --image-provider CLI option.
    """
    created: list[ImageClient] = []
    image_provider = str(request.config.getoption("--image-provider"))

    def _factory() -> ImageClient:
        config = GatewayConfig(
            image_provider=image_provider,
            trace_enabled=False,
            log_format="console",
            log_level="DEBUG",
        )
        client = ImageClient(config=config)
        created.append(client)
        return client

    yield _factory

    for client in created:
        live_image_session_stats.record_client(client)
