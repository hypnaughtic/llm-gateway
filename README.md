# llm-gateway

[![CI](https://github.com/hypnaughtic/llm-gateway/actions/workflows/ci.yml/badge.svg)](https://github.com/hypnaughtic/llm-gateway/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/llm-gateway)](https://pypi.org/project/llm-gateway/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Production-ready, vendor-agnostic LLM gateway** with config-driven provider switching, token counting, image generation, built-in cost tracking, and OpenTelemetry observability.

## Features

- **Zero-code provider switching** — Change `LLM_PROVIDER` env var, restart. Done.
- **Structured output** — Every call returns a validated Pydantic model via `response_model`.
- **Token counting** — Provider-aware `count_tokens()` without making LLM calls. Local BPE for Anthropic, SDK API for Gemini, heuristic fallback everywhere.
- **Image generation** — `ImageClient` with the same config-driven pattern (`LLM_IMAGE_PROVIDER`).
- **Multimodal input** — Pass `image_files` to `complete()` for vision/multimodal evaluation.
- **Built-in cost tracking** — Token usage and USD cost on every response; configurable warn threshold and hard limit.
- **Observability** — OpenTelemetry spans per LLM call with model, tokens, cost, latency attributes.
- **Extensible** — Add custom providers and tokenizers with a single factory function.
- **Type-safe** — Full type annotations, `py.typed`, strict mypy, zero errors.

## Quick Start

```bash
pip install 'llm-gateway[anthropic]'
```

```python
import asyncio
from pydantic import BaseModel
from llm_gateway import LLMClient

class Answer(BaseModel):
    text: str

async def main():
    async with LLMClient() as llm:  # reads LLM_* env vars
        resp = await llm.complete(
            messages=[{"role": "user", "content": "What is 2+2?"}],
            response_model=Answer,
        )
        print(resp.content.text)          # "4"
        print(resp.usage.total_cost_usd)  # 0.000123

asyncio.run(main())
```

## Installation

### As a PyPI package

```bash
pip install llm-gateway                     # core only (pydantic + tenacity)
pip install 'llm-gateway[anthropic]'        # + Anthropic provider
pip install 'llm-gateway[gemini]'           # + Gemini provider
pip install 'llm-gateway[tracing]'          # + OpenTelemetry
pip install 'llm-gateway[logging]'          # + structlog
pip install 'llm-gateway[all]'             # all optional deps
```

### As a git dependency

Pin to a specific commit SHA for reproducible builds:

```
# requirements.txt
llm-gateway @ git+https://github.com/hypnaughtic/llm-gateway.git@<COMMIT_SHA>

# pyproject.toml (PEP 621)
dependencies = [
    "llm-gateway @ git+https://github.com/hypnaughtic/llm-gateway.git@<COMMIT_SHA>",
]
```

## Configuration

All settings use the `LLM_` prefix and are read from environment variables or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `anthropic` | `anthropic`, `gemini`, `local_claude`, `fake`, or custom |
| `LLM_MODEL` | *(provider default)* | Model identifier (each provider has its own default) |
| `LLM_API_KEY` | — | API key (falls back to provider-specific env var) |
| `LLM_BASE_URL` | — | Optional base URL override |
| `LLM_MAX_TOKENS` | `4096` | Max response tokens |
| `LLM_TEMPERATURE` | `0.0` | Sampling temperature |
| `LLM_MAX_RETRIES` | `3` | Retry attempts with exponential backoff |
| `LLM_TIMEOUT_SECONDS` | `120` | Request timeout |
| `LLM_COST_LIMIT_USD` | — | Hard cost limit per client instance |
| `LLM_COST_WARN_USD` | — | Warning threshold |
| `LLM_IMAGE_PROVIDER` | `fake_image` | `openai_image`, `gemini_image`, `fake_image` |
| `LLM_IMAGE_API_KEY` | — | Image API key (falls back to provider-specific env var) |
| `LLM_TRACE_ENABLED` | `false` | Enable OTEL tracing |
| `LLM_TRACE_EXPORTER` | `none` | `none`, `console`, `otlp` |
| `LLM_LOG_LEVEL` | `INFO` | Log level |
| `LLM_LOG_FORMAT` | `json` | `json` or `console` |

## Providers

### Anthropic (default)

```bash
pip install 'llm-gateway[anthropic]'
export ANTHROPIC_API_KEY=sk-ant-...
export LLM_PROVIDER=anthropic
```

Default model: `claude-sonnet-4-5-20250514`.

### Gemini

```bash
pip install 'llm-gateway[gemini]'
export GEMINI_API_KEY=...
export LLM_PROVIDER=gemini
```

Default model: `gemini-2.5-flash`. All API-accessible Gemini models are supported with built-in pricing:

| Model | Input/1M | Output/1M | Notes |
|-------|----------|-----------|-------|
| `gemini-3.1-pro-preview` | $2.00 | $12.00 | Latest reasoning model |
| `gemini-3.1-flash-lite-preview` | $0.25 | $1.50 | Most cost-efficient |
| `gemini-3-flash-preview` | $0.50 | $3.00 | Fast + capable |
| `gemini-2.5-pro` | $1.25 | $10.00 | Best quality/price |
| `gemini-2.5-flash` | $0.15 | $0.60 | **Default** — balanced |
| `gemini-2.5-flash-lite` | $0.10 | $0.40 | Budget option |
| `gemini-2.0-flash` | $0.10 | $0.40 | Production stable |
| `gemini-2.0-flash-lite` | $0.075 | $0.30 | Lowest cost (2.0) |
| `gemini-1.5-pro` | $1.25 | $5.00 | Legacy pro |
| `gemini-1.5-flash` | $0.075 | $0.30 | Legacy flash |
| `gemini-1.5-flash-8b` | $0.0375 | $0.15 | Smallest model |

### Local Claude CLI

Use the Claude Code CLI for local inference — no API key needed:

```bash
export LLM_PROVIDER=local_claude
```

Default model: `claude-haiku-4-5-20251001`. Requires the `claude` CLI in your PATH.

### Custom Provider

```python
from llm_gateway import register_provider, LLMClient

class MyProvider:
    async def complete(self, messages, response_model, model=None, **kwargs):
        ...  # Your implementation
    def count_tokens(self, text):
        return len(text) // 4 or 1
    async def close(self):
        ...

register_provider("my_provider", lambda config: MyProvider())

# Use it
llm = LLMClient(config=GatewayConfig(provider="my_provider"))
```

## Token Counting

Count tokens without making LLM completion calls — useful for prompt budgeting, cost estimation, and input validation.

### Standalone

```python
from llm_gateway import count_tokens

n = count_tokens("Hello, world!")                    # default: anthropic
n = count_tokens("Hello, world!", provider="gemini") # gemini tokenizer
```

### Via LLMClient

```python
llm = LLMClient()
n = llm.count_tokens("Hello, world!")
```

### Build a reusable tokenizer

```python
from llm_gateway import build_tokenizer

tok = build_tokenizer("anthropic")  # cached singleton
n = tok.count_tokens("Hello, world!")
```

### Provider accuracy

| Provider | Method | Local? | Accuracy |
|---|---|---|---|
| `anthropic` | tiktoken BPE (cl100k_base) | Yes | Exact |
| `local_claude` | tiktoken BPE (fallback: heuristic) | Yes | Exact (fallback: ~75-80%) |
| `gemini` | google-genai SDK API call | No | Exact (fallback: ~80-85%) |
| `fake` | chars/4 heuristic | Yes | ~75-80% |

Unknown providers fall back to the heuristic tokenizer automatically (no error).

> **Note**: Token counts are for raw text only. They do not account for message framing overhead that providers add during completion calls.

## Image Generation

```python
from llm_gateway import ImageClient

async with ImageClient() as img:
    resp = await img.generate_image("a cat wearing a hat")
    print(resp.images[0].url)        # image URL or base64
    print(resp.usage.total_cost_usd) # cost
    print(img.total_cost_usd)        # cumulative
```

Configure with:

```bash
export LLM_IMAGE_PROVIDER=gemini_image  # or openai_image
export GEMINI_API_KEY=...               # provider-specific key
```

Available image providers:

| Provider | Config Value | Auth |
|----------|-------------|------|
| Gemini Imagen | `gemini_image` | `LLM_IMAGE_API_KEY` or `GEMINI_API_KEY` |
| OpenAI DALL-E | `openai_image` | `LLM_IMAGE_API_KEY` or `OPENAI_API_KEY` |
| Fake (testing) | `fake_image` | None |

## Multimodal (Vision)

Pass image files to `complete()` for multimodal evaluation:

```python
resp = await llm.complete(
    messages=[{"role": "user", "content": "Describe this image"}],
    response_model=Description,
    image_files=["photo.jpg"],
)
```

Currently supported by the `local_claude` provider (uses the Claude CLI's Read tool).

## Cost Tracking

Every response includes token usage and cost:

```python
resp = await llm.complete(messages, response_model=MyModel)
print(resp.usage.input_tokens)    # 150
print(resp.usage.output_tokens)   # 42
print(resp.usage.total_cost_usd)  # 0.000654

# Cumulative across all calls
print(llm.total_cost_usd)  # 0.001234
print(llm.cost_summary())  # {"total_tokens": ..., "total_cost_usd": ..., ...}
```

Set guardrails via env vars:

```bash
export LLM_COST_LIMIT_USD=1.00   # hard limit — raises CostLimitExceededError
export LLM_COST_WARN_USD=0.50    # warning threshold — logs a warning
```

## Testing

llm-gateway ships `FakeLLMProvider` and `FakeImageProvider` for consumer test suites:

```python
from llm_gateway import LLMClient, FakeLLMProvider
from pydantic import BaseModel

class Answer(BaseModel):
    text: str

# Pre-configured responses
fake = FakeLLMProvider()
fake.set_response(Answer, Answer(text="42"))

async with LLMClient(provider_instance=fake) as client:
    resp = await client.complete(
        messages=[{"role": "user", "content": "What is 6*7?"}],
        response_model=Answer,
    )
    assert resp.content.text == "42"
    assert fake.call_count == 1
    assert client.count_tokens("Hello") > 0  # token counting works too

# Or use via env var: LLM_PROVIDER=fake
```

Dynamic responses:

```python
def my_factory(response_model, messages):
    return response_model(text="dynamic response")

fake = FakeLLMProvider(response_factory=my_factory)
```

### Provider summary

| Provider | Config Value | Auth | Use Case |
|----------|-------------|------|----------|
| Anthropic | `anthropic` | `LLM_API_KEY` or `ANTHROPIC_API_KEY` | Production |
| Gemini | `gemini` | `LLM_API_KEY` or `GEMINI_API_KEY` | Production |
| Local Claude CLI | `local_claude` | None (requires `claude` CLI) | Free local dev |
| Fake (testing) | `fake` | None | Unit/integration tests |

## Architecture

```
Consumer → LLMClient  → Provider (via Registry) → LLM API / CLI
                ↕              ↕
          CostTracker    Tokenizer (via Tokenizer Registry)
                ↕
           OTEL Tracer

Consumer → ImageClient → ImageProvider (via Image Registry) → Image API
```

### Key design principles

- **Protocol-based**: `LLMProvider` Protocol with `complete()`, `count_tokens()`, `close()`. `ImageGenerationProvider` with `generate_image()`, `close()`.
- **Lazy registration**: Provider SDKs are only imported when the provider is first used. Missing optional deps are silently skipped.
- **Config-driven**: `GatewayConfig(BaseSettings)` reads `LLM_*` env vars via pydantic-settings. Zero code changes to switch providers.
- **Dependency injection**: Pass `provider_instance=` to bypass the registry for testing.
- **Optional heavy deps**: Core package has only pydantic, pydantic-settings, and tenacity. Provider SDKs, OTEL, and structlog are extras.

## Development

```bash
git clone https://github.com/hypnaughtic/llm-gateway.git
cd llm-gateway
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

### Running checks

```bash
ruff check . && ruff format --check .   # lint + format
mypy .                                   # strict type check
pytest -m unit -v                        # unit tests (322 tests, 96% coverage)
```

### Integration tests

The `integration_tests/` directory is a self-contained Python project that installs llm-gateway as an external dependency — validating the package works as consumers would use it.

```bash
cd integration_tests
pip install -e .                  # install with llm-gateway as dependency

# Dry-run (mocked, no real LLM calls — default)
pytest -v                         # 32 dry-run tests

# Live (real LLM calls — requires API keys / claude CLI)
pytest --run-live -m live -v      # LLM + tokenizer live tests
```

Live runs print a session summary:

```
========================= Live Test Suite Summary =========================
  Claude CLI calls      : 11
  Total tokens (est.)   : 1,794  (1,555 input + 239 output)
  Cost estimation       : $0.000000
                          (local_claude — no API fees)
===========================================================================
```

### Pre-commit hooks

Pre-commit mirrors the CI pipeline on every commit:

- Trailing whitespace and EOF fixes
- Ruff lint + format
- Mypy strict type checking
- Unit tests with per-file coverage (90%+ per source file)
- Integration tests (dry-run)

```bash
pre-commit install                # set up hooks
pre-commit run --all-files        # manual run
```

## CI

GitHub Actions runs on every push and PR to `main`:

| Job | What it does |
|-----|--------------|
| **lint** | ruff check, ruff format, mypy (Python 3.12) |
| **test** | Unit tests on Python 3.11, 3.12, 3.13 with per-file coverage (>= 90%) |
| **integration-test** | Installs llm-gateway as a package, runs dry-run integration tests |
| **test-minimal** | Verifies core imports work without optional extras |

## License

MIT
