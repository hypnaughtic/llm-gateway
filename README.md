# llm-gateway

[![CI](https://github.com/YOUR_ORG/llm-gateway/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_ORG/llm-gateway/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/llm-gateway)](https://pypi.org/project/llm-gateway/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Production-ready, vendor-agnostic LLM gateway** with config-driven provider switching, built-in cost tracking, and OpenTelemetry observability.

## Features

- **Zero-code provider switching** — Change `LLM_PROVIDER` env var, restart. Done.
- **Structured output** — Every call returns a validated Pydantic model via `response_model`.
- **Built-in cost tracking** — Token usage and USD cost on every response; configurable guardrails.
- **Observability** — OpenTelemetry spans per LLM call with model, tokens, cost, latency attributes.
- **Extensible** — Add custom providers with a single factory function.
- **Type-safe** — Full type annotations, `py.typed`, strict mypy.

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
    llm = LLMClient()  # reads LLM_* env vars
    resp = await llm.complete(
        messages=[{"role": "user", "content": "What is 2+2?"}],
        response_model=Answer,
    )
    print(resp.content.text)          # "4"
    print(resp.usage.total_cost_usd)  # 0.000123
    await llm.close()

asyncio.run(main())
```

## Installation

### As a PyPI package

```bash
pip install llm-gateway                     # core only
pip install 'llm-gateway[anthropic]'        # + Anthropic provider
pip install 'llm-gateway[gemini]'           # + Gemini provider
pip install 'llm-gateway[all]'              # all optional deps
```

### As a git dependency

Pin to a specific commit SHA for reproducible builds:

```
# requirements.txt
llm-gateway @ git+https://github.com/YOUR_ORG/llm-gateway.git@<COMMIT_SHA>

# pyproject.toml (PEP 621)
dependencies = [
    "llm-gateway @ git+https://github.com/YOUR_ORG/llm-gateway.git@<COMMIT_SHA>",
]
```

## Configuration

All settings use the `LLM_` prefix and are read from environment variables or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `anthropic` | `anthropic`, `gemini`, `local_claude`, `fake`, or custom |
| `LLM_MODEL` | `claude-sonnet-4-5-20250514` | Model identifier |
| `LLM_API_KEY` | — | API key (falls back to provider-specific key) |
| `LLM_MAX_TOKENS` | `4096` | Max response tokens |
| `LLM_MAX_RETRIES` | `3` | Retry attempts |
| `LLM_TIMEOUT_SECONDS` | `120` | Request timeout |
| `LLM_COST_LIMIT_USD` | — | Hard cost limit per client instance |
| `LLM_COST_WARN_USD` | — | Warning threshold |
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

### Gemini

```bash
pip install 'llm-gateway[gemini]'
export GEMINI_API_KEY=...
export LLM_PROVIDER=gemini
```

Default model: `gemini-2.5-flash`. Supported models: `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-2.0-flash`, `gemini-1.5-flash`, `gemini-1.5-pro`.

### Local Claude CLI

Use the Claude Code CLI for local inference — no API key needed:

```bash
export LLM_PROVIDER=local_claude
```

### Custom Provider

```python
from llm_gateway import register_provider

class MyProvider:
    async def complete(self, messages, response_model, model, **kwargs):
        ...  # Your implementation
    async def close(self):
        ...

register_provider("my_provider", lambda config: MyProvider())
```

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

## Testing

llm-gateway ships a `FakeLLMProvider` for use in consumer test suites:

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

# Or use via env var: LLM_PROVIDER=fake
```

You can also pass a `response_factory` for dynamic responses:

```python
def my_factory(response_model, messages):
    return response_model(text="dynamic response")

fake = FakeLLMProvider(response_factory=my_factory)
```

### Providers

| Provider | Config Value | Auth | Use Case |
|----------|-------------|------|----------|
| Anthropic | `anthropic` | `LLM_API_KEY` or `ANTHROPIC_API_KEY` | Production |
| Gemini | `gemini` | `LLM_API_KEY` or `GEMINI_API_KEY` | Production |
| Local Claude CLI | `local_claude` | None | Free local dev |
| Fake (testing) | `fake` | None | Unit/integration tests |

## Development

```bash
git clone https://github.com/YOUR_ORG/llm-gateway.git
cd llm-gateway
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

### Running checks

```bash
ruff check .                      # lint
ruff format --check .             # format check
mypy .                            # type check
pytest -m unit -v                 # unit tests (234 tests, 90%+ per-file coverage)
```

### Integration tests

The `integration_tests/` directory is a self-contained Python project that installs llm-gateway as an external dependency — validating the package works as consumers would use it.

```bash
cd integration_tests
pip install -e .                  # install with llm-gateway as dependency

# Dry-run (mocked, no real LLM calls — default)
pytest -v

# Live (real Claude CLI calls — requires `claude` in PATH)
pytest --run-live -m live -v
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
- Unit tests
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
