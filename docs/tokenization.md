# Token Counting

llm-gateway provides provider-aware token counting — count tokens without making LLM completion calls.

## Standalone Usage

```python
from llm_gateway import count_tokens

# Default provider is anthropic
n = count_tokens("Hello, world!")

# Specify a provider
n = count_tokens("Hello, world!", provider="gemini")
n = count_tokens("Hello, world!", provider="fake")
```

## Via LLMClient

```python
from llm_gateway import LLMClient

llm = LLMClient()  # uses provider from LLM_PROVIDER env var
n = llm.count_tokens("Hello, world!")
```

## Building a Tokenizer Directly

For repeated use, hold a reference to avoid repeated registry lookups:

```python
from llm_gateway import build_tokenizer

tok = build_tokenizer("anthropic")
n1 = tok.count_tokens("first text")
n2 = tok.count_tokens("second text")
```

## Provider Accuracy

| Provider | Method | Local? | Accuracy |
|---|---|---|---|
| `anthropic` | tiktoken BPE (cl100k_base) | Yes | Exact |
| `local_claude` | tiktoken BPE (fallback: heuristic) | Yes | Exact (fallback: ~75-80%) |
| `gemini` | google-genai SDK API call | No | Exact (fallback: ~80-85%) |
| `fake` | chars/4 heuristic | Yes | ~75-80% |

## Fallback Behavior

- **Anthropic/local_claude**: Uses tiktoken if available, otherwise falls back to calibrated heuristic (~4 chars/token).
- **Gemini**: Uses google-genai SDK `count_tokens()` API. Falls back to calibrated heuristic (~3.5 chars/token) if SDK is unavailable or API call fails.
- **Unknown provider**: Returns a heuristic tokenizer (no error).

## Limitations

- **Raw text only**: Token counts are for raw text input. They do not account for message framing overhead (system prompts, role prefixes, special tokens) that providers add during completion calls. Actual token usage in `complete()` calls will be higher.
- **No streaming**: Token counting is synchronous.
- **Gemini requires API key**: For exact Gemini counts, `GEMINI_API_KEY` must be set. Without it, the heuristic fallback is used.

## Custom Tokenizers

Register a custom tokenizer for a new or overridden provider:

```python
from llm_gateway import register_tokenizer

class MyTokenizer:
    def count_tokens(self, text: str) -> int:
        return len(text.split())

    @property
    def name(self) -> str:
        return "word-count"

register_tokenizer("my_provider", MyTokenizer)
```
