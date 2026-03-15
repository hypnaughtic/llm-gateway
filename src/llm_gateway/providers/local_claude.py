"""Local Claude CLI provider — runs 'claude' as a subprocess for LLM inference."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from llm_gateway.config import GatewayConfig
from llm_gateway.exceptions import CLINotFoundError, ProviderError, ResponseValidationError
from llm_gateway.types import LLMMessage, LLMResponse, TokenUsage

if TYPE_CHECKING:
    from llm_gateway.tokenizers.anthropic_tokenizer import AnthropicTokenizer
    from llm_gateway.tokenizers.heuristic_tokenizer import HeuristicTokenizer

T = TypeVar("T")

logger = logging.getLogger(__name__)

# Rough estimate: 1 token ≈ 4 characters (for heuristic usage tracking)
_CHARS_PER_TOKEN = 4

# ANSI escape sequence pattern (colors, cursor movement, etc.)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

# Keys that identify a Claude CLI JSON wrapper (not LLM content)
_WRAPPER_KEYS = {"type", "session_id", "errors"}


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    return _ANSI_RE.sub("", text)


def _extract_json_object(text: str) -> str | None:
    """Find the outermost JSON object ``{...}`` in *text*.

    Returns the substring from the first ``{`` to the matching ``}``,
    or ``None`` if no balanced pair is found.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _unwrap_cli_envelope(text: str) -> str:
    """If *text* is a JSON-encoded CLI wrapper, return the ``result`` value.

    The Claude CLI ``--output-format json`` wraps the LLM response in an
    envelope like ``{"type":"result", "result":"...", ...}``.  If the raw
    text reaching ``_parse_response`` is such an envelope (e.g. because
    ``_run_cli`` hit the fallback path), this function extracts the actual
    LLM content so Pydantic validates the right thing.
    """
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return text

    if not isinstance(data, dict):
        return text

    # Heuristic: a CLI wrapper has ``type`` and at least one other
    # wrapper-specific key (``session_id``, ``errors``).
    if not _WRAPPER_KEYS.issubset(data.keys()):
        return text

    # Prefer ``structured_output`` (from --json-schema), then ``result``
    for key in ("structured_output", "result"):
        value = data.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return json.dumps(value)

    return text


class LocalClaudeProvider:
    """LLM provider that delegates to the local ``claude`` CLI binary.

    Structured output is achieved by passing ``--json-schema`` with the
    Pydantic model's JSON schema (preferred) and falling back to
    prompt-embedded schema instructions.
    """

    # Default model for local Claude CLI when none is explicitly configured.
    DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(self, timeout_seconds: int = 120) -> None:
        self._timeout = timeout_seconds
        self._claude_path = shutil.which("claude")
        if self._claude_path is None:
            raise CLINotFoundError()
        self._tokenizer: AnthropicTokenizer | HeuristicTokenizer | None = None

    @classmethod
    def from_config(cls, config: GatewayConfig) -> LocalClaudeProvider:
        """Factory method for the provider registry."""
        return cls(timeout_seconds=config.timeout_seconds)

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        response_model: type[T],
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        image_files: Sequence[str] | None = None,
    ) -> LLMResponse[T]:
        """Run claude CLI and parse structured output."""
        effective_model = model or self.DEFAULT_MODEL
        prompt = self._build_prompt(messages, response_model)

        # When image files are provided, prepend instructions to read them
        if image_files:
            image_instructions = "\n".join(
                f"First, read and examine the image at {path} using the Read tool."
                for path in image_files
            )
            prompt = (
                f"{image_instructions}\n"
                "After examining the image(s), evaluate them visually and then "
                "respond to the instructions below.\n\n"
                f"{prompt}"
            )

        # Build JSON schema string for --json-schema flag
        json_schema: str | None = None
        if issubclass(response_model, BaseModel):
            json_schema = json.dumps(response_model.model_json_schema())

        logger.debug(
            "claude_cli_request | model=%s response_model=%s prompt_length=%d\n%s",
            effective_model,
            response_model.__name__,
            len(prompt),
            prompt[:500],
        )
        start = time.monotonic()

        try:
            result_text, wrapper_meta = await self._run_cli(
                prompt,
                model=effective_model,
                json_schema=json_schema,
                image_files=image_files,
            )
        except Exception as exc:
            logger.error("claude_cli_error | %s: %s", type(exc).__name__, exc)
            raise ProviderError("local_claude", exc) from exc

        latency_ms = (time.monotonic() - start) * 1000

        # Prefer structured_output from --json-schema (already validated by CLI)
        content: T | None = None
        structured = wrapper_meta.get("structured_output")
        if structured is not None and issubclass(response_model, BaseModel):
            try:
                if isinstance(structured, dict):
                    content = response_model.model_validate(structured)  # type: ignore[assignment]
                elif isinstance(structured, str):
                    content = response_model.model_validate_json(structured)  # type: ignore[assignment]
            except Exception:
                logger.debug("structured_output_parse_failed, falling back to result")

        if content is None:
            content = self._parse_response(result_text, response_model)

        usage = self._build_usage(prompt, result_text, wrapper_meta)

        if isinstance(content, BaseModel):
            fields = content.model_dump()
            reply = " | ".join(f"{k}={v}" for k, v in fields.items())
        else:
            reply = str(content)[:200]
        logger.info(
            "claude_cli_complete | latency=%.0fms tokens=%d+%d cost=$%.4f\n  -> %s",
            latency_ms,
            usage.input_tokens,
            usage.output_tokens,
            usage.total_cost_usd,
            reply,
        )

        return LLMResponse(
            content=content,
            usage=usage,
            model=effective_model,
            provider="local_claude",
            latency_ms=latency_ms,
        )

    def _build_prompt(
        self,
        messages: Sequence[LLMMessage],
        response_model: type[T],
    ) -> str:
        """Build a single prompt string with embedded JSON schema."""
        parts: list[str] = []

        # System instruction with JSON schema
        if issubclass(response_model, BaseModel):
            schema = json.dumps(
                response_model.model_json_schema(),
                indent=2,
            )
            parts.append(
                "You MUST respond with ONLY a valid JSON object (no markdown, "
                "no explanation, no extra text) conforming to this schema:\n\n"
                f"```json\n{schema}\n```\n"
            )

        # Conversation messages
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"[System]: {content}")
            elif role == "assistant":
                parts.append(f"[Assistant]: {content}")
            else:
                parts.append(f"[User]: {content}")

        return "\n\n".join(parts)

    async def _run_cli(
        self,
        prompt: str,
        *,
        model: str | None = None,
        json_schema: str | None = None,
        image_files: Sequence[str] | None = None,
    ) -> tuple[str, dict[str, object]]:
        """Execute the claude CLI and return (result_text, wrapper_metadata).

        When *model* is provided the ``--model`` flag selects which Claude
        model the CLI uses (e.g. ``claude-haiku-4-5-20251001``).

        When *json_schema* is provided the ``--json-schema`` flag is passed to
        the CLI, enabling native structured-output validation.  The validated
        object is returned under the ``structured_output`` key in the wrapper.

        When *image_files* is provided, the ``--allowedTools "Read"`` flag
        replaces ``--tools ""`` so Claude CLI can use its Read tool to
        examine the image files for multimodal evaluation.
        """
        assert self._claude_path is not None

        # Strip CLAUDECODE env var to allow running inside a Claude Code session
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

        cmd: list[str] = [
            self._claude_path,
            "-p",
            prompt,
            "--output-format",
            "json",
            "--no-session-persistence",
        ]

        # When image files are present, allow Read tool for multimodal input.
        # Otherwise disable all tools for faster, text-only inference.
        if image_files:
            cmd.extend(["--allowedTools", "Read"])
        else:
            cmd.extend(["--tools", ""])

        cmd.extend(["--max-budget-usd", "5.00"])

        if model is not None:
            cmd.extend(["--model", model])
        if json_schema is not None:
            cmd.extend(["--json-schema", json_schema])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=self._timeout,
            )
        except TimeoutError as exc:
            proc.kill()
            logger.error("claude_cli_timeout | timeout=%ds", self._timeout)
            msg = f"Claude CLI timed out after {self._timeout}s"
            raise TimeoutError(msg) from exc

        stdout_text = stdout_bytes.decode(errors="replace").strip()
        stderr_text = stderr_bytes.decode(errors="replace").strip()

        if stderr_text:
            logger.warning("claude_cli_stderr | %s", stderr_text[:500])

        if proc.returncode != 0:
            msg = f"Claude CLI exited with code {proc.returncode}: {stderr_text}"
            raise RuntimeError(msg)

        # Strip ANSI escape codes that may leak into stdout
        stdout_clean = _strip_ansi(stdout_text)

        wrapper = self._parse_cli_json(stdout_clean)
        if wrapper is not None:
            is_cli_wrapper = _WRAPPER_KEYS.issubset(wrapper.keys())

            if "result" in wrapper:
                result_text = str(wrapper["result"])
                logger.debug(
                    "claude_cli_raw_result | duration=%dms cost=$%.6f\n%s",
                    wrapper.get("duration_ms", 0),
                    wrapper.get("total_cost_usd", 0.0),
                    result_text[:2000],
                )
                return result_text, wrapper

            if is_cli_wrapper:
                # CLI wrapper parsed but no result field (e.g. error_max_turns).
                # Return empty text but preserve wrapper for cost data.
                subtype = wrapper.get("subtype", "unknown")
                logger.warning(
                    "claude_cli_no_result | subtype=%s wrapper_keys=%s",
                    subtype,
                    sorted(wrapper.keys()),
                )
                return "", wrapper

        logger.debug(
            "claude_cli_raw_fallback | stdout_length=%d\n%s",
            len(stdout_clean),
            stdout_clean[:1000],
        )
        return stdout_clean, {}

    @staticmethod
    def _parse_cli_json(text: str) -> dict[str, object] | None:
        """Parse the CLI JSON wrapper from *text*.

        Handles three failure modes:
        1. Clean JSON — ``json.loads`` succeeds directly.
        2. Noisy stdout — non-JSON content before/after the JSON object
           (ANSI remnants, BOM, logging lines).  Falls back to extracting
           the outermost ``{...}`` substring.
        3. JSONL (newline-delimited) — multiple JSON objects on separate
           lines.  Scans for the last ``type=result`` object.
        """
        # Fast path: clean JSON (single object)
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, TypeError):
            pass

        # JSONL: scan lines in reverse for the result object.
        # This must run before _extract_json_object so that multi-line
        # output picks the correct (result) object, not the first one.
        for line in reversed(text.strip().split("\n")):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict) and data.get("type") == "result":
                    return data
            except (json.JSONDecodeError, TypeError):
                continue

        # Last resort: extract the outermost JSON object from noisy output
        extracted = _extract_json_object(text)
        if extracted is not None:
            try:
                data = json.loads(extracted)
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, TypeError):
                pass

        return None

    @staticmethod
    def _parse_response(raw: str, response_model: type[T]) -> T:
        """Parse and validate the raw JSON string against the response model."""
        cleaned = raw.strip()

        # Extract JSON from markdown code blocks (```json ... ```)
        # The result text often contains leading newlines before the fence
        if "```" in cleaned:
            json_lines: list[str] = []
            inside = False
            for line in cleaned.split("\n"):
                stripped = line.strip()
                if stripped.startswith("```") and not inside:
                    inside = True
                    continue
                if stripped == "```" and inside:
                    break
                if inside:
                    json_lines.append(line)
            if json_lines:
                cleaned = "\n".join(json_lines)

        # Safety check: detect if cleaned text is a CLI wrapper that
        # leaked through (e.g. _run_cli fallback returned raw stdout).
        # If so, extract the ``result`` field from it before validating.
        cleaned = _unwrap_cli_envelope(cleaned)

        # Try direct JSON parsing (no code block wrapper)
        try:
            if issubclass(response_model, BaseModel):
                return response_model.model_validate_json(cleaned)  # type: ignore[return-value]
        except Exception as exc:
            raise ResponseValidationError(response_model.__name__, str(exc)) from exc

        raise ResponseValidationError(
            response_model.__name__,
            "response_model must be a Pydantic BaseModel subclass",
        )

    @staticmethod
    def _build_usage(prompt: str, response: str, wrapper: dict[str, object]) -> TokenUsage:
        """Build token usage from CLI wrapper metadata, falling back to heuristics."""
        # Extract real token counts from the wrapper's usage/modelUsage fields
        usage_data = wrapper.get("usage", {})
        model_usage = wrapper.get("modelUsage", {})
        raw_cost = wrapper.get("total_cost_usd")
        total_cost = float(str(raw_cost)) if raw_cost is not None else 0.0

        input_tokens = 0
        output_tokens = 0

        if isinstance(usage_data, dict):
            input_tokens = int(usage_data.get("input_tokens", 0) or 0)
            output_tokens = int(usage_data.get("output_tokens", 0) or 0)
            # Include cache tokens in input count for accurate tracking
            cache_read = int(usage_data.get("cache_read_input_tokens", 0) or 0)
            cache_create = int(usage_data.get("cache_creation_input_tokens", 0) or 0)
            input_tokens += cache_read + cache_create

        # If wrapper had no usage data, fall back to heuristic
        if input_tokens == 0 and output_tokens == 0:
            input_tokens = max(1, len(prompt) // _CHARS_PER_TOKEN)
            output_tokens = max(1, len(response) // _CHARS_PER_TOKEN)

        # Split cost evenly between input/output if we have a total
        input_cost = total_cost * 0.5
        output_cost = total_cost * 0.5

        # Try to get per-model cost breakdown
        if isinstance(model_usage, dict):
            for _model_name, model_data in model_usage.items():
                if isinstance(model_data, dict) and "costUSD" in model_data:
                    total_cost = float(model_data["costUSD"])
                    input_cost = total_cost * 0.5
                    output_cost = total_cost * 0.5
                    break

        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens using the Claude tokenizer (with heuristic fallback)."""
        if self._tokenizer is None:
            try:
                from llm_gateway.tokenizers.anthropic_tokenizer import (
                    AnthropicTokenizer as _AT,
                )

                self._tokenizer = _AT()
            except ImportError:
                from llm_gateway.tokenizers.heuristic_tokenizer import (
                    HeuristicTokenizer as _HT,
                )

                self._tokenizer = _HT(chars_per_token=4.0)
        return self._tokenizer.count_tokens(text)

    async def close(self) -> None:
        """No-op — no persistent resources to clean up."""
