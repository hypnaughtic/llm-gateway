"""Tests for LocalClaudeProvider."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from llm_gateway.config import GatewayConfig
from llm_gateway.exceptions import CLINotFoundError, ProviderError, ResponseValidationError
from llm_gateway.providers.local_claude import (
    _extract_json_object,
    _strip_ansi,
    _unwrap_cli_envelope,
)
from llm_gateway.types import LLMResponse


class _TestModel(BaseModel):
    answer: str


@pytest.mark.unit
class TestLocalClaudeProvider:
    def test_raises_if_cli_not_found(self) -> None:
        """CLINotFoundError if 'claude' not in PATH."""
        with patch("shutil.which", return_value=None):
            from llm_gateway.providers.local_claude import LocalClaudeProvider

            with pytest.raises(CLINotFoundError):
                LocalClaudeProvider()

    @pytest.mark.asyncio
    async def test_complete_parses_json(self) -> None:
        """Subprocess JSON output is parsed into response_model."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            provider = LocalClaudeProvider()

        json_output = json.dumps({"result": json.dumps({"answer": "world"})})

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(json_output.encode(), b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            resp = await provider.complete(
                messages=[{"role": "user", "content": "hello"}],
                response_model=_TestModel,
                model="local",
            )

        assert isinstance(resp, LLMResponse)
        assert resp.content.answer == "world"
        assert resp.provider == "local_claude"
        assert resp.usage.input_tokens > 0

    @pytest.mark.asyncio
    async def test_timeout_kills_process(self) -> None:
        """TimeoutError kills the subprocess."""
        import asyncio

        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            provider = LocalClaudeProvider(timeout_seconds=1)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
        mock_proc.kill = MagicMock()

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            patch("asyncio.wait_for", side_effect=asyncio.TimeoutError),
            pytest.raises(ProviderError),
        ):
            await provider.complete(
                messages=[{"role": "user", "content": "hello"}],
                response_model=_TestModel,
                model="local",
            )

    def test_build_prompt_includes_schema(self) -> None:
        """Prompt includes JSON schema for structured output."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            provider = LocalClaudeProvider()

        prompt = provider._build_prompt(
            messages=[{"role": "user", "content": "test"}],
            response_model=_TestModel,
        )
        assert "answer" in prompt
        assert "string" in prompt  # JSON schema type

    def test_build_prompt_system_and_assistant_roles(self) -> None:
        """Prompt formats system and assistant roles correctly."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            provider = LocalClaudeProvider()

        prompt = provider._build_prompt(
            messages=[
                {"role": "system", "content": "You are a helpful bot."},
                {"role": "user", "content": "Hi there."},
                {"role": "assistant", "content": "Hello!"},
            ],
            response_model=_TestModel,
        )
        assert "[System]: You are a helpful bot." in prompt
        assert "[User]: Hi there." in prompt
        assert "[Assistant]: Hello!" in prompt

    def test_from_config_factory(self) -> None:
        """from_config creates a provider with timeout from config."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            config = GatewayConfig(provider="local_claude", timeout_seconds=300)
            provider = LocalClaudeProvider.from_config(config)
            assert provider._timeout == 300

    def test_parse_response_valid_json(self) -> None:
        """_parse_response parses valid JSON against response model."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        result = LocalClaudeProvider._parse_response('{"answer": "42"}', _TestModel)
        assert isinstance(result, _TestModel)
        assert result.answer == "42"

    def test_parse_response_markdown_code_block(self) -> None:
        """_parse_response strips markdown code fences before parsing."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        raw = '```json\n{"answer": "hello"}\n```'
        result = LocalClaudeProvider._parse_response(raw, _TestModel)
        assert isinstance(result, _TestModel)
        assert result.answer == "hello"

    def test_parse_response_invalid_json_raises(self) -> None:
        """_parse_response raises ResponseValidationError on invalid JSON."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with pytest.raises(ResponseValidationError, match="_TestModel"):
            LocalClaudeProvider._parse_response("not valid json", _TestModel)

    @pytest.mark.asyncio
    async def test_run_cli_nonzero_exit_raises(self) -> None:
        """Non-zero exit code from CLI raises RuntimeError wrapped in ProviderError."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            provider = LocalClaudeProvider()

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"some error"))
        mock_proc.returncode = 1

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            pytest.raises(ProviderError, match="local_claude"),
        ):
            await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                response_model=_TestModel,
                model="local",
            )

    @pytest.mark.asyncio
    async def test_run_cli_raw_text_fallback(self) -> None:
        """Raw text output (not JSON wrapper) is used as-is for parsing."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            provider = LocalClaudeProvider()

        # Return raw JSON directly (no {"result": ...} wrapper)
        raw_json = '{"answer": "direct"}'
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(raw_json.encode(), b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            resp = await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                response_model=_TestModel,
                model="local",
            )
        assert resp.content.answer == "direct"

    @pytest.mark.asyncio
    async def test_close_is_noop(self) -> None:
        """close() completes without error."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            provider = LocalClaudeProvider()
        await provider.close()  # Should not raise

    def test_build_usage_heuristic_fallback(self) -> None:
        """_build_usage falls back to heuristic when wrapper has no usage data."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        usage = LocalClaudeProvider._build_usage("hello world", '{"answer": "ok"}', {})
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0
        assert usage.input_cost_usd == 0.0
        assert usage.output_cost_usd == 0.0

    def test_build_usage_from_wrapper(self) -> None:
        """_build_usage extracts real token counts from wrapper metadata."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        wrapper: dict[str, object] = {
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "total_cost_usd": 0.01,
        }
        usage = LocalClaudeProvider._build_usage("prompt", "response", wrapper)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_cost_usd == pytest.approx(0.01, abs=0.001)

    @pytest.mark.asyncio
    async def test_complete_uses_structured_output(self) -> None:
        """structured_output from --json-schema is preferred over result parsing."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            provider = LocalClaudeProvider()

        wrapper = {
            "type": "result",
            "result": "some text that is not JSON",
            "structured_output": {"answer": "from_schema"},
            "session_id": "abc",
            "errors": [],
        }
        json_output = json.dumps(wrapper)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(json_output.encode(), b""),
        )
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            resp = await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                response_model=_TestModel,
                model="local",
            )

        assert resp.content.answer == "from_schema"

    @pytest.mark.asyncio
    async def test_complete_passes_json_schema_flag(self) -> None:
        """--json-schema flag is passed to the CLI subprocess."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            provider = LocalClaudeProvider()

        wrapper = {"result": json.dumps({"answer": "ok"})}
        json_output = json.dumps(wrapper)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(json_output.encode(), b""),
        )
        mock_proc.returncode = 0

        captured_cmd: list[str] = []

        async def capture_exec(*args: str, **kwargs: object) -> AsyncMock:
            captured_cmd.extend(args)
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=capture_exec):
            await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                response_model=_TestModel,
                model="local",
            )

        assert "--json-schema" in captured_cmd

    @pytest.mark.asyncio
    async def test_run_cli_strips_ansi_codes(self) -> None:
        """ANSI escape codes in stdout are stripped before JSON parsing."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            provider = LocalClaudeProvider()

        # Wrap valid JSON with ANSI codes
        wrapper = {"result": json.dumps({"answer": "clean"})}
        ansi_output = f"\x1b[32m{json.dumps(wrapper)}\x1b[0m"

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(ansi_output.encode(), b""),
        )
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            resp = await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                response_model=_TestModel,
                model="local",
            )

        assert resp.content.answer == "clean"

    def test_parse_response_unwraps_leaked_envelope(self) -> None:
        """_parse_response detects a leaked CLI wrapper and extracts result."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        # Simulate the exact error case: full wrapper JSON reaches _parse_response
        wrapper = {
            "type": "result",
            "subtype": "success",
            "result": json.dumps({"answer": "extracted"}),
            "session_id": "test-session-f66eedad",
            "errors": [],
        }
        raw = json.dumps(wrapper)

        result = LocalClaudeProvider._parse_response(raw, _TestModel)
        assert isinstance(result, _TestModel)
        assert result.answer == "extracted"

    def test_parse_cli_json_jsonl_format(self) -> None:
        """_parse_cli_json handles JSONL with multiple lines."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        lines = [
            json.dumps({"type": "assistant", "content": "thinking..."}),
            json.dumps(
                {
                    "type": "result",
                    "result": '{"answer":"ok"}',
                    "session_id": "s1",
                }
            ),
        ]
        text = "\n".join(lines)

        wrapper = LocalClaudeProvider._parse_cli_json(text)
        assert wrapper is not None
        assert wrapper["type"] == "result"
        assert wrapper["result"] == '{"answer":"ok"}'

    def test_parse_cli_json_noisy_output(self) -> None:
        """_parse_cli_json extracts JSON from noisy stdout with prefix text."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        noisy = 'Loading config...\n{"result": "hello", "type": "result"}\n'
        wrapper = LocalClaudeProvider._parse_cli_json(noisy)
        assert wrapper is not None
        assert wrapper["result"] == "hello"

    @pytest.mark.asyncio
    async def test_image_files_uses_allowed_tools_read(self) -> None:
        """When image_files is provided, --allowedTools Read replaces --tools ''."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            provider = LocalClaudeProvider()

        wrapper = {"result": json.dumps({"answer": "image_eval"})}
        json_output = json.dumps(wrapper)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(json_output.encode(), b""),
        )
        mock_proc.returncode = 0

        captured_cmd: list[str] = []

        async def capture_exec(*args: str, **kwargs: object) -> AsyncMock:
            captured_cmd.extend(args)
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=capture_exec):
            resp = await provider.complete(
                messages=[{"role": "user", "content": "evaluate this image"}],
                response_model=_TestModel,
                model="local",
                image_files=["/tmp/test_image.png"],
            )

        assert resp.content.answer == "image_eval"
        assert "--allowedTools" in captured_cmd
        assert "Read" in captured_cmd
        assert "--tools" not in captured_cmd

    @pytest.mark.asyncio
    async def test_no_image_files_uses_tools_empty(self) -> None:
        """Without image_files, --tools '' disables all tools."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            provider = LocalClaudeProvider()

        wrapper = {"result": json.dumps({"answer": "text_only"})}
        json_output = json.dumps(wrapper)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(json_output.encode(), b""),
        )
        mock_proc.returncode = 0

        captured_cmd: list[str] = []

        async def capture_exec(*args: str, **kwargs: object) -> AsyncMock:
            captured_cmd.extend(args)
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=capture_exec):
            await provider.complete(
                messages=[{"role": "user", "content": "hello"}],
                response_model=_TestModel,
                model="local",
            )

        assert "--tools" in captured_cmd
        assert "--allowedTools" not in captured_cmd

    @pytest.mark.asyncio
    async def test_image_files_prepends_read_instructions(self) -> None:
        """Image files add read instructions to the prompt."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            provider = LocalClaudeProvider()

        wrapper = {"result": json.dumps({"answer": "ok"})}
        json_output = json.dumps(wrapper)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(json_output.encode(), b""),
        )
        mock_proc.returncode = 0

        captured_cmd: list[str] = []

        async def capture_exec(*args: str, **kwargs: object) -> AsyncMock:
            captured_cmd.extend(args)
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=capture_exec):
            await provider.complete(
                messages=[{"role": "user", "content": "eval"}],
                response_model=_TestModel,
                model="local",
                image_files=["/tmp/img1.png", "/tmp/img2.png"],
            )

        # The prompt is passed via -p flag, which is the arg after "-p"
        p_index = captured_cmd.index("-p")
        prompt = captured_cmd[p_index + 1]
        assert "/tmp/img1.png" in prompt
        assert "/tmp/img2.png" in prompt
        assert "Read tool" in prompt


@pytest.mark.unit
class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_strip_ansi_removes_codes(self) -> None:
        """_strip_ansi removes ANSI escape sequences."""
        assert _strip_ansi("\x1b[32mhello\x1b[0m") == "hello"
        assert _strip_ansi("no codes") == "no codes"

    def test_extract_json_object_basic(self) -> None:
        """_extract_json_object finds a balanced JSON object."""
        assert _extract_json_object('prefix {"a": 1} suffix') == '{"a": 1}'

    def test_extract_json_object_nested(self) -> None:
        """_extract_json_object handles nested braces."""
        text = 'x {"a": {"b": 2}} y'
        assert _extract_json_object(text) == '{"a": {"b": 2}}'

    def test_extract_json_object_with_strings(self) -> None:
        """_extract_json_object ignores braces inside strings."""
        text = '{"a": "x{y}z"}'
        assert _extract_json_object(text) == '{"a": "x{y}z"}'

    def test_extract_json_object_no_json(self) -> None:
        """_extract_json_object returns None when no JSON found."""
        assert _extract_json_object("no json here") is None

    def test_unwrap_cli_envelope_extracts_result(self) -> None:
        """_unwrap_cli_envelope extracts result from wrapper."""
        wrapper = json.dumps(
            {
                "type": "result",
                "session_id": "s1",
                "errors": [],
                "result": '{"answer": "42"}',
            }
        )
        assert _unwrap_cli_envelope(wrapper) == '{"answer": "42"}'

    def test_unwrap_cli_envelope_prefers_structured_output(self) -> None:
        """_unwrap_cli_envelope prefers structured_output over result."""
        wrapper = json.dumps(
            {
                "type": "result",
                "session_id": "s1",
                "errors": [],
                "result": "raw text",
                "structured_output": {"answer": "validated"},
            }
        )
        unwrapped = _unwrap_cli_envelope(wrapper)
        assert json.loads(unwrapped) == {"answer": "validated"}

    def test_unwrap_cli_envelope_passthrough_normal_json(self) -> None:
        """_unwrap_cli_envelope passes through non-wrapper JSON."""
        normal = '{"answer": "42"}'
        assert _unwrap_cli_envelope(normal) == normal

    def test_unwrap_cli_envelope_passthrough_non_json(self) -> None:
        """_unwrap_cli_envelope passes through non-JSON text."""
        assert _unwrap_cli_envelope("not json") == "not json"

    def test_extract_json_object_escaped_quotes(self) -> None:
        """_extract_json_object handles escaped quotes in strings."""
        text = r'{"a": "he said \"hi\""}'
        result = _extract_json_object(text)
        assert result == text
