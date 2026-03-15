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

    def test_extract_json_object_unbalanced_braces(self) -> None:
        """_extract_json_object returns None for unbalanced braces (line 71)."""
        # Opening brace with no closing brace
        assert _extract_json_object('{"a": 1') is None

    def test_unwrap_cli_envelope_non_dict_json(self) -> None:
        """_unwrap_cli_envelope passes through JSON that parses to non-dict (line 89)."""
        assert _unwrap_cli_envelope("[1, 2, 3]") == "[1, 2, 3]"
        assert _unwrap_cli_envelope('"just a string"') == '"just a string"'

    def test_unwrap_cli_envelope_dict_result_returns_value_str(self) -> None:
        """_unwrap_cli_envelope returns JSON-dumped dict from result (line 103)."""
        wrapper = json.dumps(
            {
                "type": "result",
                "session_id": "s1",
                "errors": [],
                "result": {"answer": "nested_dict"},
            }
        )
        result = _unwrap_cli_envelope(wrapper)
        assert json.loads(result) == {"answer": "nested_dict"}

    def test_unwrap_cli_envelope_no_result_or_structured_output(self) -> None:
        """_unwrap_cli_envelope returns text if wrapper has no result/structured_output (line 106)."""
        wrapper = json.dumps(
            {
                "type": "error",
                "session_id": "s1",
                "errors": ["something broke"],
            }
        )
        result = _unwrap_cli_envelope(wrapper)
        # Should fall through the for loop and return the original text
        assert json.loads(result)["type"] == "error"

    def test_unwrap_cli_envelope_structured_output_dict(self) -> None:
        """_unwrap_cli_envelope extracts structured_output dict via json.dumps (line 103 via structured_output)."""
        wrapper = json.dumps(
            {
                "type": "result",
                "session_id": "s1",
                "errors": [],
                "structured_output": {"answer": "schema_validated"},
            }
        )
        result = _unwrap_cli_envelope(wrapper)
        assert json.loads(result) == {"answer": "schema_validated"}


@pytest.mark.unit
class TestLocalClaudeProviderAdditional:
    """Additional tests for uncovered lines and branches."""

    @pytest.mark.asyncio
    async def test_structured_output_str_parsing(self) -> None:
        """structured_output as string is parsed via model_validate_json (lines 191-192)."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            provider = LocalClaudeProvider()

        wrapper = {
            "type": "result",
            "result": "fallback text",
            "structured_output": json.dumps({"answer": "str_schema"}),
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

        assert resp.content.answer == "str_schema"

    @pytest.mark.asyncio
    async def test_structured_output_invalid_falls_back(self) -> None:
        """Invalid structured_output falls back to result parsing (lines 193-194)."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            provider = LocalClaudeProvider()

        wrapper = {
            "type": "result",
            "result": json.dumps({"answer": "from_result"}),
            "structured_output": {"bad_field": "wrong_schema"},
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

        # Should fall back to result text parsing since structured_output failed
        assert resp.content.answer == "from_result"

    @pytest.mark.asyncio
    async def test_content_not_basemodel_logs_str_reply(self) -> None:
        """When content is not a BaseModel instance, reply uses str(content) (line 205).

        This is hard to trigger since response_model must be BaseModel subclass,
        but we can verify the branch by mocking _parse_response to return a string.
        """
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

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            patch.object(
                LocalClaudeProvider,
                "_parse_response",
                return_value="plain_string_result",
            ),
        ):
            resp = await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                response_model=_TestModel,
                model="local",
            )

        assert resp.content == "plain_string_result"  # type: ignore[comparison-overlap]

    @pytest.mark.asyncio
    async def test_run_cli_timeout_raises_timeout_error(self) -> None:
        """TimeoutError during communicate kills process and raises (line 205 timeout)."""

        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            provider = LocalClaudeProvider(timeout_seconds=1)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.kill = MagicMock()

        async def mock_wait_for(coro: object, timeout: float) -> None:
            raise TimeoutError("timed out")

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            patch("asyncio.wait_for", side_effect=mock_wait_for),
            pytest.raises(ProviderError),
        ):
            await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                response_model=_TestModel,
                model="local",
            )

        mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_cli_no_model_omits_flag(self) -> None:
        """When model is None, --model flag is not passed (line 300->302 branch)."""
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
            # Call _run_cli directly with model=None and json_schema=None
            result_text, meta = await provider._run_cli(
                "test prompt", model=None, json_schema=None
            )

        assert "--model" not in captured_cmd
        assert "--json-schema" not in captured_cmd

    @pytest.mark.asyncio
    async def test_run_cli_wrapper_no_result_field(self) -> None:
        """CLI wrapper parsed but has no result field (lines 354-360)."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            provider = LocalClaudeProvider()

        # CLI wrapper without "result" key (e.g. error_max_turns subtype)
        wrapper = {
            "type": "error",
            "subtype": "error_max_turns",
            "session_id": "s1",
            "errors": ["max turns reached"],
        }
        json_output = json.dumps(wrapper)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(json_output.encode(), b""),
        )
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result_text, meta = await provider._run_cli("test prompt", model="local")

        # Should return empty string for result_text
        assert result_text == ""
        # Wrapper metadata should be preserved
        assert meta["subtype"] == "error_max_turns"

    @pytest.mark.asyncio
    async def test_run_cli_no_json_fallback(self) -> None:
        """Non-JSON stdout falls through to raw fallback (line 338->362)."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            provider = LocalClaudeProvider()

        # Return plain text, not JSON at all
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(b"Just some plain text output", b""),
        )
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result_text, meta = await provider._run_cli("test prompt", model="local")

        assert result_text == "Just some plain text output"
        assert meta == {}

    def test_parse_cli_json_jsonl_scans_reversed_for_result(self) -> None:
        """_parse_cli_json scans JSONL lines in reverse for type=result (lines 392-401)."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        # Multiple JSONL lines, result is NOT the last line
        lines = [
            json.dumps({"type": "system", "data": "init"}),
            json.dumps({"type": "result", "result": "correct", "session_id": "s1", "errors": []}),
            json.dumps({"type": "assistant", "content": "trailing"}),
        ]
        text = "\n".join(lines)

        # The "result" line should be found (scanned in reverse, assistant line skipped)
        wrapper = LocalClaudeProvider._parse_cli_json(text)
        assert wrapper is not None
        assert wrapper["type"] == "result"

    def test_parse_cli_json_jsonl_with_empty_lines(self) -> None:
        """_parse_cli_json skips empty lines in JSONL (line 395)."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        lines = [
            "",
            json.dumps({"type": "result", "result": "found", "session_id": "s1", "errors": []}),
            "",
            "  ",
        ]
        text = "\n".join(lines)

        wrapper = LocalClaudeProvider._parse_cli_json(text)
        assert wrapper is not None
        assert wrapper["result"] == "found"

    def test_parse_cli_json_jsonl_with_invalid_lines(self) -> None:
        """_parse_cli_json continues past non-JSON lines in JSONL (line 400)."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        lines = [
            "not json at all",
            json.dumps({"type": "result", "result": "ok", "session_id": "s1", "errors": []}),
            "also not json",
        ]
        text = "\n".join(lines)

        wrapper = LocalClaudeProvider._parse_cli_json(text)
        assert wrapper is not None
        assert wrapper["result"] == "ok"

    def test_parse_cli_json_noisy_output_extract_fallback(self) -> None:
        """_parse_cli_json uses _extract_json_object when JSONL fails (lines 404-411)."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        # Text with no valid JSONL result lines, but extractable JSON object.
        # The JSON doesn't have type=result so JSONL scan won't match.
        noisy = 'Loading...\n{"some_key": "some_value"}\nDone.'
        wrapper = LocalClaudeProvider._parse_cli_json(noisy)
        assert wrapper is not None
        assert wrapper["some_key"] == "some_value"

    def test_parse_cli_json_returns_none_for_garbage(self) -> None:
        """_parse_cli_json returns None when nothing is parseable (line 413)."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        assert LocalClaudeProvider._parse_cli_json("total garbage no json") is None

    def test_parse_cli_json_extracted_json_invalid(self) -> None:
        """_parse_cli_json returns None when extracted JSON object is invalid (line 410)."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        # Contains braces but the content between them is not valid JSON
        text = "prefix {not: valid: json} suffix"
        result = LocalClaudeProvider._parse_cli_json(text)
        assert result is None

    def test_parse_response_non_basemodel_raises(self) -> None:
        """_parse_response raises for non-BaseModel response_model (line 449)."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        class NotAModel:
            pass

        with pytest.raises(ResponseValidationError, match="must be a Pydantic BaseModel"):
            LocalClaudeProvider._parse_response('{"answer": "ok"}', NotAModel)  # type: ignore[arg-type,unused-ignore]

    def test_parse_response_markdown_code_block_with_language(self) -> None:
        """_parse_response handles ```json code blocks with content before fence (line 425->434)."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        raw = 'Here is the result:\n```json\n{"answer": "fenced"}\n```\nSome trailing text.'
        result = LocalClaudeProvider._parse_response(raw, _TestModel)
        assert result.answer == "fenced"

    def test_parse_response_code_block_empty_lines(self) -> None:
        """_parse_response with code block that has no JSON lines falls back (line 434->440)."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        # Code block with only whitespace inside — json_lines will be empty
        raw = "```\n```"
        # This should fallback to parsing the cleaned text directly,
        # which will fail validation
        with pytest.raises(ResponseValidationError):
            LocalClaudeProvider._parse_response(raw, _TestModel)

    def test_build_usage_with_model_usage_cost_data(self) -> None:
        """_build_usage extracts costUSD from modelUsage (lines 484-490)."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        wrapper: dict[str, object] = {
            "usage": {"input_tokens": 200, "output_tokens": 80},
            "total_cost_usd": 0.05,
            "modelUsage": {
                "claude-haiku-4-5-20251001": {
                    "costUSD": 0.03,
                    "inputTokens": 200,
                    "outputTokens": 80,
                }
            },
        }
        usage = LocalClaudeProvider._build_usage("prompt", "response", wrapper)
        assert usage.input_tokens == 200
        assert usage.output_tokens == 80
        # Cost should come from modelUsage, not total_cost_usd
        assert usage.total_cost_usd == pytest.approx(0.03, abs=0.001)
        assert usage.input_cost_usd == pytest.approx(0.015, abs=0.001)
        assert usage.output_cost_usd == pytest.approx(0.015, abs=0.001)

    def test_build_usage_model_usage_no_costUSD(self) -> None:
        """_build_usage skips modelUsage entries without costUSD (line 486 branch)."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        wrapper: dict[str, object] = {
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "total_cost_usd": 0.02,
            "modelUsage": {
                "claude-haiku": {"inputTokens": 100}  # No costUSD key
            },
        }
        usage = LocalClaudeProvider._build_usage("prompt", "response", wrapper)
        # Should fall back to total_cost_usd split
        assert usage.total_cost_usd == pytest.approx(0.02, abs=0.001)

    def test_build_usage_model_usage_non_dict_entry(self) -> None:
        """_build_usage handles non-dict model_data in modelUsage (line 486 branch)."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        wrapper: dict[str, object] = {
            "usage": {"input_tokens": 50, "output_tokens": 25},
            "total_cost_usd": 0.01,
            "modelUsage": {"claude-haiku": "not_a_dict"},
        }
        usage = LocalClaudeProvider._build_usage("prompt", "response", wrapper)
        # Should fall back to total_cost_usd
        assert usage.total_cost_usd == pytest.approx(0.01, abs=0.001)

    def test_build_usage_non_dict_usage_data(self) -> None:
        """_build_usage handles non-dict usage data gracefully (line 466->475)."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        wrapper: dict[str, object] = {
            "usage": "not_a_dict",
            "total_cost_usd": 0.0,
        }
        usage = LocalClaudeProvider._build_usage("hello", "world", wrapper)
        # Should fall back to heuristic since usage_data is not a dict
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0

    def test_build_usage_with_cache_tokens(self) -> None:
        """_build_usage includes cache tokens in input count (lines 470-472)."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        wrapper: dict[str, object] = {
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 30,
                "cache_creation_input_tokens": 20,
            },
            "total_cost_usd": 0.01,
        }
        usage = LocalClaudeProvider._build_usage("prompt", "response", wrapper)
        # input_tokens should be 100 + 30 + 20 = 150
        assert usage.input_tokens == 150
        assert usage.output_tokens == 50

    def test_build_usage_non_dict_model_usage(self) -> None:
        """_build_usage handles non-dict modelUsage gracefully (line 484->492)."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        wrapper: dict[str, object] = {
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "total_cost_usd": 0.005,
            "modelUsage": "not_a_dict",
        }
        usage = LocalClaudeProvider._build_usage("p", "r", wrapper)
        # Should use total_cost_usd since modelUsage is not iterable as dict
        assert usage.total_cost_usd == pytest.approx(0.005, abs=0.001)

    @pytest.mark.asyncio
    async def test_image_files_passed_to_run_cli(self) -> None:
        """image_files parameter is forwarded to _run_cli (line 449 image path)."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            provider = LocalClaudeProvider()

        wrapper = {"result": json.dumps({"answer": "img"})}
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
                messages=[{"role": "user", "content": "eval image"}],
                response_model=_TestModel,
                model="local",
                image_files=["/tmp/photo.jpg"],
            )

        assert resp.content.answer == "img"
        # Verify image path appears in prompt
        p_index = captured_cmd.index("-p")
        prompt = captured_cmd[p_index + 1]
        assert "/tmp/photo.jpg" in prompt
        assert "Read tool" in prompt

    @pytest.mark.asyncio
    async def test_complete_default_model(self) -> None:
        """complete() uses DEFAULT_MODEL when model is None (line 159->162)."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            provider = LocalClaudeProvider()

        wrapper = {"result": json.dumps({"answer": "default"})}
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
                messages=[{"role": "user", "content": "test"}],
                response_model=_TestModel,
                # model=None triggers default
            )

        assert resp.model == "claude-haiku-4-5-20251001"
        assert "--model" in captured_cmd
        model_idx = captured_cmd.index("--model")
        assert captured_cmd[model_idx + 1] == "claude-haiku-4-5-20251001"

    def test_build_prompt_no_messages(self) -> None:
        """_build_prompt with empty messages only has schema (line 232->244)."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            provider = LocalClaudeProvider()

        prompt = provider._build_prompt(messages=[], response_model=_TestModel)
        assert "answer" in prompt
        assert "[User]" not in prompt


@pytest.mark.unit
class TestLocalClaudeCountTokens:
    """Tests for LocalClaudeProvider.count_tokens()."""

    def test_count_tokens_returns_positive(self) -> None:
        """count_tokens should return a positive int for non-empty text."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            provider = LocalClaudeProvider()

        result = provider.count_tokens("Hello, world!")
        assert isinstance(result, int)
        assert result > 0

    def test_count_tokens_empty_string(self) -> None:
        """count_tokens should return 0 for empty string."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            provider = LocalClaudeProvider()

        assert provider.count_tokens("") == 0

    def test_count_tokens_lazy_init(self) -> None:
        """Tokenizer should be lazily initialized."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            provider = LocalClaudeProvider()

        assert provider._tokenizer is None
        provider.count_tokens("test")
        assert provider._tokenizer is not None

    def test_count_tokens_consistent(self) -> None:
        """Same text should return same count."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            provider = LocalClaudeProvider()

        text = "The quick brown fox."
        assert provider.count_tokens(text) == provider.count_tokens(text)
