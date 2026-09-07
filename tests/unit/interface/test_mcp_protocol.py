"""Pure-protocol unit tests for the MCP stdio server: framing, dispatch,
version negotiation, error codes, envelope validation, notifications, and
tool schema/argument validation. No engine I/O happens in this module.
(tool-handler error paths run without any engine)."""

from __future__ import annotations

import io
import json
import pathlib
import subprocess
import sys
from typing import Any

import pytest

from simgrep.errors import SearchError
from simgrep.mcp_server import (
    LATEST_PROTOCOL_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS,
    handle_line,
    handle_message,
    negotiate_version,
    serve,
)
from simgrep.tool_registry import (
    _tool_similar,
    server_version,
    validate_arguments,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

# Runs in a subprocess: the pytest session itself may already hold heavy modules.
_LAZY_IMPORT_SCRIPT = r"""
import io
import json
import sys

import simgrep.mcp_server as mcp_server

request = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
           "params": {"protocolVersion": "2025-06-18"}}
stream_out = io.StringIO()
code = mcp_server.serve(stdin=io.StringIO(json.dumps(request) + "\n"), stdout=stream_out)
heavy_roots = {"sentence_transformers", "transformers", "torch", "huggingface_hub", "unstructured"}
heavy = sorted({name.split(".")[0] for name in sys.modules} & heavy_roots)
print("CODE:" + str(code))
print("HEAVY:" + ",".join(heavy))
print("OUT:" + stream_out.getvalue().strip())
"""


def _request(method: str, request_id: int | str | None = 1, params: dict[str, Any] | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if request_id is not None:
        message["id"] = request_id
    if params is not None:
        message["params"] = params
    return message


def _error_code(response: dict[str, Any]) -> int:
    error = response.get("error")
    assert isinstance(error, dict)
    return int(error["code"])


class TestVersionNegotiation:
    @pytest.mark.parametrize("requested", sorted(SUPPORTED_PROTOCOL_VERSIONS))
    def test_supported_versions_are_echoed(self, requested: str) -> None:
        assert negotiate_version(requested) == requested

    @pytest.mark.parametrize("requested", ["1999-01-01", "", "2025-13-99"])
    def test_unsupported_versions_fall_back_to_latest(self, requested: str) -> None:
        assert negotiate_version(requested) == LATEST_PROTOCOL_VERSION

    def test_missing_version_falls_back_to_latest(self) -> None:
        assert negotiate_version(None) == LATEST_PROTOCOL_VERSION

    def test_legacy_ceiling_version_is_echoed(self) -> None:
        assert negotiate_version("2025-11-25") == "2025-11-25"

    def test_unknown_request_replies_with_2025_11_25(self) -> None:
        assert negotiate_version("2026-07-28") == "2025-11-25"

    def test_initialize_echoes_supported_version_and_shape(self) -> None:
        response = handle_message(_request("initialize", 1, {"protocolVersion": "2024-11-05"}))
        assert response is not None
        result = response["result"]
        assert result["protocolVersion"] == "2024-11-05"
        assert result["capabilities"] == {"tools": {"listChanged": False}}
        assert result["serverInfo"]["name"] == "simgrep"
        assert isinstance(result["serverInfo"]["version"], str) and result["serverInfo"]["version"]

    def test_initialize_without_params_uses_latest(self) -> None:
        response = handle_message(_request("initialize", 1, {}))
        assert response is not None
        assert response["result"]["protocolVersion"] == LATEST_PROTOCOL_VERSION

    def test_server_version_falls_back_when_package_metadata_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from importlib.metadata import PackageNotFoundError

        import simgrep.tool_registry as registry_module

        def missing(name: str) -> str:
            raise PackageNotFoundError(name)

        monkeypatch.setattr(registry_module, "version", missing)
        assert server_version() == "0.1.0"


class TestCoreMethods:
    def test_ping_returns_empty_result(self) -> None:
        response = handle_message(_request("ping", 3))
        assert response is not None
        assert response["id"] == 3
        assert response["result"] == {}

    def test_tools_list_has_exactly_nine_tools(self) -> None:
        response = handle_message(_request("tools/list", 2))
        assert response is not None
        tools = response["result"]["tools"]
        assert {tool["name"] for tool in tools} == {
            "search",
            "similar",
            "clusters",
            "diff",
            "status",
            "index",
            "expand",
            "pack",
            "debt",
        }
        assert len(tools) == 9

    def test_every_tool_schema_is_a_valid_object_schema(self) -> None:
        response = handle_message(_request("tools/list", 2))
        assert response is not None
        for tool in response["result"]["tools"]:
            schema = tool["inputSchema"]
            assert schema["type"] == "object"
            assert isinstance(schema.get("properties"), dict)
            required = schema.get("required", [])
            assert isinstance(required, list)
            assert set(required) <= set(schema["properties"])

    def test_required_fields_match_spec(self) -> None:
        response = handle_message(_request("tools/list", 2))
        assert response is not None
        schemas = {tool["name"]: tool["inputSchema"] for tool in response["result"]["tools"]}
        assert schemas["search"]["required"] == ["query"]
        assert schemas["similar"]["required"] == ["source"]
        assert "required" not in schemas["status"]
        assert "required" not in schemas["index"]


class TestErrorCodes:
    def test_unknown_method_returns_minus_32601(self) -> None:
        response = handle_message(_request("foo/bar", 7))
        assert response is not None
        assert response["id"] == 7
        assert _error_code(response) == -32601

    def test_malformed_json_line_is_parse_error_with_null_id(self) -> None:
        response = handle_line("{not json")
        assert response is not None
        assert response["id"] is None
        assert _error_code(response) == -32700

    @pytest.mark.parametrize(
        ("raw", "reason"),
        [
            ("[]", "non-object envelope"),
            ("42", "scalar envelope"),
            ('{"method": "ping"}', "missing jsonrpc"),
            ('{"jsonrpc": "1.0", "id": 1, "method": "ping"}', "wrong jsonrpc version"),
            ('{"jsonrpc": "2.0", "id": 1}', "missing method"),
            ('{"jsonrpc": "2.0", "id": 1, "method": 5}', "non-string method"),
            ('{"jsonrpc": "2.0", "id": 1.5, "method": "ping"}', "float id"),
            ('{"jsonrpc": "2.0", "id": [1], "method": "ping"}', "list id"),
            ('{"jsonrpc": "2.0", "id": true, "method": "ping"}', "boolean id"),
        ],
    )
    def test_invalid_envelopes_return_minus_32600(self, raw: str, reason: str) -> None:
        response = handle_line(raw)
        assert response is not None, reason
        assert _error_code(response) == -32600, reason

    def test_non_object_params_rejected(self) -> None:
        response = handle_line('{"jsonrpc": "2.0", "id": 1, "method": "ping", "params": []}')
        assert response is not None
        assert _error_code(response) == -32600


class TestNotifications:
    @pytest.mark.parametrize(
        "message",
        [
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "method": "some/unknown/notification"},
            {"jsonrpc": "2.0", "method": "unknown/method-without-id"},
        ],
    )
    def test_notifications_produce_no_response(self, message: dict[str, Any]) -> None:
        assert handle_message(message) is None


class TestArgumentValidation:
    SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top": {"type": "integer", "minimum": 1},
            "min_score": {"type": "number", "minimum": 0, "maximum": 1},
            "include": {"type": "array", "items": {"type": "string"}},
            "diversity": {"type": "string", "enum": ["none", "window", "file", "package"]},
        },
        "required": ["query"],
    }

    def test_missing_required_argument_is_reported(self) -> None:
        errors = validate_arguments(self.SCHEMA, {})
        assert any("query" in err for err in errors)

    @pytest.mark.parametrize(
        ("arguments", "bad_key"),
        [
            ({"query": 5}, "query"),
            ({"query": "x", "top": "ten"}, "top"),
            ({"query": "x", "top": True}, "top"),
            ({"query": "x", "min_score": "high"}, "min_score"),
            ({"query": "x", "include": [1]}, "include"),
            ({"query": "x", "diversity": "bogus"}, "diversity"),
            ({"query": "x", "mystery": 1}, "mystery"),
        ],
    )
    def test_invalid_arguments_are_reported(self, arguments: dict[str, Any], bad_key: str) -> None:
        errors = validate_arguments(self.SCHEMA, arguments)
        assert errors, f"expected rejection of '{bad_key}'"
        assert any(bad_key in err for err in errors)

    @pytest.mark.parametrize(
        ("arguments", "bad_key", "bound"),
        [
            ({"query": "x", "top": -5}, "top", ">="),
            ({"query": "x", "top": 0}, "top", ">="),
            ({"query": "x", "min_score": 1.5}, "min_score", "<="),
        ],
    )
    def test_bound_violations_are_reported(self, arguments: dict[str, Any], bad_key: str, bound: str) -> None:
        errors = validate_arguments(self.SCHEMA, arguments)
        assert any(bad_key in err and bound in err for err in errors), f"expected {bound} bound error for '{bad_key}'"

    def test_boundary_values_pass(self) -> None:
        arguments = {"query": "x", "top": 1, "min_score": 0}
        assert validate_arguments(self.SCHEMA, arguments) == []

    def test_upper_boundary_value_passes(self) -> None:
        arguments = {"query": "x", "min_score": 1.0}
        assert validate_arguments(self.SCHEMA, arguments) == []

    def test_valid_arguments_pass(self) -> None:
        arguments = {"query": "retry logic", "top": 3, "min_score": 0.5, "include": ["*.py"], "diversity": "file"}
        assert validate_arguments(self.SCHEMA, arguments) == []


class TestToolsCallValidation:
    def test_unknown_tool_is_minus_32602(self) -> None:
        response = handle_message(_request("tools/call", 9, {"name": "nope", "arguments": {}}))
        assert response is not None
        assert _error_code(response) == -32602

    def test_non_string_tool_name_is_minus_32602(self) -> None:
        response = handle_message(_request("tools/call", 9, {"name": 4}))
        assert response is not None
        assert _error_code(response) == -32602

    def test_invalid_search_arguments_are_is_error_result_not_32602(self) -> None:
        response = handle_message(_request("tools/call", 9, {"name": "search", "arguments": {"top": "many"}}))
        assert response is not None
        assert "error" not in response
        result = response["result"]
        assert result["isError"] is True
        assert "top" in result["content"][0]["text"]

    def test_missing_required_query_is_is_error_result_not_32602(self) -> None:
        response = handle_message(_request("tools/call", 9, {"name": "search", "arguments": {}}))
        assert response is not None
        assert "error" not in response
        result = response["result"]
        assert result["isError"] is True
        assert "query" in result["content"][0]["text"]

    def test_out_of_bound_argument_is_is_error_result_with_bound_message(self) -> None:
        response = handle_message(_request("tools/call", 9, {"name": "search", "arguments": {"query": "x", "top": -5}}))
        assert response is not None
        assert "error" not in response
        result = response["result"]
        assert result["isError"] is True
        assert ">=" in result["content"][0]["text"]

    def test_non_object_arguments_are_minus_32602(self) -> None:
        response = handle_message(_request("tools/call", 9, {"name": "status", "arguments": [1]}))
        assert response is not None
        assert _error_code(response) == -32602


class TestServeLoop:
    def _run(self, lines: list[str]) -> tuple[int, list[dict[str, Any]]]:
        stream_in = io.StringIO("\n".join(lines) + "\n")
        stream_out = io.StringIO()
        code = serve(stdin=stream_in, stdout=stream_out)
        out_lines = [line for line in stream_out.getvalue().splitlines() if line]
        parsed = [json.loads(line) for line in out_lines]
        return code, parsed

    def test_eof_exit_code_zero(self) -> None:
        stream_in = io.StringIO("")
        assert serve(stdin=stream_in, stdout=io.StringIO()) == 0

    def test_blank_line_is_skipped_silently(self) -> None:
        code, responses = self._run(["   ", '{"jsonrpc": "2.0", "id": 1, "method": "ping"}'])
        assert code == 0
        assert len(responses) == 1
        assert responses[0]["result"] == {}

    def test_mixed_stream_keeps_loop_alive_until_eof(self) -> None:
        code, responses = self._run(
            [
                "{broken",
                '{"jsonrpc": "2.0", "method": "notifications/initialized"}',
                '{"jsonrpc": "2.0", "id": 1, "method": "no/such"}',
                '{"jsonrpc": "2.0", "id": 2, "method": "ping"}',
            ]
        )
        assert code == 0
        assert len(responses) == 3
        assert _error_code(responses[0]) == -32700
        assert _error_code(responses[1]) == -32601
        assert responses[2]["result"] == {}

    def test_request_ids_are_preserved_verbatim(self) -> None:
        _, responses = self._run(['{"jsonrpc": "2.0", "id": "client-1", "method": "ping"}'])
        assert responses[0]["id"] == "client-1"

    def test_explicit_null_id_request_is_answered_with_null_id(self) -> None:
        response = handle_message({"jsonrpc": "2.0", "id": None, "method": "ping"})
        assert response == {"jsonrpc": "2.0", "id": None, "result": {}}

    def test_importing_mcp_server_pulls_no_heavy_modules(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-c", _LAZY_IMPORT_SCRIPT],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        fields = dict(line.split(":", 1) for line in proc.stdout.splitlines() if ":" in line)
        assert fields["CODE"] == "0"
        assert fields["HEAVY"] == ""
        response = json.loads(fields["OUT"])
        assert response["id"] == 1
        assert response["result"]["protocolVersion"] == "2025-06-18"


def test_similar_tool_requires_active_project(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(empty)
    with pytest.raises(SearchError, match="No active project found") as err:
        _tool_similar({"source": "plain anchor text"})
    assert err.value.hint is not None and "simgrep init" in err.value.hint
