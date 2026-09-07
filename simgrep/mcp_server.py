"""Dependency-free MCP stdio server for simgrep.

Implements the subset of the Model Context Protocol needed to expose simgrep
as a tool over the stdio transport: newline-delimited JSON-RPC 2.0 on stdin,
one response line per request on stdout. stdout carries protocol messages
only; diagnostics never go there. The loop never crashes on malformed input —
protocol violations become JSON-RPC error responses and processing continues
until EOF.

Importing this module is deliberately light: no transformers/torch/model code
is pulled in. Heavy modules (`runtime`, `search`, `indexing`, `store`) are
imported inside handler bodies so `initialize`/`ping`/`tools/list` respond
without loading the embedding model.
"""

from __future__ import annotations

import json
import sys
from typing import IO, Any

from simgrep.tool_registry import SERVER_NAME, TOOLS, TOOLS_BY_NAME, server_version, validate_arguments

SUPPORTED_PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25")
LATEST_PROTOCOL_VERSION = "2025-11-25"
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602


def negotiate_version(requested: str | None) -> str:
    """Echo a supported client-requested protocol version; otherwise latest."""
    if requested in SUPPORTED_PROTOCOL_VERSIONS:
        return requested
    return LATEST_PROTOCOL_VERSION


class ProtocolError(Exception):
    """JSON-RPC protocol violation carrying its error code."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# --- envelope parsing -------------------------------------------------------


def _error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _result_response(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _parse_envelope(message: Any) -> tuple[str, dict[str, Any], Any, bool]:
    """Return (method, params, id, is_notification); raise ProtocolError(-32600)."""
    if not isinstance(message, dict):
        raise ProtocolError(INVALID_REQUEST, "Request must be a JSON object.")
    if message.get("jsonrpc") != "2.0":
        raise ProtocolError(INVALID_REQUEST, "Missing or invalid 'jsonrpc' field (expected \"2.0\").")
    method = message.get("method")
    if not isinstance(method, str):
        raise ProtocolError(INVALID_REQUEST, "Missing or non-string 'method' field.")
    params = message.get("params", {})
    if not isinstance(params, dict):
        raise ProtocolError(INVALID_REQUEST, "'params' must be an object.")
    is_notification = "id" not in message
    request_id = message.get("id")
    if request_id is not None and (isinstance(request_id, bool) or not isinstance(request_id, (str, int))):
        raise ProtocolError(INVALID_REQUEST, "'id' must be a string, integer, or null.")
    return method, params, request_id, is_notification


# --- method dispatch --------------------------------------------------------


def _handle_initialize(params: dict[str, Any]) -> dict[str, Any]:
    requested = params.get("protocolVersion")
    return {
        "protocolVersion": negotiate_version(requested if isinstance(requested, str) else None),
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {"name": SERVER_NAME, "version": server_version()},
    }


def _handle_tools_call(params: dict[str, Any]) -> dict[str, Any]:
    name = params.get("name")
    if not isinstance(name, str) or name not in TOOLS_BY_NAME:
        raise ProtocolError(INVALID_PARAMS, f"Unknown tool: {name!r}")
    spec = TOOLS_BY_NAME[name]
    arguments = params.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ProtocolError(INVALID_PARAMS, "'arguments' must be an object.")
    errors = validate_arguments(spec.input_schema, arguments)
    if errors:
        # SEP-1303: input-validation failures are tool execution errors, not
        # protocol errors — a normal isError result lets the model self-correct.
        return {"content": [{"type": "text", "text": " ".join(errors)}], "isError": True}
    try:
        payload = spec.handler(arguments)
    except Exception as exc:
        text = str(exc)
        hint = getattr(exc, "hint", None)
        if hint:
            text = f"{text} Hint: {hint}"
        return {"content": [{"type": "text", "text": text}], "isError": True}
    # Structured handler result -> MCP text content; serialization is the
    # transport's job, the registry returned a JSON value.
    if isinstance(payload, str):
        text = payload
    else:
        text = json.dumps(payload, indent=2)
    return {"content": [{"type": "text", "text": text}], "isError": False}


def handle_message(message: Any) -> dict[str, Any] | None:
    """Dispatch a parsed JSON-RPC message; None for notifications."""
    try:
        method, params, request_id, is_notification = _parse_envelope(message)
    except ProtocolError as exc:
        return _error_response(None, exc.code, exc.message)
    if is_notification:
        return None
    try:
        result = _dispatch(method, params)
    except ProtocolError as exc:
        return _error_response(request_id, exc.code, exc.message)
    return _result_response(request_id, result)


def _dispatch(method: str, params: dict[str, Any]) -> dict[str, Any]:
    if method == "initialize":
        return _handle_initialize(params)
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": [{"name": t.name, "description": t.description, "inputSchema": t.input_schema} for t in TOOLS]}
    if method == "tools/call":
        return _handle_tools_call(params)
    raise ProtocolError(METHOD_NOT_FOUND, f"Method not found: {method}")


def handle_line(line: str) -> dict[str, Any] | None:
    """Framing layer: one stdin line -> one response dict (None for notifications)."""
    try:
        message = json.loads(line)
    except json.JSONDecodeError:
        return _error_response(None, PARSE_ERROR, "Invalid JSON.")
    return handle_message(message)


def serve(stdin: IO[str] | None = None, stdout: IO[str] | None = None) -> int:
    """Read newline-delimited requests until EOF; exit code 0."""
    stream_in = stdin if stdin is not None else sys.stdin
    stream_out = stdout if stdout is not None else sys.stdout
    for line in stream_in:
        stripped = line.strip()
        if not stripped:
            continue
        response = handle_line(stripped)
        if response is not None:
            stream_out.write(json.dumps(response) + "\n")
            stream_out.flush()
    return 0
