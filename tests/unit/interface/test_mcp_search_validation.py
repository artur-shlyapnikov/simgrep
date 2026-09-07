"""MCP `search` tool must reject empty/whitespace queries like the CLI does."""

from __future__ import annotations

import json
from typing import Any

from simgrep.mcp_server import handle_message


def _call_search(arguments: dict[str, Any]) -> dict[str, Any]:
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "search", "arguments": arguments},
    }
    response = handle_message(request)
    assert response is not None
    return json.loads(response) if isinstance(response, str) else response


def _result_text(response: dict[str, Any]) -> str:
    result = response["result"]
    return "".join(block["text"] for block in result["content"] if block.get("type") == "text")


def test_whitespace_query_is_rejected() -> None:
    response = _call_search({"query": "   "})
    assert response["result"]["isError"] is True
    assert "Query cannot be empty" in _result_text(response)


def test_empty_query_is_rejected() -> None:
    response = _call_search({"query": ""})
    assert response["result"]["isError"] is True
    assert "Query cannot be empty" in _result_text(response)


def test_missing_query_is_rejected_by_validator() -> None:
    response = _call_search({})
    assert response["result"]["isError"] is True
    assert "Missing required argument 'query'" in _result_text(response)
