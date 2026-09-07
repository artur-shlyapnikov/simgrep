"""Unit tests for the MCP `expand` tool: pinned CLI-json payload shape,
argument validation, error mapping, and the search `whole_unit` result
transform — all over real files and a faked search runtime so no embedding
model or index build is ever touched."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from simgrep.config import save_app_config
from simgrep.mcp_server import handle_line
from simgrep.models import AppConfig, FileRole, SearchOutcome, SearchResult

# ----------------------------------------------------------------------- helpers


def _rpc(method: str, params: dict[str, Any]) -> dict[str, Any]:
    request = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    response = handle_line(json.dumps(request))
    assert response is not None
    parsed: dict[str, Any] = json.loads(json.dumps(response))
    return parsed


def _call_expand(arguments: Any) -> dict[str, Any]:
    return _rpc("tools/call", {"name": "expand", "arguments": arguments})


def _call_search(arguments: dict[str, Any]) -> dict[str, Any]:
    return _rpc("tools/call", {"name": "search", "arguments": arguments})


def _result_text(response: dict[str, Any]) -> str:
    result = response["result"]
    return "".join(block["text"] for block in result["content"] if block.get("type") == "text")


def _payload(response: dict[str, Any]) -> dict[str, Any]:
    assert response["result"]["isError"] is False
    payload: dict[str, Any] = json.loads(_result_text(response))
    return payload


PAYMENT_FN = (
    "def charge_order(items):\n"
    "    total = sum(items)\n"
    "    receipt = make_receipt(total)\n"
    "    receipt.charge()\n"
    "    log_charge(receipt.id)\n"
    "    audit(receipt)\n"
    "    note(receipt)\n"
    "    return receipt.ok\n"
)
# The enclosing unit always ends BEFORE the last line's newline.
UNIT_TEXT = PAYMENT_FN[:-1]
TAIL_FN = "\n\ndef helper():\n    pass\n"

_JSON_KEYS = {"path", "start_line", "end_line", "start_char", "end_char", "language", "family", "text", "truncated"}

_REGISTRY = {
    "search",
    "similar",
    "clusters",
    "status",
    "index",
    "diff",
    "expand",
    "pack",
    "debt",
}


@pytest.fixture()
def payment_file(tmp_path: Path) -> Path:
    src = tmp_path / "payment.py"
    src.write_text(PAYMENT_FN + TAIL_FN, encoding="utf-8")
    return src


# ----------------------------------------------------------------------- registry


def test_registry_ends_with_debt() -> None:
    tools = _rpc("tools/list", {})["result"]["tools"]
    assert len(tools) == 9  # debt: ninth and final tool
    by_name = {tool["name"]: tool for tool in tools}
    assert set(by_name) == _REGISTRY
    assert [tool["name"] for tool in tools][-1:] == ["debt"]
    spec = by_name["expand"]
    assert "semantic unit" in spec["description"].lower()


def test_expand_schema_pins_bounds_and_required_arguments() -> None:
    tools = _rpc("tools/list", {})["result"]["tools"]
    schema = next(tool for tool in tools if tool["name"] == "expand")["inputSchema"]
    assert schema["type"] == "object"
    assert schema["required"] == ["path", "line"]
    props = schema["properties"]
    assert props["line"]["minimum"] == 1
    assert props["max_chars"]["minimum"] == 200
    assert props["max_chars"]["maximum"] == 200000
    assert props["language"]["enum"] == ["dedent", "brace", "paragraph"]


@pytest.mark.parametrize(
    ("arguments", "fragment"),
    [
        ({"path": "x.py"}, "Missing required argument 'line'"),
        ({"line": 3}, "Missing required argument 'path'"),
        ({"path": "x.py", "line": 0}, "'line' must be >="),
        ({"path": "x.py", "line": 3, "max_chars": 100}, "'max_chars' must be >="),
        ({"path": "x.py", "line": 3, "max_chars": 500000}, "'max_chars' must be <="),
        ({"path": "x.py", "line": 3, "language": "cobol"}, "'language' must be one of"),
        ({"path": "x.py", "line": "three"}, "'line' must be of type integer"),
    ],
)
def test_schema_violations_are_tool_errors(arguments: dict[str, Any], fragment: str) -> None:
    """SEP-1303: input validation failures are isError results, never protocol errors."""
    response = _call_expand(arguments)
    assert response["result"]["isError"] is True
    assert fragment in _result_text(response)


# ----------------------------------------------------------------------- payload


def test_payload_matches_cli_json_contract(payment_file: Path) -> None:
    response = _call_expand({"path": str(payment_file), "line": 3})
    payload = _payload(response)
    assert set(payload) == _JSON_KEYS
    assert payload["path"] == str(payment_file.resolve())
    assert payload["family"] == "dedent"
    assert payload["language"] == "python"
    assert payload["truncated"] is False
    assert payload["start_line"] == 1
    assert payload["end_line"] == 8
    assert payload["start_char"] == 0
    assert payload["end_char"] == len(PAYMENT_FN) - 1
    assert payload["text"] == UNIT_TEXT


def test_language_override_switches_family_in_payload(payment_file: Path) -> None:
    # Paragraph family treats the function as one blank-line-delimited block.
    response = _call_expand({"path": str(payment_file), "line": 3, "language": "paragraph"})
    payload = _payload(response)
    assert payload["family"] == "paragraph"
    assert payload["language"] == "python"  # file language never becomes a family name
    assert payload["start_line"] == 1
    assert payload["end_line"] == 8
    assert payload["text"] == UNIT_TEXT


def test_max_chars_caps_text_but_not_bounds(tmp_path: Path) -> None:
    src = tmp_path / "big.py"
    source = "def big_fn():\n" + "".join(f"    x{i} = {i}\n" for i in range(30))
    src.write_text(source, encoding="utf-8")
    response = _call_expand({"path": str(src), "line": 2, "max_chars": 200})
    payload = _payload(response)
    assert payload["truncated"] is True
    assert payload["text"].endswith("...")
    assert payload["text"] != source.rstrip("\n")
    # Bounds still describe the FULL unit.
    assert payload["start_line"] == 1
    assert payload["end_line"] == 31
    assert payload["start_char"] == 0
    assert payload["end_char"] == len(source) - 1


def test_brace_family_expands_to_matching_brace(tmp_path: Path) -> None:
    src = tmp_path / "main.go"
    src.write_text("func main() {\n\tif ok {\n\t\tfmt.Println(1)\n\t}\n}\n", encoding="utf-8")
    payload = _payload(_call_expand({"path": str(src), "line": 3}))
    assert payload["family"] == "brace"
    assert payload["language"] == "go"
    assert payload["text"].startswith("func main() {")
    assert payload["text"].endswith("}")


# ----------------------------------------------------------------------- errors


def test_line_out_of_range_is_tool_error_with_exit_two_semantics(payment_file: Path) -> None:
    total_lines = len((PAYMENT_FN + TAIL_FN).splitlines())
    response = _call_expand({"path": str(payment_file), "line": total_lines + 1})
    assert response["result"]["isError"] is True
    text = _result_text(response)
    assert "out of range" in text
    assert f"file has {total_lines} lines" in text  # hint appended per SimgrepError convention


def test_missing_file_is_tool_error_with_exit_two_semantics(tmp_path: Path) -> None:
    response = _call_expand({"path": str(tmp_path / "gone.py"), "line": 1})
    assert response["result"]["isError"] is True
    text = _result_text(response)
    assert "Path not found" in text
    assert "Pass an existing PATH" in text  # usage-level hint


def test_lone_cr_file_yields_clean_out_of_range_error(tmp_path: Path) -> None:
    """splitlines() counts lone \r as a break but the \n offset model does not;
    line accounting must follow the offset model, never raise IndexError."""
    src = tmp_path / "cr.txt"
    src.write_bytes(b"alpha\rbeta\r")
    response = _call_expand({"path": str(src), "line": 2})
    assert response["result"]["isError"] is True
    text = _result_text(response)
    assert "out of range" in text
    assert "file has 1 lines" in text


def test_unreadable_file_is_tool_error_with_exit_one_semantics(tmp_path: Path) -> None:
    response = _call_expand({"path": str(tmp_path), "line": 1})  # a directory cannot be read as text
    assert response["result"]["isError"] is True
    text = _result_text(response)
    assert "cannot read" in text
    assert "check that the path exists and is readable" in text  # exit-1 runtime hint


# ----------------------------------------------------------------------- whole_unit


class FakeSearchEngine:
    """Duck-typed SearchEngine returning one canned chunk hit inside PAYMENT_FN."""

    def __init__(self, runtime: Any) -> None:
        del runtime

    def search_project(self, *args: Any, **kwargs: Any) -> SearchOutcome:
        base_path, results = CANNED_CALL
        return SearchOutcome(results=list(results), base_path=base_path)


class FakeRuntimeFactory:
    """Duck-typed RuntimeFactory handing back an inert sentinel runtime."""

    def for_app(self, app_config: Any) -> Any:
        return object()

    def for_project(self, project: Any) -> Any:
        return object()


CANNED_CALL: tuple[Path, list[SearchResult]] = (Path("."), [])


@pytest.fixture()
def fake_search(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Isolate HOME config and swap the lazy search seams for fakes."""
    home_config = tmp_path / "home" / ".config" / "simgrep"
    home_config.mkdir(parents=True)
    save_app_config(AppConfig(model="fake"), home_config / "config.toml")
    monkeypatch.setenv("HOME", str(home_config.parents[1]))

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    src = corpus / "payment.py"
    src.write_text(PAYMENT_FN + TAIL_FN, encoding="utf-8")

    needle = "receipt.charge()"
    offset = PAYMENT_FN.index(needle)
    chunk = SearchResult(
        label=7,
        score=0.9,
        file_path=src,
        chunk_text=needle,
        start_char=offset,
        end_char=offset + len(needle),
        line_start=4,
        line_end=4,
        file_role=FileRole.source,
        language="python",
    )
    global CANNED_CALL
    CANNED_CALL = (corpus, [chunk])

    import simgrep.execution as runtime_module
    import simgrep.project as project_module
    import simgrep.search as search_module

    class FakeActive:
        model = "other-model"
        chunk_size = 64
        chunk_overlap = 8

    monkeypatch.setattr(project_module, "find_active_project", lambda path: FakeActive())
    monkeypatch.setattr(search_module, "SearchEngine", FakeSearchEngine)
    monkeypatch.setattr(runtime_module, "RuntimeFactory", FakeRuntimeFactory)
    return src


def test_search_whole_unit_expands_hits_to_enclosing_unit(fake_search: Path) -> None:
    response = _call_search({"query": "charge", "whole_unit": True})
    records = json.loads(_result_text(response))
    assert len(records) == 1
    record = records[0]
    assert record["text"] == UNIT_TEXT  # full function, not the one-line chunk
    assert record["start_char"] == 0
    assert record["end_char"] == len(PAYMENT_FN) - 1
    assert record["line_start"] == 1
    assert record["line_end"] == 8


@pytest.mark.parametrize("arguments", [{}, {"whole_unit": False}])
def test_search_without_whole_unit_leaves_results_unchanged(arguments: dict[str, Any], fake_search: Path) -> None:
    needle = "receipt.charge()"
    offset = PAYMENT_FN.index(needle)
    response = _call_search({"query": "charge", **arguments})
    records = json.loads(_result_text(response))
    assert len(records) == 1
    assert records[0]["text"] == needle
    assert records[0]["start_char"] == offset
    assert records[0]["end_char"] == offset + len(needle)
