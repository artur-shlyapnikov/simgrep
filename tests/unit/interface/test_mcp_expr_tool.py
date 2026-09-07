"""Unit tests for the MCP `search` tool's `expr` support: schema surface,
mutual exclusion with `query`, expr-only option construction (pure semantic
scoring), and error envelopes — all over a faked engine so no embedding model
is ever loaded."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from simgrep.config import save_app_config
from simgrep.mcp_server import handle_message
from simgrep.models import (
    AppConfig,
    FileRole,
    ProjectConfig,
    SearchOptions,
    SearchOutcome,
    SearchResult,
)

PROJECT_ROOT = Path("/tmp/fake-project")


def _result(label: int, name: str, score: float) -> SearchResult:
    return SearchResult(
        label=label,
        score=score,
        file_path=PROJECT_ROOT / "src" / name,
        chunk_text=f"def handler_{label}(): ...",
        start_char=0,
        end_char=24,
        line_start=1,
        line_end=2,
        file_role=FileRole.source,
        language="python",
    )


def _outcome(labels_scores: list[tuple[int, str, float]]) -> SearchOutcome:
    return SearchOutcome(
        results=[_result(label, name, score) for label, name, score in labels_scores],
        base_path=PROJECT_ROOT,
        files_seen=3,
        chunks_searched=10,
        semantic_candidates=len(labels_scores),
    )


QUERY_OUTCOME = _outcome([(7, "a.py", 0.91), (11, "b.py", 0.42)])
EXPR_OUTCOME = _outcome([(17, "c.py", 0.88), (23, "d.py", 0.13)])


def _project() -> ProjectConfig:
    return ProjectConfig(
        schema_version=1,
        name="fake",
        root=PROJECT_ROOT,
        indexed_paths=(PROJECT_ROOT,),
        model="fake",
        chunk_size=512,
        chunk_overlap=64,
    )


def _rpc(method: str, params: dict[str, Any]) -> dict[str, Any]:
    request = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    response = handle_message(request)
    assert response is not None
    return json.loads(response) if isinstance(response, str) else response


def _call_search(arguments: Any) -> dict[str, Any]:
    return _rpc("tools/call", {"name": "search", "arguments": arguments})


def _result_text(response: dict[str, Any]) -> str:
    result = response["result"]
    return "".join(block["text"] for block in result["content"] if block.get("type") == "text")


class FakeSearchEngine:
    """Duck-typed SearchEngine returning canned outcomes; records calls."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.calls: list[tuple[str, SearchOptions]] = []

    def search_project(self, project: Any, app_config: Any, options: SearchOptions, freshness: Any = None) -> SearchOutcome:
        del project, app_config, freshness
        self.calls.append(("search_project", options))
        return EXPR_OUTCOME if options.expr is not None else QUERY_OUTCOME

    def search_path(self, path: Any, app_config: Any, options: SearchOptions, ephemeral: Any = None) -> SearchOutcome:
        del path, app_config, ephemeral
        self.calls.append(("search_path", options))
        return EXPR_OUTCOME if options.expr is not None else QUERY_OUTCOME


class CallsRecorder:
    """Aggregates calls across every engine instance the handler builds."""

    def __init__(self) -> None:
        self.created: list[FakeSearchEngine] = []

    @property
    def calls(self) -> list[tuple[str, SearchOptions]]:
        return [call for engine in self.created for call in engine.calls]


class FakeRuntimeFactory:
    """Duck-typed RuntimeFactory handing back an inert sentinel runtime."""

    def for_app(self, config: AppConfig) -> object:
        del config
        return object()

    def for_project(self, config: ProjectConfig) -> object:
        del config
        return object()


@pytest.fixture()
def fake_engine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> CallsRecorder:
    """Isolate HOME config and swap the lazy engine seams for fakes."""
    home_config = tmp_path / "home" / ".config" / "simgrep"
    home_config.mkdir(parents=True)
    save_app_config(AppConfig(model="fake"), home_config / "config.toml")
    monkeypatch.setenv("HOME", str(home_config.parents[1]))
    import simgrep.search as search_module

    recorder = CallsRecorder()

    def build(runtime: Any) -> FakeSearchEngine:
        instance = FakeSearchEngine(runtime)
        recorder.created.append(instance)
        return instance

    monkeypatch.setattr(search_module, "SearchEngine", build)
    monkeypatch.setattr("simgrep.execution.RuntimeFactory", FakeRuntimeFactory)
    monkeypatch.setattr("simgrep.project.find_active_project", lambda cwd: _project())
    return recorder


def test_tools_list_nine_tools_and_schema_exposes_optional_expr() -> None:
    tools = _rpc("tools/list", {})["result"]["tools"]
    assert len(tools) == 9
    by_name = {tool["name"]: tool for tool in tools}
    schema = by_name["search"]["inputSchema"]
    assert schema["required"] == ["query"]
    assert schema["properties"]["expr"] == {
        "type": "string",
        "description": 'Boolean semantic expression (UPPERCASE AND/OR/NOT, quotes for phrases). Send with query:"" when using expr.',
    }


def test_query_and_expr_mutual_exclusion_is_error_envelope(fake_engine: CallsRecorder) -> None:
    response = _call_search({"query": "auth", "expr": "(auth OR login) AND NOT oauth"})
    assert response["result"]["isError"] is True
    text = _result_text(response)
    assert "query and expr are mutually exclusive" in text
    assert fake_engine.calls == []


def test_search_with_expr_happy_path_json_array_matches_query_shape(fake_engine: CallsRecorder) -> None:
    response = _call_search({"query": "", "expr": "(auth OR login) AND NOT oauth"})
    assert response["result"]["isError"] is False
    expr_rows = json.loads(_result_text(response))
    query_rows = json.loads(_result_text(_call_search({"query": "auth"})))
    assert isinstance(expr_rows, list) and len(expr_rows) == 2
    assert set(expr_rows[0]) == set(query_rows[0])
    assert {row["path"] for row in expr_rows} == {"src/c.py", "src/d.py"}
    expr_mode, expr_options = fake_engine.calls[0]
    assert expr_mode == "search_project"
    assert expr_options.expr == "(auth OR login) AND NOT oauth"
    assert expr_options.query == expr_options.expr
    assert (expr_options.lexical_top, expr_options.lexical_weight) == (0, 0.0)
    _, query_options = fake_engine.calls[1]
    assert query_options.expr is None


def test_expr_only_routes_through_persistent_project_path(fake_engine: CallsRecorder) -> None:
    response = _call_search({"query": "", "expr": "cache AND invalidation"})
    assert response["result"]["isError"] is False
    payload = json.loads(_result_text(response))
    assert isinstance(payload, list) and len(payload) == 2
    mode, options = fake_engine.calls[-1]
    assert mode == "search_project"
    assert options.expr == "cache AND invalidation"


def test_expr_with_uncovered_path_routes_ephemeral(fake_engine: CallsRecorder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    uncovered = tmp_path / "outside"
    uncovered.mkdir()
    monkeypatch.setattr("simgrep.project.project_covers_path", lambda active, path: False)
    response = _call_search({"query": "", "expr": "alpha OR beta", "path": str(uncovered)})
    assert response["result"]["isError"] is False
    payload = json.loads(_result_text(response))
    assert isinstance(payload, list) and len(payload) == 2
    mode, options = fake_engine.calls[-1]
    assert mode == "search_path"
    assert options.expr == "alpha OR beta"
    assert (options.lexical_top, options.lexical_weight) == (0, 0.0)


def test_empty_expr_falls_back_to_query_validation_error(fake_engine: CallsRecorder) -> None:
    response = _call_search({"query": "", "expr": "   "})
    assert response["result"]["isError"] is True
    assert "Expression cannot be empty." in _result_text(response)
    assert "Provide non-empty expr text or omit it to search by query." in _result_text(response)
    assert fake_engine.calls == []


def test_query_only_options_keep_hybrid_defaults(fake_engine: CallsRecorder) -> None:
    response = _call_search({"query": "auth"})
    assert response["result"]["isError"] is False
    payload = json.loads(_result_text(response))
    assert isinstance(payload, list) and len(payload) == 2
    _, options = fake_engine.calls[-1]
    assert options.expr is None
    assert (options.lexical_top, options.lexical_weight) == (50, 0.25)
