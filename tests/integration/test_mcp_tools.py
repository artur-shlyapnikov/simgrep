"""Integration tests for MCP tool handlers against the FakeRuntime engine in a
tmp project: search payload parity + ephemeral path, similar (incl. '-' rejection
and '@file'), status, index summary, and the no-project error path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from simgrep.config import save_app_config
from simgrep.indexing import IndexEngine
from simgrep.mcp_server import handle_message
from simgrep.models import (
    SCHEMA_VERSION,
    AppConfig,
    IndexOptions,
    ProjectConfig,
    RenderOptions,
    ResultFormat,
)
from simgrep.output import enrich_result, format_json
from simgrep.project import find_active_project, init_project
from simgrep.search import SearchEngine, SearchOptions
from tests.conftest import FakeRuntime

COMMON = "def retry_request():\n    return retry(request)\n"


class FakeRuntimeFactory:
    """Duck-typed stand-in for `RuntimeFactory` (the tool handlers only call for_app/for_project)."""

    def __init__(self, runtime: FakeRuntime) -> None:
        self._fake = runtime

    def for_app(self, config: AppConfig) -> FakeRuntime:
        del config
        return self._fake

    def for_project(self, config: ProjectConfig) -> FakeRuntime:
        del config
        return self._fake


class RecordingRuntimeFactory(FakeRuntimeFactory):
    """FakeRuntimeFactory that records which construction path handlers choose."""

    def __init__(self, runtime: FakeRuntime) -> None:
        super().__init__(runtime)
        self.for_app_calls: int = 0
        self.for_project_calls: int = 0

    def for_app(self, config: AppConfig) -> FakeRuntime:
        self.for_app_calls += 1
        return super().for_app(config)

    def for_project(self, config: ProjectConfig) -> FakeRuntime:
        self.for_project_calls += 1
        return super().for_project(config)


def _isolate_global_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str = "fake") -> None:
    """Point HOME at a tmp dir whose global config matches `model` (default: the project's)."""
    config_file = tmp_path / "home" / ".config" / "simgrep" / "config.toml"
    config_file.parent.mkdir(parents=True)
    save_app_config(AppConfig(model=model), config_file)
    monkeypatch.setenv("HOME", str(config_file.parents[2]))


def _expected_options(query: str) -> SearchOptions:
    """SearchOptions exactly as `_tool_search` builds them for the fake project."""
    app_config = AppConfig(model="fake")
    return SearchOptions(query=query, lexical_top=app_config.lexical_top, lexical_weight=app_config.lexical_weight)


@pytest.fixture
def mcp_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_runtime: FakeRuntime) -> Path:
    """Indexed tmp project with cwd inside it and RuntimeFactory faked out."""
    (tmp_path / "a.py").write_text(COMMON + "MARKER_A unique alpha\n", encoding="utf-8")
    (tmp_path / "b.py").write_text(COMMON + "MARKER_B unique beta\n", encoding="utf-8")
    app_config = AppConfig(model="fake")
    init_project(tmp_path, app_config)
    project = find_active_project(tmp_path)
    assert project is not None
    IndexEngine(fake_runtime).index_project(project, app_config, IndexOptions(rebuild=True))
    monkeypatch.setattr("simgrep.execution.RuntimeFactory", lambda: FakeRuntimeFactory(fake_runtime))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _call_tool(name: str, arguments: dict[str, Any] | None = None, request_id: int = 1) -> dict[str, Any]:
    params: dict[str, Any] = {"name": name}
    if arguments is not None:
        params["arguments"] = arguments
    response = handle_message({"jsonrpc": "2.0", "id": request_id, "method": "tools/call", "params": params})
    assert response is not None
    result = response["result"]
    assert isinstance(result, dict)
    assert len(result["content"]) == 1
    assert result["content"][0]["type"] == "text"
    return {"isError": bool(result["isError"]), "text": str(result["content"][0]["text"])}


def _payload(call: dict[str, Any]) -> list[dict[str, Any]]:
    assert not call["isError"]
    parsed = json.loads(call["text"])
    assert isinstance(parsed, list)
    return parsed


class TestSearchTool:
    def test_payload_matches_cli_json_pipeline(self, mcp_project: Path, fake_runtime: FakeRuntime) -> None:
        call = _call_tool("search", {"query": "retry_request"})
        records = _payload(call)

        app_config = AppConfig(model="fake")
        project = ProjectConfig(SCHEMA_VERSION, "p", mcp_project, (mcp_project,), "fake", 128, 20)
        outcome = SearchEngine(fake_runtime).search_project(project, app_config, _expected_options("retry_request"), app_config.freshness)
        expected = format_json(
            [
                enrich_result(
                    r,
                    RenderOptions(format=ResultFormat.json, base_path=outcome.base_path, query="retry_request"),
                )
                for r in outcome.results
            ],
            show_scores=True,
            show_why=False,
        )
        assert json.loads(call["text"]) == json.loads(expected)
        assert {Path(r["path"]).name for r in records} >= {"a.py", "b.py"}

    def test_ephemeral_search_outside_project(self, mcp_project: Path, tmp_path: Path) -> None:
        other = tmp_path / "elsewhere"
        other.mkdir()
        (other / "note.md").write_text("ephemeral quokka notes\n", encoding="utf-8")
        call = _call_tool("search", {"query": "quokka notes", "path": str(other)})
        records = _payload(call)
        assert any("note.md" in r["path"] for r in records)

    def test_persistent_search_outside_project_is_tool_error(self, mcp_project: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
        other = tmp_path_factory.mktemp("outside_project")
        (other / "note.md").write_text("ephemeral quokka notes\n", encoding="utf-8")
        call = _call_tool("search", {"query": "quokka notes", "path": str(other), "persistent": True})
        assert call["isError"]
        assert "active project" in call["text"].lower()

    def test_top_plumbing_limits_results(self, mcp_project: Path, tmp_path: Path) -> None:
        for i in range(4):
            (tmp_path / f"top_{i}.py").write_text(COMMON + f"TOP_MARKER_{i} distinct\n", encoding="utf-8")
        call = _call_tool("search", {"query": COMMON.strip().split()[0], "top": 1})
        assert not call["isError"]
        assert len(_payload(call)) == 1
        why_call = _call_tool("search", {"query": COMMON.strip().split()[0], "top": 1, "show_why": True})
        assert not why_call["isError"]
        assert len(_payload(why_call)) <= 1


class TestSimilarTool:
    def test_at_file_anchor_excludes_own_chunks(self, mcp_project: Path) -> None:
        anchor = mcp_project / "a.py"
        call = _call_tool("similar", {"source": f"@{anchor}"})
        records = _payload(call)
        names = [str(Path(r["path"]).name) for r in records]
        assert "a.py" not in names
        assert "b.py" in names

    def test_dash_anchor_is_rejected_as_error(self, mcp_project: Path) -> None:
        call = _call_tool("similar", {"source": "-"})
        assert call["isError"]
        assert "stdin" in call["text"].lower()

    def test_unlike_dash_is_rejected_as_error(self, mcp_project: Path) -> None:
        call = _call_tool("similar", {"source": "@a.py", "unlike": "-"})
        assert call["isError"]
        assert "stdin" in call["text"].lower()

    def test_unlike_anchor_contrastive_path_succeeds(self, mcp_project: Path) -> None:
        call = _call_tool("similar", {"source": f"@{mcp_project / 'a.py'}", "unlike": f"@{mcp_project / 'b.py'}"})
        assert not call["isError"]
        records = _payload(call)
        assert isinstance(records, list)
        # include_self defaults to False: the like-anchor file must be absent.
        # NOTE: `unlike` demotes b.py, it does not exclude it — never assert its absence.
        assert "a.py" not in {str(Path(r["path"]).name) for r in records}


class TestStatusAndIndexTools:
    def test_status_reports_project_fields(self, mcp_project: Path) -> None:
        call = _call_tool("status")
        assert not call["isError"]
        payload = json.loads(call["text"])
        assert payload["project_root"] == str(mcp_project)
        assert payload["files"] > 0
        assert payload["chunks"] > 0
        assert payload["index_state"] in {"ready", None}
        assert isinstance(payload["indexed_paths"], list)
        assert payload["model"]

    def test_index_returns_summary_counts(self, mcp_project: Path) -> None:
        (mcp_project / "new_file.py").write_text("brand new content gamma\n", encoding="utf-8")
        call = _call_tool("index")
        summary = json.loads(call["text"])
        assert not call["isError"]
        assert summary["files_indexed"] == 1
        assert summary["chunks_indexed"] >= 1


class TestProjectRuntimeSelection:
    def test_matching_overrides_reuse_app_runtime(self, mcp_project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_runtime: FakeRuntime) -> None:
        _isolate_global_config(tmp_path, monkeypatch)
        factory = RecordingRuntimeFactory(fake_runtime)
        monkeypatch.setattr("simgrep.execution.RuntimeFactory", lambda: factory)

        call = _call_tool("index")
        assert not call["isError"]
        assert factory.for_app_calls >= 1
        assert factory.for_project_calls == 0

    def test_diverging_overrides_use_project_runtime(
        self, mcp_project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_runtime: FakeRuntime
    ) -> None:
        _isolate_global_config(tmp_path, monkeypatch)
        factory = RecordingRuntimeFactory(fake_runtime)
        monkeypatch.setattr("simgrep.execution.RuntimeFactory", lambda: factory)
        init_project(mcp_project, AppConfig(model="fake", chunk_size=64), yes=True)

        call = _call_tool("index")
        assert not call["isError"]
        assert factory.for_project_calls >= 1


class TestNoProjectErrors:
    def test_search_without_project_is_error_with_init_hint(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_runtime: FakeRuntime) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setattr("simgrep.execution.RuntimeFactory", lambda: FakeRuntimeFactory(fake_runtime))
        monkeypatch.chdir(empty)
        call = _call_tool("search", {"query": "anything"})
        assert call["isError"]
        assert "simgrep init" in call["text"]

    def test_status_without_project_is_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_runtime: FakeRuntime) -> None:
        empty = tmp_path / "empty2"
        empty.mkdir()
        monkeypatch.setattr("simgrep.execution.RuntimeFactory", lambda: FakeRuntimeFactory(fake_runtime))
        monkeypatch.chdir(empty)
        call = _call_tool("status")
        assert call["isError"]
