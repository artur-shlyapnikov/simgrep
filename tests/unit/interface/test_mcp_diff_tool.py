"""Unit tests for the MCP `diff` tool: pinned JSON payload shape, argument
validation conventions, and rename-insensitive semantic tree comparison — all
over a faked engine so no embedding model is ever loaded."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from simgrep.config import save_app_config
from simgrep.diff_engine import DiffEngine as RealDiffEngine
from simgrep.errors import DiffError
from simgrep.mcp_server import handle_message
from simgrep.models import (
    AppConfig,
    DiffEntry,
    DiffOptions,
    DiffOutcome,
    FileRollup,
)


def _entry(label: int, file_path: str, line_start: int, line_end: int) -> DiffEntry:
    return DiffEntry(label=label, file_path=file_path, line_start=line_start, line_end=line_end)


def _rpc(method: str, params: dict[str, Any]) -> dict[str, Any]:
    request = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    response = handle_message(request)
    assert response is not None
    return json.loads(response) if isinstance(response, str) else response


def _call_diff(arguments: Any) -> dict[str, Any]:
    return _rpc("tools/call", {"name": "diff", "arguments": arguments})


def _result_text(response: dict[str, Any]) -> str:
    result = response["result"]
    return "".join(block["text"] for block in result["content"] if block.get("type") == "text")


class FakeDiffEngine:
    """Duck-typed DiffEngine returning a canned outcome or raising; records calls."""

    def __init__(self, runtime: Any, outcome: DiffOutcome | Exception | None = None) -> None:
        self.runtime = runtime
        self.outcome = outcome
        self.calls: list[tuple[Path, Path, DiffOptions | None]] = []

    def diff_paths(
        self,
        path_a: Path,
        path_b: Path,
        app_config: Any,
        options: DiffOptions | None = None,
    ) -> DiffOutcome:
        del app_config
        self.calls.append((path_a, path_b, options))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        assert self.outcome is not None
        return self.outcome


class FakeRuntimeFactory:
    """Duck-typed RuntimeFactory handing back an inert sentinel runtime."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def for_app(self, app_config: Any) -> Any:
        del app_config
        return object()

    def for_project(self, project: Any) -> Any:
        del project
        return object()


@pytest.fixture()
def engine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeDiffEngine:
    """Isolate HOME config and swap the lazy engine seams for one shared fake."""
    home_config = tmp_path / "home" / ".config" / "simgrep"
    home_config.mkdir(parents=True)
    save_app_config(AppConfig(model="fake"), home_config / "config.toml")
    monkeypatch.setenv("HOME", str(home_config.parents[1]))
    import simgrep.diff_engine as diff_engine_module

    instance = FakeDiffEngine(object())

    def build(runtime: Any) -> FakeDiffEngine:
        return instance

    monkeypatch.setattr(diff_engine_module, "DiffEngine", build)
    monkeypatch.setattr("simgrep.execution.RuntimeFactory", FakeRuntimeFactory)
    return instance


def test_tools_list_exposes_diff_with_input_schema() -> None:
    tools = _rpc("tools/list", {})["result"]["tools"]
    tail = [tool["name"] for tool in tools][-5:]
    assert tail == ["status", "index", "diff", "expand", "pack", "debt"][-5:]
    spec = next(tool for tool in tools if tool["name"] == "diff")
    assert "semantic" in spec["description"].lower()
    schema = spec["inputSchema"]
    assert set(schema["required"]) == {"a", "b"}
    assert schema["additionalProperties"] is False
    assert schema["properties"]["a"]["type"] == "string"
    assert schema["properties"]["b"]["type"] == "string"
    assert schema["properties"]["threshold"]["exclusiveMinimum"] == 0
    assert schema["properties"]["threshold"]["maximum"] == 1
    assert schema["properties"]["top"]["minimum"] == 1
    assert schema["properties"]["max_chunks"]["minimum"] == 1


def test_call_returns_pinned_diff_payload(
    engine: FakeDiffEngine,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tree_a = tmp_path / "tree-a"
    tree_b = tmp_path / "tree-b"
    tree_a.mkdir()
    tree_b.mkdir()
    engine.outcome = DiffOutcome(
        added=(_entry(17, "src/new.py", 1, 10),),
        removed=(_entry(3, "src/old.py", 40, 52),),
        matched=12,
        files=(FileRollup(file_path="src/new.py", added=1, removed=0, matched=0),),
        chunks_a=14,
        chunks_b=15,
        threshold=0.9,
    )
    monkeypatch.setattr("simgrep.project.find_active_project", lambda cwd: None)
    response = _call_diff({"a": str(tree_a), "b": str(tree_b), "threshold": 0.9, "top": 7, "max_chunks": 999})
    assert response["result"]["isError"] is False
    payload = json.loads(_result_text(response))
    assert set(payload) >= {"matched", "added", "removed", "files"}
    assert payload["matched"] == 12
    assert payload["chunks_a"] == 14
    assert payload["chunks_b"] == 15
    assert payload["added"][0]["file_path"] == "src/new.py"
    assert payload["removed"][0]["label"] == 3
    assert payload["files"][0] == {"file_path": "src/new.py", "added": 1, "removed": 0, "matched": 0}
    path_a, path_b, options = engine.calls[-1]
    assert path_a == tree_a.resolve()
    assert path_b == tree_b.resolve()
    assert options == DiffOptions(threshold=0.9, top=7, max_chunks=999)


def test_rename_is_invisible_in_diff_payload(engine: FakeDiffEngine, tmp_path: Path) -> None:
    """Identical content merely moved between trees shows up neither as added nor removed."""
    tree_a = tmp_path / "tree-a"
    tree_b = tmp_path / "tree-b"
    tree_a.mkdir()
    tree_b.mkdir()
    engine.outcome = DiffOutcome(
        added=(),
        removed=(),
        matched=3,
        files=(FileRollup(file_path="src/moved.py", added=0, removed=0, matched=3),),
        chunks_a=3,
        chunks_b=3,
        threshold=0.8,
    )
    response = _call_diff({"a": str(tree_a), "b": str(tree_b)})
    assert response["result"]["isError"] is False
    payload = json.loads(_result_text(response))
    assert payload["added"] == []
    assert payload["removed"] == []
    assert payload["matched"] == 3


def test_missing_a_is_tool_error(engine: FakeDiffEngine) -> None:
    response = _call_diff({"b": "/tmp/tree-b"})
    assert response["result"]["isError"] is True
    assert "'a'" in _result_text(response)
    assert engine.calls == []


def test_missing_b_is_tool_error(engine: FakeDiffEngine) -> None:
    response = _call_diff({"a": "/tmp/tree-a"})
    assert response["result"]["isError"] is True
    assert "'b'" in _result_text(response)
    assert engine.calls == []


def test_threshold_above_one_is_rejected(engine: FakeDiffEngine) -> None:
    response = _call_diff({"a": "/tmp/a", "b": "/tmp/b", "threshold": 2.5})
    assert response["result"]["isError"] is True
    assert engine.calls == []


def test_negative_threshold_is_rejected(engine: FakeDiffEngine) -> None:
    """Threshold validity is a domain contract on DiffEngine.diff_paths."""
    with pytest.raises(Exception) as exc:
        RealDiffEngine(None).diff_paths(Path("/tmp/a"), Path("/tmp/b"), AppConfig(model="fake"), DiffOptions(threshold=-0.1))
    assert "Threshold" in str(exc.value)


def test_top_zero_is_rejected(engine: FakeDiffEngine) -> None:
    response = _call_diff({"a": "/tmp/a", "b": "/tmp/b", "top": 0})
    assert response["result"]["isError"] is True
    assert engine.calls == []


def test_threshold_zero_is_rejected(engine: FakeDiffEngine) -> None:
    with pytest.raises(Exception) as exc:
        RealDiffEngine(None).diff_paths(Path("/tmp/a"), Path("/tmp/b"), AppConfig(model="fake"), DiffOptions(threshold=0))
    assert "Threshold" in str(exc.value)


def test_unknown_extra_argument_is_rejected(engine: FakeDiffEngine) -> None:
    response = _call_diff({"a": "/tmp/a", "b": "/tmp/b", "format": "rich"})
    assert response["result"]["isError"] is True
    assert engine.calls == []


def test_diff_error_is_tool_error_with_hint(
    engine: FakeDiffEngine,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tree_a = tmp_path / "tree-a"
    tree_b = tmp_path / "tree-b"
    tree_a.mkdir()
    tree_b.mkdir()
    engine.outcome = DiffError(
        "Chunk budget exceeded: 60000 > max_chunks 50000.",
        hint="Narrow the scope (e.g. a subdirectory) or raise --max-chunks.",
    )
    monkeypatch.setattr("simgrep.project.find_active_project", lambda cwd: None)
    response = _call_diff({"a": str(tree_a), "b": str(tree_b)})
    assert response["result"]["isError"] is True
    text = _result_text(response)
    assert "Chunk budget exceeded" in text
    assert "Hint:" in text
    assert "--max-chunks" in text
