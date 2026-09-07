"""Unit tests for the MCP `search` tool handler's ephemeral-scan branch: a PATH outside
every active project must be scanned on the fly via RuntimeFactory.for_app."""

from __future__ import annotations

from pathlib import Path

import pytest

from simgrep.config import save_app_config
from simgrep.models import AppConfig, ProjectConfig
from simgrep.tool_registry import _tool_search
from tests.conftest import FakeRuntime


class RecordingAppFactory:
    """Duck-typed RuntimeFactory recording the for_app construction path."""

    def __init__(self, runtime: FakeRuntime) -> None:
        self._fake = runtime
        self.for_app_calls = 0

    def for_app(self, config: AppConfig) -> FakeRuntime:
        del config
        self.for_app_calls += 1
        return self._fake

    def for_project(self, config: ProjectConfig) -> FakeRuntime:
        del config
        return self._fake


def test_search_tool_ephemeral_scans_path_outside_any_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_runtime: FakeRuntime) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "note.md").write_text("ephemeral quokka notes\n", encoding="utf-8")
    home_config = tmp_path / "home" / ".config" / "simgrep"
    home_config.mkdir(parents=True)
    save_app_config(AppConfig(model="fake"), home_config / "config.toml")
    monkeypatch.setenv("HOME", str(home_config.parents[1]))
    factory = RecordingAppFactory(fake_runtime)
    monkeypatch.setattr("simgrep.execution.RuntimeFactory", lambda: factory)

    payload = _tool_search({"query": "quokka notes", "path": str(outside)})
    assert isinstance(payload, list)
    assert any("note.md" in str(r["path"]) for r in payload)
    assert factory.for_app_calls >= 1
