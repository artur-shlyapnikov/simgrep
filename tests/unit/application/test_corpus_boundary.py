"""Table-driven application-boundary tests: one policy table pinning the corpus
decisions that CLI, MCP and HTTP all inherit through
:func:`simgrep.execution.open_resolved_corpus`.

Invariants (review recommendation 1):

- active project + no path          -> persistent project corpus
- covered path                      -> persistent project corpus
- uncovered path                    -> ephemeral corpus
- uncovered path + --persistent     -> error
- no project + path                 -> ephemeral corpus
- freshness=check/auto/skip         -> preserved into CorpusAccess.open_project
- project runtime differs from app  -> project runtime selected
- project settings match app config -> app runtime reused
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pytest

from simgrep import execution
from simgrep.models import AppConfig, FreshnessMode, ProjectConfig


@dataclass
class _Probe:
    """Records which corpus branch ran and with which runtime/freshness."""

    branches: list[str] = field(default_factory=list)
    runtimes: list[str] = field(default_factory=list)
    freshness: list[FreshnessMode | None] = field(default_factory=list)
    ephemeral_paths: list[Path] = field(default_factory=list)


class _FakeRuntime:
    ndim = 2

    def __init__(self, tag: str) -> None:
        self.tag = tag


class _FakeFactory:
    """Two distinct runtimes so runtime selection is observable."""

    def __init__(self) -> None:
        self.app_runtime = _FakeRuntime("app")
        self.project_runtime = _FakeRuntime("project")

    def for_app(self, _config: Any) -> _FakeRuntime:
        return self.app_runtime

    def for_project(self, _project: Any) -> _FakeRuntime:
        return self.project_runtime


class _FakeReader:
    def snapshot(self) -> str:
        return "batch"

    def counts(self, name: str = "") -> None:
        return None

    def close(self) -> None:
        return None


def _patch_corpus_access(monkeypatch: pytest.MonkeyPatch, probe: _Probe) -> None:
    import simgrep.corpus as corpus_module

    @contextmanager
    def open_project(self: Any, project: Any, app_config: Any, *, freshness: Any = None) -> Iterator[_FakeReader]:
        probe.branches.append("persistent")
        probe.runtimes.append(self.runtime.tag)
        probe.freshness.append(freshness)
        yield _FakeReader()

    @contextmanager
    def open_ephemeral(self: Any, paths: Any, app_config: Any, options: Any = None) -> Iterator[_FakeReader]:
        probe.branches.append("ephemeral")
        probe.runtimes.append(self.runtime.tag)
        probe.freshness.append(None)
        probe.ephemeral_paths.append(Path(paths[0]))
        yield _FakeReader()

    monkeypatch.setattr(corpus_module.CorpusAccess, "open_project", open_project)
    monkeypatch.setattr(corpus_module.CorpusAccess, "open_ephemeral", open_ephemeral)


def _project(root: Path, *, model: str = "same-model", chunk_size: int = 128, chunk_overlap: int = 20) -> ProjectConfig:
    return ProjectConfig(
        schema_version=1,
        name="fake",
        root=root,
        indexed_paths=(root,),
        model=model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def _activate(monkeypatch: pytest.MonkeyPatch, project: ProjectConfig | None, *, covers: bool = True) -> None:
    import simgrep.project as project_module

    monkeypatch.setattr(project_module, "find_active_project", lambda start=None: project)
    monkeypatch.setattr(project_module, "project_covers_path", lambda config, path: covers)
    monkeypatch.setattr(execution, "RuntimeFactory", _FakeFactory)


def test_active_project_with_no_path_goes_persistent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _Probe()
    _patch_corpus_access(monkeypatch, probe)
    _activate(monkeypatch, _project(tmp_path / "proj"))

    from simgrep.execution import CorpusRequest, _search_scope_errors, open_resolved_corpus

    with open_resolved_corpus(CorpusRequest(path=None), AppConfig(model="same-model"), errors=_search_scope_errors()):
        pass
    assert probe.branches == ["persistent"]


def test_covered_path_goes_persistent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _Probe()
    _patch_corpus_access(monkeypatch, probe)
    project = _project(tmp_path / "proj")
    _activate(monkeypatch, project, covers=True)
    inside = project.root
    inside.mkdir(exist_ok=True)

    from simgrep.execution import CorpusRequest, _search_scope_errors, open_resolved_corpus

    with open_resolved_corpus(CorpusRequest(path=inside), AppConfig(model="same-model"), errors=_search_scope_errors()):
        pass
    assert probe.branches == ["persistent"]


def test_uncovered_path_goes_ephemeral(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _Probe()
    _patch_corpus_access(monkeypatch, probe)
    _activate(monkeypatch, _project(tmp_path / "proj"), covers=False)
    outside = tmp_path / "outside"
    outside.mkdir()

    from simgrep.execution import CorpusRequest, _search_scope_errors, open_resolved_corpus

    with open_resolved_corpus(CorpusRequest(path=outside), AppConfig(model="same-model"), errors=_search_scope_errors()):
        pass
    assert probe.branches == ["ephemeral"]
    assert probe.ephemeral_paths == [outside]


def test_uncovered_path_with_persistent_flag_is_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _Probe()
    _patch_corpus_access(monkeypatch, probe)
    _activate(monkeypatch, _project(tmp_path / "proj"), covers=False)
    outside = tmp_path / "outside"
    outside.mkdir()

    from simgrep.errors import SearchError
    from simgrep.execution import CorpusRequest, _search_scope_errors, open_resolved_corpus

    with pytest.raises(SearchError) as exc:
        with open_resolved_corpus(CorpusRequest(path=outside, persistent=True), AppConfig(model="same-model"), errors=_search_scope_errors()):
            pass
    assert "covering the requested path" in str(exc.value)
    assert probe.branches == []


def test_no_project_with_path_goes_ephemeral(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _Probe()
    _patch_corpus_access(monkeypatch, probe)
    _activate(monkeypatch, None)
    outside = tmp_path / "outside"
    outside.mkdir()

    from simgrep.execution import CorpusRequest, _search_scope_errors, open_resolved_corpus

    with open_resolved_corpus(CorpusRequest(path=outside), AppConfig(model="same-model"), errors=_search_scope_errors()):
        pass
    assert probe.branches == ["ephemeral"]
    assert probe.ephemeral_paths == [outside]


@pytest.mark.parametrize("freshness", [FreshnessMode.auto, FreshnessMode.check, FreshnessMode.skip])
def test_freshness_is_preserved_into_the_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, freshness: FreshnessMode) -> None:
    probe = _Probe()
    _patch_corpus_access(monkeypatch, probe)
    _activate(monkeypatch, _project(tmp_path / "proj"))

    from simgrep.execution import CorpusRequest, _search_scope_errors, open_resolved_corpus

    with open_resolved_corpus(
        CorpusRequest(path=None),
        AppConfig(model="same-model", freshness=FreshnessMode.auto),
        freshness=freshness,
        errors=_search_scope_errors(),
    ):
        pass
    assert probe.freshness == [freshness]


def test_diverging_project_model_selects_project_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _Probe()
    _patch_corpus_access(monkeypatch, probe)
    _activate(monkeypatch, _project(tmp_path / "proj", model="other-model", chunk_size=512, chunk_overlap=64))

    from simgrep.execution import CorpusRequest, _search_scope_errors, open_resolved_corpus

    with open_resolved_corpus(CorpusRequest(path=None), AppConfig(model="same-model"), errors=_search_scope_errors()):
        pass
    assert probe.branches == ["persistent"]
    assert probe.runtimes == ["project"]


def test_matching_project_settings_reuse_app_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _Probe()
    _patch_corpus_access(monkeypatch, probe)
    _activate(monkeypatch, _project(tmp_path / "proj", model="same-model"))

    from simgrep.execution import CorpusRequest, _search_scope_errors, open_resolved_corpus

    with open_resolved_corpus(CorpusRequest(path=None), AppConfig(model="same-model"), errors=_search_scope_errors()):
        pass
    assert probe.branches == ["persistent"]
    assert probe.runtimes == ["app"]
