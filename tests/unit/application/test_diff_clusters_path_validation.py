from __future__ import annotations

from pathlib import Path

import pytest

from simgrep.clusters_engine import ClustersEngine
from simgrep.diff_engine import DiffEngine
from simgrep.errors import ClustersError, DiffError
from simgrep.models import AppConfig, DiffOptions
from tests.conftest import FakeRuntime


def test_diff_paths_missing_path_a_raises(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    engine = DiffEngine(fake_runtime)
    real = tmp_path / "real_a"
    real.mkdir()
    with pytest.raises(DiffError, match="Path not found"):
        engine.diff_paths(tmp_path / "nope", real, AppConfig(model="fake"), DiffOptions())


def test_diff_paths_missing_path_b_raises(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    engine = DiffEngine(fake_runtime)
    real = tmp_path / "real_b"
    real.mkdir()
    with pytest.raises(DiffError, match="Path not found"):
        engine.diff_paths(real, tmp_path / "missing", AppConfig(model="fake"), DiffOptions())


def test_clusters_path_missing_raises(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    engine = ClustersEngine(fake_runtime)
    with pytest.raises(ClustersError, match="Path not found"):
        engine.clusters_path(tmp_path / "ghost", AppConfig(model="fake"))


def test_diff_paths_existing_dirs_still_work(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    for base in (a, b):
        base.mkdir()
        (base / "same.py").write_text("x = 1\n")
    (b / "other.py").write_text("y = 2\n")
    outcome = DiffEngine(fake_runtime).diff_paths(a, b, AppConfig(model="fake"), DiffOptions())
    assert outcome.added or outcome.removed or outcome.matched  # non-empty result, did not raise
