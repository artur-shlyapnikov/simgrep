"""Unit tests for DebtEngine over FakeRuntime corpora and real tmp git repos."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from simgrep.corpus import CorpusAccess
from simgrep.debt_engine import DebtEngine
from simgrep.errors import DebtError
from simgrep.indexing import IndexEngine, IndexOptions
from simgrep.models import SCHEMA_VERSION, AppConfig, DebtOptions, DebtReport, FreshnessMode, ProjectConfig
from tests.conftest import FakeRuntime

# FakeEmbedder vectors depend ONLY on text length, so equal-length chunks get
# identical vectors. A near-1 threshold keeps exact-length themes together while
# any other length (cosine << 1) stays apart.
_THRESHOLD = 0.999_99


def _git(repo: Path, *args: str, env: dict[str, str] | None = None) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)


def _padded(text: str, length: int) -> str:
    assert len(text) <= length, f"{len(text)} > {length}"
    return text + "#" * (length - len(text))


def _init_repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, capture_output=True)
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "tester")
    return tmp_path


def _commit(repo: Path, message: str, *, when: str) -> None:
    env = dict(os.environ)
    env["GIT_AUTHOR_DATE"] = when
    env["GIT_COMMITTER_DATE"] = when
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message, env=env)


@pytest.fixture
def themed_repo(tmp_path: Path) -> Path:
    """Two same-length retry files, two same-length auth files, one unique-length singleton."""
    repo = _init_repo(tmp_path)
    retry_a = repo / "src" / "retry_a.py"
    retry_b = repo / "src" / "retry_b.py"
    auth_a = repo / "src" / "auth_a.py"
    solo = repo / "src" / "solo.py"
    retry_a.write_text(_padded("# TODO retry backoff handling\n# FIXME jitter wait\n", 60))
    retry_b.write_text(_padded("# TODO retry backoff handling again ok\n", 60))
    auth_a.write_text(_padded("# FIXME auth token refresh flow\n", 80))
    solo.write_text(_padded("# HACK quick shim around upstream bug\n", 101))
    _commit(repo, "old retry", when="2020-01-01T00:00:00 +0000")
    retry_b.write_text(_padded("# TODO retry backoff handling v2 now\n", 60))
    _commit(repo, "new retry", when="2025-06-01T00:00:00 +0000")
    return repo


def _app_config() -> AppConfig:
    return AppConfig(model="fake")


def _options(**overrides: object) -> DebtOptions:
    values: dict[str, object] = {"threshold": _THRESHOLD}
    values.update(overrides)
    return DebtOptions(**values)  # type: ignore[arg-type]


def _project(root: Path) -> ProjectConfig:
    return ProjectConfig(SCHEMA_VERSION, "t", root, (root,), "fake", 128, 20)


# ---------------------------------------------------------------- engine core


def test_persistent_and_ephemeral_parity(themed_repo: Path, fake_runtime: FakeRuntime) -> None:
    # All marker files share one length so both corpora collapse to identical clusters.
    for name in ("retry_a.py", "retry_b.py", "auth_a.py", "solo.py"):
        path = themed_repo / "src" / name
        path.write_text(_padded(f"# TODO parity theme {name}\n", 70))
    _commit(themed_repo, "parity", when="2024-01-01T00:00:00 +0000")

    IndexEngine(fake_runtime).index_project(_project(themed_repo), _app_config(), IndexOptions(rebuild=True))
    options = _options()

    persistent = DebtEngine(fake_runtime).debt_project(_project(themed_repo), _app_config(), options, freshness=FreshnessMode.skip)
    ephemeral = DebtEngine(fake_runtime).debt_path(themed_repo, _app_config(), options)

    assert persistent == ephemeral
    assert persistent.markers_found == 4


def test_themes_ages_and_scattered_from_tmp_git_repo(themed_repo: Path, fake_runtime: FakeRuntime) -> None:
    old_epoch = 1577836800  # 2020-01-01T00:00:00Z
    report = DebtEngine(fake_runtime).debt_path(themed_repo, _app_config(), _options())

    assert {theme.size for theme in report.themes} == {2}
    retry_theme = next(theme for theme in report.themes if any(match.marker == "FIXME" or "backoff" in match.snippet for match in theme.matches))
    assert retry_theme.oldest_epoch == old_epoch  # min(2020 commit, 2025 commit)
    assert report.scattered == 2  # lone auth file + unique-length HACK file stay below min_size
    assert report.markers_found == 5
    assert report.chunks_scanned == 4
    assert report.truncated is False
    assert report.passed is None

    # injecting now_epoch stores nothing different: ages stay raw epochs
    gated_direct = DebtEngine(fake_runtime).debt_path(themed_repo, _app_config(), _options(max_age_days=10_000.0))
    gated_injected = _run_with_now_epoch(DebtEngine(fake_runtime), themed_repo, _app_config(), _options(max_age_days=10_000.0))
    assert gated_direct == gated_injected
    assert gated_direct.passed is True


def _run_with_now_epoch(engine: DebtEngine, root: Path, app_config: AppConfig, options: DebtOptions) -> DebtReport:
    with CorpusAccess(engine.runtime).open_ephemeral([root], app_config) as corpus:
        return engine.run_batch(corpus.snapshot(), root, options, now_epoch=1_900_000_000.0)


def test_no_markers_yields_clean_report(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "clean.py").write_text("def clean():\n    return 1\n")

    report = DebtEngine(fake_runtime).debt_path(plain, _app_config(), _options())

    assert report.themes == ()
    assert report.scattered == 0
    assert report.markers_found == 0
    assert report.chunks_scanned == 1
    assert report.truncated is False
    assert report.passed is None


def test_no_markers_with_gate_passes(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "clean.py").write_text("def clean():\n    return 1\n")

    report = DebtEngine(fake_runtime).debt_path(plain, _app_config(), _options(max_age_days=30.0))
    assert report.passed is True


def test_max_chunks_guard_raises_with_scope_hint(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    plain = tmp_path / "many"
    plain.mkdir()
    (plain / "a.py").write_text("# TODO a\n")
    (plain / "b.py").write_text("# TODO b\n")

    with pytest.raises(DebtError, match="Corpus too large") as exc_info:
        DebtEngine(fake_runtime).debt_path(plain, _app_config(), _options(max_chunks=1))
    assert exc_info.value.hint == "Narrow the scope (e.g. a subdirectory)."


def test_gate_without_git_raises_no_ages_error(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    plain = tmp_path / "no-git"
    plain.mkdir()
    (plain / "a.py").write_text("# TODO something\n")
    (plain / "b.py").write_text("# TODO something\n")

    with pytest.raises(DebtError, match="no git ages available") as exc_info:
        DebtEngine(fake_runtime).debt_path(plain, _app_config(), _options(max_age_days=90.0))
    assert "git repository" in (exc_info.value.hint or "")


def test_non_git_dir_without_gate_reports_null_ages(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    plain = tmp_path / "no-git"
    plain.mkdir()
    (plain / "a.py").write_text(_padded("# TODO alpha one\n", 40))
    (plain / "b.py").write_text(_padded("# TODO alpha two\n", 40))

    report = DebtEngine(fake_runtime).debt_path(plain, _app_config(), _options(min_size=2))

    assert len(report.themes) == 1
    assert report.themes[0].oldest_epoch is None
    assert report.passed is None


def test_gate_fails_on_old_theme_in_git_repo(themed_repo: Path, fake_runtime: FakeRuntime) -> None:
    report = DebtEngine(fake_runtime).debt_path(themed_repo, _app_config(), _options(max_age_days=90.0))
    assert report.passed is False


def test_leading_markerless_chunk_does_not_crash(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    # Regression: a markerless chunk sorted before any marked chunk used to
    # shift the global row indices and crash with KeyError in build_report.
    plain = tmp_path / "lead"
    plain.mkdir()
    (plain / "aaa_clean.py").write_text("def clean():\n    return 1\n")
    (plain / "b_marked.py").write_text(_padded("# TODO retry backoff here\n", 50))
    (plain / "c_marked.py").write_text(_padded("# TODO retry backoff again\n", 50))

    report = DebtEngine(fake_runtime).debt_path(plain, _app_config(), _options())

    assert len(report.themes) == 1
    assert report.themes[0].size == 2
    assert {match.file_path for match in report.themes[0].matches} == {"b_marked.py", "c_marked.py"}
    assert report.markers_found == 2
    assert report.scattered == 0


def test_multi_marker_chunk_keeps_all_occurrences(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    # Regression: global-vs-kept row indexing corrupted matches whenever one
    # chunk carried several markers (duplicates from the wrong chunk).
    plain = tmp_path / "multi"
    plain.mkdir()
    (plain / "a.py").write_text(_padded("# TODO alpha one\n# FIXME beta two\n", 44))
    (plain / "b.py").write_text(_padded("# TODO gamma three\n", 44))

    report = DebtEngine(fake_runtime).debt_path(plain, _app_config(), _options())

    assert len(report.themes) == 1
    theme = report.themes[0]
    assert theme.size == 2
    assert {(match.marker, match.snippet) for match in theme.matches} == {
        ("TODO", "alpha one"),
        ("FIXME", "beta two"),
        ("TODO", "gamma three"),
    }
    assert report.markers_found == 3
    assert report.scattered == 0
