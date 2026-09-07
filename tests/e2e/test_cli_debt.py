"""E2E coverage for `simgrep debt` (semantic debt-marker radar with themes and ages)."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
from collections.abc import Sequence
from typing import Any, cast

import pytest
from typer.testing import CliRunner, Result

from simgrep.main import app

try:
    runner = CliRunner(mix_stderr=False)  # type: ignore[call-arg]
except TypeError:
    runner = CliRunner()


def run_simgrep_command(args: Sequence[str]) -> Result:
    """In-process CLI invocation with a wide terminal for stable wrapping."""
    return runner.invoke(app, list(args), env={"COLUMNS": "200"})


# FakeEmbedder vectors depend ONLY on chunk-text length: equal-length files get
# identical vectors (cosine 1.0) while any two distinct lengths stay below
# ~0.999 cosine, so a near-1 threshold separates the themes deterministically.
_THRESHOLD = "0.999"

_OLD_WHEN = "2020-01-01T00:00:00 +0000"
_NEW_WHEN = "2025-06-01T00:00:00 +0000"


def _padded(text: str, length: int) -> str:
    assert len(text) <= length, f"{len(text)} > {length}"
    return text + "#" * (length - len(text))


def _git(repo: pathlib.Path, *args: str, env: dict[str, str] | None = None) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)


def _commit(repo: pathlib.Path, message: str, *, when: str) -> None:
    env = dict(os.environ)
    env["GIT_AUTHOR_DATE"] = when
    env["GIT_COMMITTER_DATE"] = when
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message, env=env)


@pytest.fixture
def debt_repo(tmp_path: pathlib.Path) -> pathlib.Path:
    """Git repo with two same-length marker pairs (retry/auth) and one singleton."""
    repo = tmp_path / "proj"
    (repo / "src").mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "tester")
    (repo / "src" / "retry_a.py").write_text(_padded("# TODO retry backoff handling\n# FIXME jitter wait\n", 60), encoding="utf-8")
    (repo / "src" / "retry_b.py").write_text(_padded("# TODO retry backoff handling again ok\n", 60), encoding="utf-8")
    (repo / "src" / "auth_a.py").write_text(_padded("# FIXME auth token refresh flow\n", 80), encoding="utf-8")
    (repo / "src" / "auth_b.py").write_text(_padded("# TODO auth token expiry check\n", 80), encoding="utf-8")
    (repo / "src" / "solo.py").write_text(_padded("# HACK quick shim around upstream bug\n", 101), encoding="utf-8")
    _commit(repo, "old work", when=_OLD_WHEN)
    (repo / "src" / "retry_b.py").write_text(_padded("# TODO retry backoff handling v2 now\n", 60), encoding="utf-8")
    _commit(repo, "new retry", when=_NEW_WHEN)
    return repo


def _run(repo: pathlib.Path, *extra: str) -> Result:
    return run_simgrep_command(["debt", str(repo), "--threshold", _THRESHOLD, "--format", "json", *extra])


def _payload(repo: pathlib.Path, *extra: str) -> dict[str, Any]:
    result = _run(repo, *extra)
    assert result.exit_code == 0, (result.stdout, result.stderr)
    return cast(dict[str, Any], json.loads(result.stdout))


def test_rich_lists_theme_labels_and_marker_paths(debt_repo: pathlib.Path) -> None:
    result = run_simgrep_command(["debt", str(debt_repo), "--threshold", _THRESHOLD])
    assert result.exit_code == 0, (result.stdout, result.stderr)
    assert "Debt Themes (2)" in result.stdout
    assert "5 scattered" in result.stdout or "scattered" in result.stdout
    assert "src/retry_a.py" in result.stdout
    assert "src/auth_a.py" in result.stdout
    assert "FIXME" in result.stdout


def test_json_payload_matches_pinned_shape(debt_repo: pathlib.Path) -> None:
    payload = _payload(debt_repo)
    assert set(payload) == {
        "themes",
        "scattered",
        "markers_found",
        "chunks_scanned",
        "truncated",
        "threshold",
        "max_age_days",
        "passed",
    }
    assert len(payload["themes"]) == 2
    theme = payload["themes"][0]
    assert set(theme) == {"label", "size", "oldest_epoch", "matches"}
    assert theme["size"] == 2
    match = theme["matches"][0]
    assert set(match) == {"file_path", "line_start", "marker", "snippet"}
    labels = {theme["label"] for theme in payload["themes"]}
    assert any("backoff" in label or "token" in label or "auth" in label for label in labels), labels
    # retry theme is the oldest (retry_a.py untouched since 2020); ranks first.
    oldest = [theme["oldest_epoch"] for theme in payload["themes"]]
    assert all(epoch is not None for epoch in oldest)
    assert min(oldest) == 1577836800  # 2020-01-01T00:00:00Z
    assert oldest[0] == min(oldest)  # oldest-dated theme ranks first
    assert payload["markers_found"] == 6
    assert payload["passed"] is None  # no gate requested
    assert payload["threshold"] == float(_THRESHOLD)
    assert payload["max_age_days"] is None


def test_max_age_gate_failure_exits_one(debt_repo: pathlib.Path) -> None:
    result = run_simgrep_command(["debt", str(debt_repo), "--threshold", _THRESHOLD, "--format", "json", "--max-age", "90"])
    assert result.exit_code == 1, (result.stdout, result.stderr)
    payload = json.loads(result.stdout)
    assert payload["passed"] is False
    assert payload["max_age_days"] == 90.0


def test_max_age_gate_pass_exits_zero_when_commits_are_recent(debt_repo: pathlib.Path) -> None:
    fresh = debt_repo.parent / "fresh-repo"
    subprocess.run(["cp", "-R", str(debt_repo), str(fresh)], check=True)
    # Re-date every themed file into the present by committing touch-ups now.
    (fresh / "src" / "retry_a.py").write_text(_padded("# TODO retry backoff handling v3 ok\n", 60), encoding="utf-8")
    (fresh / "src" / "retry_b.py").write_text(_padded("# TODO retry backoff handling v4 now\n", 60), encoding="utf-8")
    (fresh / "src" / "auth_a.py").write_text(_padded("# FIXME auth token refresh flow v2\n", 80), encoding="utf-8")
    (fresh / "src" / "auth_b.py").write_text(_padded("# TODO auth token expiry check v2\n", 80), encoding="utf-8")
    _commit(fresh, "touch up themed files", when="2026-08-20T00:00:00 +0000")
    result = run_simgrep_command(["debt", str(fresh), "--threshold", _THRESHOLD, "--format", "json", "--max-age", "90"])
    assert result.exit_code == 0, (result.stdout, result.stderr)
    assert json.loads(result.stdout)["passed"] is True


def test_jsonl_emits_theme_records_then_summary_tail(debt_repo: pathlib.Path) -> None:
    result = run_simgrep_command(["debt", str(debt_repo), "--threshold", _THRESHOLD, "--format", "jsonl"])
    assert result.exit_code == 0, (result.stdout, result.stderr)
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert len(rows) == 3
    for row in rows[:-1]:
        assert set(row) == {"label", "size", "oldest_epoch", "matches"}
    summary = rows[-1]
    assert summary["kind"] == "summary"
    assert set(summary) == {
        "kind",
        "scattered",
        "markers_found",
        "chunks_scanned",
        "truncated",
        "threshold",
        "max_age_days",
        "passed",
    }


def test_no_markers_exits_zero_with_clean_message(tmp_path: pathlib.Path) -> None:
    plain = tmp_path / "plain"
    (plain / "src").mkdir(parents=True)
    (plain / "src" / "alpha.py").write_text("x = compute(a) + refine(b)\n", encoding="utf-8")
    result = run_simgrep_command(["debt", str(plain)])
    assert result.exit_code == 0, (result.stdout, result.stderr)
    assert "No debt markers found." in result.stdout


def test_non_git_directory_without_gate_reports_null_ages(tmp_path: pathlib.Path) -> None:
    plain = tmp_path / "nogit"
    (plain / "src").mkdir(parents=True)
    (plain / "src" / "retry_a.py").write_text(_padded("# TODO retry backoff handling\n", 60), encoding="utf-8")
    (plain / "src" / "retry_b.py").write_text(_padded("# FIXME retry jitter wait here ok\n", 60), encoding="utf-8")
    payload = _payload(plain)
    assert len(payload["themes"]) == 1
    assert payload["themes"][0]["oldest_epoch"] is None
    assert payload["passed"] is None


def test_top_above_bounds_exits_two(debt_repo: pathlib.Path) -> None:
    result = run_simgrep_command(["debt", str(debt_repo), "--top", "201"])
    assert result.exit_code == 2


def test_min_size_below_bounds_exits_two(debt_repo: pathlib.Path) -> None:
    result = run_simgrep_command(["debt", str(debt_repo), "--min-size", "0"])
    assert result.exit_code == 2


def test_max_members_above_bounds_exits_two(debt_repo: pathlib.Path) -> None:
    result = run_simgrep_command(["debt", str(debt_repo), "--max-members", "51"])
    assert result.exit_code == 2


def test_threshold_out_of_bounds_exits_two(debt_repo: pathlib.Path) -> None:
    assert run_simgrep_command(["debt", str(debt_repo), "--threshold", "0"]).exit_code == 2
    assert run_simgrep_command(["debt", str(debt_repo), "--threshold", "1.5"]).exit_code == 2


def test_max_age_non_positive_exits_two(debt_repo: pathlib.Path) -> None:
    assert run_simgrep_command(["debt", str(debt_repo), "--max-age", "0"]).exit_code == 2


def test_invalid_format_exits_two(debt_repo: pathlib.Path) -> None:
    result = run_simgrep_command(["debt", str(debt_repo), "--format", "bogus"])
    assert result.exit_code == 2


def test_no_target_and_no_active_project_is_typed_error(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    monkeypatch.chdir(fresh)
    monkeypatch.setenv("HOME", str(tmp_path))
    result = run_simgrep_command(["debt"])
    assert result.exit_code != 0
    assert "No active project found." in (result.stderr or "")


def test_help_renders() -> None:
    result = run_simgrep_command(["debt", "--help"])
    assert result.exit_code == 0
    assert "--threshold" in result.stdout
    assert "--max-age" in result.stdout
    assert "--max-members" in result.stdout


def test_deterministic_jsonl_across_runs(debt_repo: pathlib.Path) -> None:
    args = ["debt", str(debt_repo), "--threshold", _THRESHOLD, "--format", "jsonl"]
    first = run_simgrep_command(args).stdout
    second = run_simgrep_command(args).stdout
    assert first == second
