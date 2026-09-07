"""Tests for benchmark harness failure propagation and outcome honesty."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import pytest
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parents[2]))

from benchmarks.measure import (  # noqa: E402
    BenchmarkCaseResult,
    BenchmarkConfig,
    BenchmarkCorpusInfo,
    BenchmarkOutcome,
    BenchmarkStatus,
    measure_case,
    measure_case_with_phases,
)
from benchmarks.run_speed import app  # noqa: E402


def _corpus() -> BenchmarkCorpusInfo:
    return BenchmarkCorpusInfo(profile="tiny", files=1, bytes=10, expected_indexable_files=1)


def _config() -> BenchmarkConfig:
    return BenchmarkConfig(workers=1, batch_size=64)


def _case_result(name: str, outcome: BenchmarkOutcome, samples: List[float]) -> BenchmarkCaseResult:
    median = sorted(samples)[len(samples) // 2] if samples else 0.0
    return BenchmarkCaseResult(
        name=name,
        suite="ci",
        corpus=_corpus(),
        config=_config(),
        iterations=len(samples),
        warmups=0,
        samples_seconds=samples,
        median_seconds=median,
        p95_seconds=max(samples) if samples else 0.0,
        min_seconds=min(samples) if samples else 0.0,
        max_seconds=max(samples) if samples else 0.0,
        outcome=outcome,
        status=BenchmarkStatus(passed=outcome.errors == 0),
    )


def test_measure_case_fails_when_every_iteration_raises() -> None:
    def fn() -> BenchmarkOutcome:
        raise RuntimeError("boom")

    result = measure_case(
        name="case.all.fail",
        suite="ci",
        corpus=_corpus(),
        config=_config(),
        fn=fn,
        iterations=3,
        warmups=0,
    )

    assert result.status.passed is False
    assert result.samples_seconds == []
    assert result.outcome.errors == 3


def test_measure_case_fails_when_any_iteration_raises() -> None:
    calls = {"n": 0}

    def fn() -> BenchmarkOutcome:
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("boom")
        return BenchmarkOutcome()

    result = measure_case(
        name="case.some.fail",
        suite="ci",
        corpus=_corpus(),
        config=_config(),
        fn=fn,
        iterations=3,
        warmups=0,
    )

    assert result.status.passed is False
    assert result.outcome.errors > 0
    assert len(result.samples_seconds) == 2


def test_measure_case_passes_when_clean() -> None:
    def fn() -> BenchmarkOutcome:
        return BenchmarkOutcome()

    result = measure_case(
        name="case.clean",
        suite="ci",
        corpus=_corpus(),
        config=_config(),
        fn=fn,
        iterations=3,
        warmups=0,
    )

    assert result.status.passed is True


def test_measure_case_with_phases_fails_on_errors() -> None:
    def fn() -> tuple[BenchmarkOutcome, None]:
        return (BenchmarkOutcome(errors=1), None)

    result = measure_case_with_phases(
        name="case.phases.err",
        suite="ci",
        corpus=_corpus(),
        config=_config(),
        fn=fn,
        iterations=3,
        warmups=0,
    )

    assert result.status.passed is False


def _write_baseline(path: Path, case_names: List[str]) -> None:
    cases = {
        name: {
            "median_seconds": 1.0,
            "p95_seconds": 2.0,
            "max_regression_ratio": 100.0,
            "hard_max_seconds": None,
        }
        for name in case_names
    }
    path.write_text(json.dumps({"schema_version": 1, "cases": cases}))


def _patch_suites(monkeypatch: pytest.MonkeyPatch, results: List[BenchmarkCaseResult]) -> None:
    import benchmarks.run_speed as rs

    def fake_ci(workspace: Path, iterations: int, warmups: int) -> List[BenchmarkCaseResult]:
        del workspace, iterations, warmups
        return results

    monkeypatch.setattr(rs, "run_ci_suite", fake_ci)
    monkeypatch.setattr(rs, "setup_temp_workspace", lambda: Path("/tmp/simgrep-bench-test"))


def test_main_exits_nonzero_when_case_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clean = _case_result("case.clean.ok", BenchmarkOutcome(), [0.5])
    errored = _case_result("case.errors.bad", BenchmarkOutcome(errors=3), [0.6])
    _patch_suites(monkeypatch, [clean, errored])

    baseline = tmp_path / "baseline.json"
    output = tmp_path / "report.json"
    _write_baseline(baseline, ["case.clean.ok", "case.errors.bad"])

    runner = CliRunner()
    result = runner.invoke(app, ["--suite", "ci", "--output", str(output), "--compare", str(baseline)])

    assert result.exit_code == 1


def test_update_baseline_skips_error_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clean = _case_result("case.clean.ok", BenchmarkOutcome(), [0.5])
    errored = _case_result("case.errors.bad", BenchmarkOutcome(errors=3), [0.6])
    _patch_suites(monkeypatch, [clean, errored])

    baseline = tmp_path / "baseline.json"
    output = tmp_path / "report.json"
    _write_baseline(baseline, ["case.clean.ok", "case.errors.bad"])

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--suite",
            "ci",
            "--output",
            str(output),
            "--compare",
            str(baseline),
            "--update-baseline",
        ],
    )

    assert "Baseline updated" in result.output

    data = json.loads(baseline.read_text())
    assert "case.clean.ok" in data["cases"]
    assert "case.errors.bad" not in data["cases"]
