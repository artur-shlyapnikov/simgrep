"""Measurement utilities, report dataclasses, and JSON serialization."""

from __future__ import annotations

import gc
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class BenchmarkCaseResult:
    """Result of a single benchmark case."""

    name: str
    suite: str  # "ci", "real", "stress"
    corpus: "BenchmarkCorpusInfo"
    config: "BenchmarkConfig"
    iterations: int
    warmups: int
    samples_seconds: List[float]
    median_seconds: float
    p95_seconds: float
    min_seconds: float
    max_seconds: float
    outcome: "BenchmarkOutcome"
    status: "BenchmarkStatus"
    metrics: Optional["BenchmarkMetrics"] = None
    phases: Optional["BenchmarkPhases"] = None


@dataclass
class BenchmarkCorpusInfo:
    """Corpus information for a benchmark case."""

    profile: str
    files: int
    bytes: int
    expected_indexable_files: int


@dataclass
class BenchmarkConfig:
    """Configuration used for a benchmark case."""

    workers: int
    batch_size: int
    freshness: Optional[str] = None
    change_detection: Optional[str] = None
    model_mode: str = "fake"  # "fake" or "real"


@dataclass
class BenchmarkMetrics:
    """Computed metrics for a benchmark case."""

    files_per_second: Optional[float] = None
    chunks_per_second: Optional[float] = None
    queries_per_second: Optional[float] = None


@dataclass
class BenchmarkPhases:
    """Phase timings for a benchmark case."""

    plan_seconds: Optional[float] = None
    scan_seconds: Optional[float] = None
    extract_chunk_seconds: Optional[float] = None
    embedding_seconds: Optional[float] = None
    store_seconds: Optional[float] = None
    index_save_seconds: Optional[float] = None
    index_load_seconds: Optional[float] = None
    query_embedding_seconds: Optional[float] = None
    vector_search_seconds: Optional[float] = None
    ranking_seconds: Optional[float] = None
    freshness_seconds: Optional[float] = None
    render_seconds: Optional[float] = None


@dataclass
class BenchmarkOutcome:
    """Outcome information for a benchmark case."""

    files_seen: Optional[int] = None
    files_indexed: Optional[int] = None
    files_skipped_unchanged: Optional[int] = None
    files_pruned_deleted: Optional[int] = None
    chunks_indexed: Optional[int] = None
    results_count: Optional[int] = None
    errors: int = 0


@dataclass
class BenchmarkStatus:
    """Status of a benchmark case."""

    passed: Optional[bool] = None
    skipped: bool = False
    skip_reason: Optional[str] = None


@dataclass
class EnvironmentInfo:
    """Environment information."""

    python_version: str = ""
    platform: str = ""
    machine: str = ""
    cpu_count: int = 0
    os: str = ""
    uv_version: Optional[str] = None


@dataclass
class SimgrepInfo:
    """Simgrep version and configuration info."""

    version: Optional[str] = None
    default_model: str = ""


@dataclass
class SpeedBenchmarkReport:
    """Complete benchmark report."""

    schema_version: int = 1
    created_at: str = ""
    git_commit: Optional[str] = None
    environment: EnvironmentInfo = field(default_factory=EnvironmentInfo)
    simgrep: SimgrepInfo = field(default_factory=SimgrepInfo)
    benchmarks: List[BenchmarkCaseResult] = field(default_factory=list)


@dataclass
class BaselineCase:
    """Baseline case definition."""

    median_seconds: float
    p95_seconds: float
    max_regression_ratio: float = 1.25
    hard_max_seconds: Optional[float] = None


@dataclass
class SpeedBaseline:
    """Baseline for speed benchmarks."""

    schema_version: int = 1
    cases: Dict[str, BaselineCase] = field(default_factory=dict)


def get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def get_uv_version() -> Optional[str]:
    """Get uv version if available."""
    try:
        import subprocess

        result = subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip().split()[0] if result.returncode == 0 else None
    except Exception:
        return None


def get_environment_info() -> EnvironmentInfo:
    """Collect environment information."""
    return EnvironmentInfo(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        machine=platform.machine(),
        cpu_count=os.cpu_count() or 0,
        os=platform.system(),
        uv_version=get_uv_version(),
    )


def get_simgrep_info() -> SimgrepInfo:
    """Get simgrep version and configuration."""
    version = None
    try:
        from importlib.metadata import version as get_version

        version = get_version("simgrep")
    except Exception:
        pass

    return SimgrepInfo(
        version=version,
        default_model="ibm-granite/granite-embedding-30m-english",
    )


def calculate_stats(samples: List[float]) -> Dict[str, float]:
    """Calculate statistics from samples."""
    if not samples:
        return {"median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}

    sorted_samples = sorted(samples)
    n = len(sorted_samples)

    median = statistics.median(sorted_samples)
    p95_idx = int(n * 0.95)
    p95 = sorted_samples[min(p95_idx, n - 1)]

    return {
        "median": median,
        "p95": p95,
        "min": min(sorted_samples),
        "max": max(sorted_samples),
    }


def run_with_gc_control(fn: Callable) -> Any:
    """Run function with GC disabled for more stable timing."""
    gc.disable()
    try:
        return fn()
    finally:
        gc.enable()


def measure_case(
    name: str,
    suite: str,
    corpus: BenchmarkCorpusInfo,
    config: BenchmarkConfig,
    fn: Callable[[], BenchmarkOutcome],
    iterations: int = 10,
    warmups: int = 2,
) -> BenchmarkCaseResult:
    """
    Measure a benchmark case with warmups and iterations.

    Args:
        name: Case name
        suite: Suite name (ci, real, stress)
        corpus: Corpus information
        config: Configuration
        fn: Function to benchmark (returns BenchmarkOutcome)
        iterations: Number of measured iterations
        warmups: Number of warmup iterations

    Returns:
        BenchmarkCaseResult with timing and outcome data
    """
    samples: List[float] = []
    errors = 0

    # Run warmups
    for _ in range(warmups):
        try:
            outcome = run_with_gc_control(fn)
        except Exception:
            # On warmup failure, continue but track
            errors += 1
            break

    # Run measured iterations
    for _ in range(iterations):
        try:
            t0 = time.perf_counter()
            outcome = run_with_gc_control(fn)
            elapsed = time.perf_counter() - t0
            samples.append(elapsed)
        except Exception:
            # Record failure but continue
            errors += 1
            outcome = BenchmarkOutcome(errors=errors)
    # Preserve both raised-exception count and errors reported by fn itself.
    outcome.errors = max(outcome.errors, errors)
    stats = calculate_stats(samples)

    return BenchmarkCaseResult(
        name=name,
        suite=suite,
        corpus=corpus,
        config=config,
        iterations=iterations,
        warmups=warmups,
        samples_seconds=samples,
        median_seconds=stats["median"],
        p95_seconds=stats["p95"],
        min_seconds=stats["min"],
        max_seconds=stats["max"],
        outcome=outcome,
        status=BenchmarkStatus(passed=outcome.errors == 0),
    )


def measure_case_with_phases(
    name: str,
    suite: str,
    corpus: BenchmarkCorpusInfo,
    config: BenchmarkConfig,
    fn: Callable[[], tuple[BenchmarkOutcome, Optional[dict]]],
    iterations: int = 10,
    warmups: int = 2,
) -> BenchmarkCaseResult:
    """
    Measure a benchmark case that returns both outcome and phase timings.

    Args:
        fn: Function returning (BenchmarkOutcome, phase_timings_dict)

    Returns:
        BenchmarkCaseResult with phase timings
    """
    samples: List[float] = []
    phases_accum: Dict[str, List[float]] = {}
    outcome = BenchmarkOutcome()
    errors = 0

    # Run warmups
    for _ in range(warmups):
        try:
            result = run_with_gc_control(fn)
            if isinstance(result, tuple):
                outcome, phases = result
            else:
                outcome = result
                phases = None
        except Exception:
            errors += 1
            phases = None
            break

    # Run measured iterations
    for _ in range(iterations):
        try:
            t0 = time.perf_counter()
            result = run_with_gc_control(fn)
            elapsed = time.perf_counter() - t0
            samples.append(elapsed)

            if isinstance(result, tuple):
                outcome, phases = result
                if phases:
                    for k, v in phases.items():
                        if v is not None:
                            phases_accum.setdefault(k, []).append(v)
            else:
                outcome = result
                phases = None
        except Exception:
            errors += 1

    stats = calculate_stats(samples)

    # Build phase timings
    phases = None
    if phases_accum:
        phases = BenchmarkPhases(
            plan_seconds=statistics.median(phases_accum.get("plan_seconds", [0.0])),
            scan_seconds=statistics.median(phases_accum.get("scan_seconds", [0.0])),
            extract_chunk_seconds=statistics.median(phases_accum.get("extract_chunk_seconds", [0.0])),
            embedding_seconds=statistics.median(phases_accum.get("embedding_seconds", [0.0])),
            store_seconds=statistics.median(phases_accum.get("store_seconds", [0.0])),
            index_save_seconds=statistics.median(phases_accum.get("index_save_seconds", [0.0])),
            index_load_seconds=statistics.median(phases_accum.get("index_load_seconds", [0.0])),
            query_embedding_seconds=statistics.median(phases_accum.get("query_embedding_seconds", [0.0])),
            vector_search_seconds=statistics.median(phases_accum.get("vector_search_seconds", [0.0])),
            ranking_seconds=statistics.median(phases_accum.get("ranking_seconds", [0.0])),
            freshness_seconds=statistics.median(phases_accum.get("freshness_seconds", [0.0])),
        )
    # Preserve both raised-exception count and errors reported by fn itself.
    outcome.errors = max(outcome.errors, errors)

    # Compute metrics
    metrics = None
    if outcome.chunks_indexed and stats["median"] > 0:
        metrics = BenchmarkMetrics(
            chunks_per_second=outcome.chunks_indexed / stats["median"],
        )
    if outcome.files_indexed and stats["median"] > 0:
        metrics = metrics or BenchmarkMetrics()
        metrics.files_per_second = outcome.files_indexed / stats["median"]

    return BenchmarkCaseResult(
        name=name,
        suite=suite,
        corpus=corpus,
        config=config,
        iterations=iterations,
        warmups=warmups,
        samples_seconds=samples,
        median_seconds=stats["median"],
        p95_seconds=stats["p95"],
        min_seconds=stats["min"],
        max_seconds=stats["max"],
        metrics=metrics,
        phases=phases,
        outcome=outcome,
        status=BenchmarkStatus(passed=outcome.errors == 0),
    )


def create_skipped_result(
    name: str,
    suite: str,
    corpus: BenchmarkCorpusInfo,
    config: BenchmarkConfig,
    reason: str,
) -> BenchmarkCaseResult:
    """Create a skipped benchmark result."""
    return BenchmarkCaseResult(
        name=name,
        suite=suite,
        corpus=corpus,
        config=config,
        iterations=0,
        warmups=0,
        samples_seconds=[],
        median_seconds=0.0,
        p95_seconds=0.0,
        min_seconds=0.0,
        max_seconds=0.0,
        outcome=BenchmarkOutcome(),
        status=BenchmarkStatus(skipped=True, skip_reason=reason),
    )


def create_report(cases: List[BenchmarkCaseResult]) -> SpeedBenchmarkReport:
    """Create a complete benchmark report."""
    now = datetime.now(timezone.utc).isoformat()

    return SpeedBenchmarkReport(
        schema_version=1,
        created_at=now,
        git_commit=get_git_commit(),
        environment=get_environment_info(),
        simgrep=get_simgrep_info(),
        benchmarks=cases,
    )


def report_to_dict(report: SpeedBenchmarkReport) -> Dict[str, Any]:
    """Convert report to dictionary for JSON serialization."""
    result: Dict[str, Any] = {
        "schema_version": report.schema_version,
        "created_at": report.created_at,
        "git_commit": report.git_commit,
        "environment": {
            "python_version": report.environment.python_version,
            "platform": report.environment.platform,
            "machine": report.environment.machine,
            "cpu_count": report.environment.cpu_count,
            "os": report.environment.os,
            "uv_version": report.environment.uv_version,
        },
        "simgrep": {
            "version": report.simgrep.version,
            "default_model": report.simgrep.default_model,
        },
        "benchmarks": [],
    }

    for case in report.benchmarks:
        case_dict = {
            "name": case.name,
            "suite": case.suite,
            "corpus": {
                "profile": case.corpus.profile,
                "files": case.corpus.files,
                "bytes": case.corpus.bytes,
                "expected_indexable_files": case.corpus.expected_indexable_files,
            },
            "config": {
                "workers": case.config.workers,
                "batch_size": case.config.batch_size,
                "freshness": case.config.freshness,
                "change_detection": case.config.change_detection,
                "model_mode": case.config.model_mode,
            },
            "iterations": case.iterations,
            "warmups": case.warmups,
            "samples_seconds": case.samples_seconds,
            "median_seconds": case.median_seconds,
            "p95_seconds": case.p95_seconds,
            "min_seconds": case.min_seconds,
            "max_seconds": case.max_seconds,
            "outcome": {
                "files_seen": case.outcome.files_seen,
                "files_indexed": case.outcome.files_indexed,
                "files_skipped_unchanged": case.outcome.files_skipped_unchanged,
                "files_pruned_deleted": case.outcome.files_pruned_deleted,
                "chunks_indexed": case.outcome.chunks_indexed,
                "results_count": case.outcome.results_count,
                "errors": case.outcome.errors,
            },
            "status": {
                "passed": case.status.passed,
                "skipped": case.status.skipped,
                "skip_reason": case.status.skip_reason,
            },
        }

        if case.metrics:
            case_dict["metrics"] = {
                "files_per_second": case.metrics.files_per_second,
                "chunks_per_second": case.metrics.chunks_per_second,
                "queries_per_second": case.metrics.queries_per_second,
            }

        if case.phases:
            case_dict["phases"] = {
                "plan_seconds": case.phases.plan_seconds,
                "scan_seconds": case.phases.scan_seconds,
                "extract_chunk_seconds": case.phases.extract_chunk_seconds,
                "embedding_seconds": case.phases.embedding_seconds,
                "store_seconds": case.phases.store_seconds,
                "index_save_seconds": case.phases.index_save_seconds,
                "index_load_seconds": case.phases.index_load_seconds,
                "query_embedding_seconds": case.phases.query_embedding_seconds,
                "vector_search_seconds": case.phases.vector_search_seconds,
                "ranking_seconds": case.phases.ranking_seconds,
                "freshness_seconds": case.phases.freshness_seconds,
                "render_seconds": case.phases.render_seconds,
            }

        result["benchmarks"].append(case_dict)

    return result


def load_baseline(path: Path) -> SpeedBaseline:
    """Load baseline from JSON file."""
    if not path.exists():
        return SpeedBaseline()

    with open(path) as f:
        data = json.load(f)

    cases = {}
    for name, case_data in data.get("cases", {}).items():
        cases[name] = BaselineCase(
            median_seconds=case_data["median_seconds"],
            p95_seconds=case_data["p95_seconds"],
            max_regression_ratio=case_data.get("max_regression_ratio", 1.25),
            hard_max_seconds=case_data.get("hard_max_seconds"),
        )

    return SpeedBaseline(schema_version=data.get("schema_version", 1), cases=cases)


def compare_to_baseline(
    result: BenchmarkCaseResult,
    baseline: SpeedBaseline,
) -> tuple[bool, List[str]]:
    """
    Compare a result to baseline and return (passed, reasons).

    Returns (True, []) if passed or (False, [reasons]) if failed.
    """
    if result.status.skipped:
        return True, []

    baseline_case = baseline.cases.get(result.name)
    if not baseline_case:
        return False, [f"No baseline found for '{result.name}'"]

    reasons = []

    # Check median regression
    median_limit = baseline_case.median_seconds * baseline_case.max_regression_ratio
    if result.median_seconds > median_limit:
        reasons.append(
            f"Median regression: {result.median_seconds:.3f}s > {median_limit:.3f}s "
            f"(baseline: {baseline_case.median_seconds:.3f}s, ratio: {result.median_seconds / baseline_case.median_seconds:.2f})"
        )

    # Check p95 regression
    p95_limit = baseline_case.p95_seconds * baseline_case.max_regression_ratio
    if result.p95_seconds > p95_limit:
        reasons.append(f"P95 regression: {result.p95_seconds:.3f}s > {p95_limit:.3f}s " f"(baseline: {baseline_case.p95_seconds:.3f}s)")

    # Check hard max if defined
    if baseline_case.hard_max_seconds and result.max_seconds > baseline_case.hard_max_seconds:
        reasons.append(f"Hard max exceeded: {result.max_seconds:.3f}s > {baseline_case.hard_max_seconds:.3f}s")

    return len(reasons) == 0, reasons


def format_markdown_table(reports: List[SpeedBenchmarkReport]) -> str:
    """Format benchmark results as markdown table."""
    lines = [
        "# Speed Benchmark Results",
        "",
        "| Case | Suite | Median | P95 | Files/s | Chunks/s | Status |",
        "|------|-------|--------|-----|---------|----------|--------|",
    ]

    for report in reports:
        for case in report.benchmarks:
            status = "SKIP" if case.status.skipped else ("PASS" if case.status.passed else "FAIL")
            files_s = f"{case.metrics.files_per_second:.1f}" if case.metrics and case.metrics.files_per_second else "-"
            chunks_s = f"{case.metrics.chunks_per_second:.1f}" if case.metrics and case.metrics.chunks_per_second else "-"

            lines.append(f"| {case.name} | {case.suite} | " f"{case.median_seconds:.3f}s | {case.p95_seconds:.3f}s | " f"{files_s} | {chunks_s} | {status} |")

    return "\n".join(lines)
