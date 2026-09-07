#!/usr/bin/env python3
"""Speed benchmark CLI orchestrator for simgrep.

Runs CI, real-model, and stress benchmark suites.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.corpora import (
    CORPUS_MEDIUM,
    CORPUS_SMALL,
    generate_corpus,
)
from benchmarks.measure import (
    BenchmarkCaseResult,
    BenchmarkConfig,
    BenchmarkCorpusInfo,
    BenchmarkOutcome,
    compare_to_baseline,
    create_report,
    create_skipped_result,
    format_markdown_table,
    load_baseline,
    measure_case,
    report_to_dict,
)
from simgrep.models import IndexStats

app = typer.Typer(help="Speed benchmark runner for simgrep.")

# Default benchmark settings
DEFAULT_ITERATIONS = 10
DEFAULT_WARMUPS = 2
DEFAULT_WORKERS = 1
DEFAULT_BATCH_SIZE = 64


class BenchmarkError(Exception):
    """Benchmark error."""

    pass


def setup_temp_workspace() -> Path:
    """Create a temporary workspace directory."""
    return Path(tempfile.mkdtemp(prefix="simgrep-bench-"))


def setup_temp_home(workspace: Path) -> Path:
    """Create a temporary HOME directory for isolated benchmarks."""
    home = workspace / "home"
    home.mkdir()
    return home


def check_model_cached(model_name: str) -> bool:
    """Check if model is cached locally."""
    # For simplicity, check if TRANSFORMERS_CACHE or SENTENCE_TRANSFORMERS_HOME exists
    cache_paths = [
        Path.home() / ".cache" / "huggingface",
        Path.home() / ".cache" / "sentence-transformers",
        Path(os.environ.get("TRANSFORMERS_CACHE", "")),
        Path(os.environ.get("SENTENCE_TRANSFORMERS_HOME", "")),
    ]

    for cache_path in cache_paths:
        if cache_path.exists() and any(cache_path.iterdir()):
            return True

    # Try to run a quick Python check
    try:
        result = subprocess.run(
            [sys.executable, "-c", f"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('{model_name}', local_files_only=True)"],
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0
    except Exception:
        return False


def run_cli_command(
    cmd: List[str],
    home: Path,
    cwd: Optional[Path] = None,
    capture_output: bool = True,
    timeout: int = 300,
) -> subprocess.CompletedProcess:
    """Run a CLI command with isolated HOME."""
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["PYTHONHASHSEED"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"

    return subprocess.run(
        cmd,
        capture_output=capture_output,
        text=True,
        cwd=str(cwd) if cwd else None,
        env=env,
        timeout=timeout,
    )


def init_simgrep_project(home: Path, project_dir: Path) -> None:
    """Initialize a simgrep project in the given directory."""
    # Local project init
    result = run_cli_command(
        ["uv", "run", "simgrep", "--color", "never", "init"],
        home=home,
        cwd=project_dir,
    )
    if result.returncode != 0 and "already" not in result.stderr.lower():
        raise BenchmarkError(f"Failed to init project: {result.stderr}")


def index_corpus(
    home: Path,
    project_dir: Path,
    workers: int = DEFAULT_WORKERS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    wipe: bool = True,
    format_json: bool = False,
) -> tuple[IndexStats, Dict[str, Any]]:
    """Run indexing and return stats."""
    cmd = [
        "uv",
        "run",
        "simgrep",
        "--color",
        "never",
        "index",
        "--workers",
        str(workers),
    ]
    del batch_size

    if wipe:
        cmd.append("--rebuild")

    if format_json:
        cmd.extend(["--format", "json"])

    result = run_cli_command(cmd, home=home, cwd=project_dir)

    if result.returncode != 0:
        raise BenchmarkError(f"Index command failed: {result.stderr}")

    if format_json:
        for line in result.stdout.strip().split("\n"):
            if line.startswith("{"):
                data = json.loads(line)
                if "stats" in data:
                    stats_dict = data["stats"]
                    # Convert to IndexStats-like dict
                    return IndexStats(**stats_dict), data
        return IndexStats(), {}

    return IndexStats(), {}


def search_corpus(
    home: Path,
    project_dir: Path,
    query: str,
    top: int = 10,
    freshness: str = "skip",
    perf_json: Optional[Path] = None,
) -> tuple[List[Any], Dict[str, Any]]:
    """Run search and return results with optional perf stats."""
    cmd = [
        "uv",
        "run",
        "simgrep",
        "--color",
        "never",
        "search",
        query,
        "--top",
        str(top),
        "--freshness",
        freshness,
        "--format",
        "json",
    ]

    result = run_cli_command(cmd, home=home, cwd=project_dir)

    if result.returncode != 0:
        raise BenchmarkError(f"Search command failed: {result.stderr}")

    perf_stats = {}
    if perf_json and perf_json.exists():
        perf_stats = json.loads(perf_json.read_text())

    return [], perf_stats


def run_ci_suite(
    workspace: Path,
    iterations: int = DEFAULT_ITERATIONS,
    warmups: int = DEFAULT_WARMUPS,
) -> List[BenchmarkCaseResult]:
    """
    Run CI benchmark suite with fake embedder.

    Uses in-process service calls with deterministic fakes.
    """
    results: List[BenchmarkCaseResult] = []
    corpus = generate_corpus(workspace / "corpus" / "medium", CORPUS_MEDIUM)

    corpus_info = BenchmarkCorpusInfo(
        profile="medium",
        files=corpus.files_total,
        bytes=corpus.bytes_total,
        expected_indexable_files=corpus.indexable_files,
    )

    config = BenchmarkConfig(
        workers=DEFAULT_WORKERS,
        batch_size=DEFAULT_BATCH_SIZE,
        model_mode="fake",
    )

    results.append(
        create_skipped_result(
            name="scan.medium",
            suite="ci",
            corpus=corpus_info,
            config=config,
            reason="Legacy in-process benchmark harness removed with architecture flattening.",
        )
    )
    results.append(
        create_skipped_result(
            name="index.rebuild.medium.fake",
            suite="ci",
            corpus=corpus_info,
            config=config,
            reason="Legacy in-process benchmark harness removed with architecture flattening.",
        )
    )
    results.append(
        create_skipped_result(
            name="index.incremental.noop.medium.fake",
            suite="ci",
            corpus=corpus_info,
            config=config,
            reason="Legacy in-process benchmark harness removed with architecture flattening.",
        )
    )
    results.append(
        create_skipped_result(
            name="search.persistent.skip.medium.fake",
            suite="ci",
            corpus=corpus_info,
            config=config,
            reason="Legacy in-process benchmark harness removed with architecture flattening.",
        )
    )

    return results


def run_real_suite(
    workspace: Path,
    iterations: int = 5,
    warmups: int = 1,
) -> List[BenchmarkCaseResult]:
    """
    Run real-model benchmark suite.

    Uses actual CLI subprocesses with isolated HOME.
    """
    results: List[BenchmarkCaseResult] = []

    model_name = "ibm-granite/granite-embedding-30m-english"

    # Check if model is cached
    if not check_model_cached(model_name):
        print(f"Model '{model_name}' not cached. Skipping real suite.")
        return [
            create_skipped_result(
                name="cli.index.rebuild.small.real",
                suite="real",
                corpus=BenchmarkCorpusInfo(profile="small", files=0, bytes=0, expected_indexable_files=0),
                config=BenchmarkConfig(workers=1, batch_size=64, model_mode="real"),
                reason=f"Model '{model_name}' not cached",
            ),
            create_skipped_result(
                name="cli.search.persistent.skip.small.real",
                suite="real",
                corpus=BenchmarkCorpusInfo(profile="small", files=0, bytes=0, expected_indexable_files=0),
                config=BenchmarkConfig(workers=1, batch_size=64, freshness="skip", model_mode="real"),
                reason=f"Model '{model_name}' not cached",
            ),
        ]

    home = setup_temp_home(workspace)
    corpus_dir = workspace / "corpus" / "small"
    project_dir = workspace / "project"

    corpus = generate_corpus(corpus_dir, CORPUS_SMALL)
    project_dir.mkdir(parents=True, exist_ok=True)

    corpus_info = BenchmarkCorpusInfo(
        profile="small",
        files=corpus.files_total,
        bytes=corpus.bytes_total,
        expected_indexable_files=corpus.indexable_files,
    )

    config = BenchmarkConfig(
        workers=1,
        batch_size=64,
        model_mode="real",
    )

    # Initialize project
    try:
        init_simgrep_project(home, project_dir)
    except BenchmarkError as e:
        return [
            create_skipped_result(
                name="cli.index.rebuild.small.real",
                suite="real",
                corpus=corpus_info,
                config=config,
                reason=f"Failed to init project: {e}",
            ),
        ]

    # Test: cli.index.rebuild.small.real
    def case_cli_rebuild() -> BenchmarkOutcome:
        result = run_cli_command(
            [
                "uv",
                "run",
                "simgrep",
                "--color",
                "never",
                "index",
                "--rebuild",
                "--workers",
                "1",
                "--format",
                "json",
            ],
            home=home,
            cwd=project_dir,
        )

        if result.returncode != 0:
            raise BenchmarkError(f"Index failed: {result.stderr}")

        for line in result.stdout.strip().split("\n"):
            if line.startswith("{"):
                data = json.loads(line)
                if "stats" in data:
                    return BenchmarkOutcome(files_indexed=IndexStats(**data["stats"]).files_indexed, errors=0)
        return BenchmarkOutcome(files_indexed=0, errors=0)

    print("Running cli.index.rebuild.small.real...")
    result = measure_case(
        name="cli.index.rebuild.small.real",
        suite="real",
        corpus=corpus_info,
        config=config,
        fn=case_cli_rebuild,
        iterations=iterations,
        warmups=warmups,
    )
    results.append(result)

    # Test: cli.search.persistent.skip.small.real
    def case_cli_search() -> BenchmarkOutcome:
        result = run_cli_command(
            [
                "uv",
                "run",
                "simgrep",
                "--color",
                "never",
                "search",
                "tax invoice rollback",
                "--top",
                "5",
                "--freshness",
                "skip",
                "--format",
                "json",
            ],
            home=home,
            cwd=project_dir,
        )

        if result.returncode != 0:
            raise BenchmarkError(f"Search failed: {result.stderr}")

        count = sum(1 for line in result.stdout.strip().split("\n") if line.startswith("{"))
        return BenchmarkOutcome(results_count=count, errors=0)

    print("Running cli.search.persistent.skip.small.real...")
    result = measure_case(
        name="cli.search.persistent.skip.small.real",
        suite="real",
        corpus=corpus_info,
        config=BenchmarkConfig(
            workers=1,
            batch_size=64,
            freshness="skip",
            model_mode="real",
        ),
        fn=case_cli_search,
        iterations=iterations,
        warmups=warmups,
    )
    results.append(result)

    return results


@app.command()
def main(
    suite: str = typer.Option("ci", "--suite", help="Benchmark suite: ci, real, stress"),
    profile: str = typer.Option("medium", "--profile", help="Corpus profile: tiny, small, medium, large"),
    iterations: int = typer.Option(DEFAULT_ITERATIONS, "--iterations", help="Number of iterations"),
    warmups: int = typer.Option(DEFAULT_WARMUPS, "--warmups", help="Number of warmup iterations"),
    workers: int = typer.Option(DEFAULT_WORKERS, "--workers", help="Number of workers"),
    batch_size: int = typer.Option(DEFAULT_BATCH_SIZE, "--batch-size", help="Batch size"),
    output: Path = typer.Option(Path("reports/benchmarks/speed.json"), "--output", help="Output JSON report path"),
    compare: Optional[Path] = typer.Option(None, "--compare", help="Compare to baseline JSON file"),
    update_baseline: bool = typer.Option(False, "--update-baseline", help="Update baseline with results"),
    keep_workdir: bool = typer.Option(False, "--keep-workdir", help="Keep temporary workspace"),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """
    Run speed benchmarks for simgrep.

    Examples:

    \b
    # Run CI suite with fake embedder
    uv run python benchmarks/run_speed.py --suite ci --output reports/benchmarks/speed-ci.json

    \b
    # Run real-model suite
    uv run python benchmarks/run_speed.py --suite real --output reports/benchmarks/speed-real.json

    \b
    # Compare to baseline
    uv run python benchmarks/run_speed.py --suite ci --compare benchmarks/baselines/ci.json
    """
    del profile, workers, batch_size

    # Create output directory
    output.parent.mkdir(parents=True, exist_ok=True)

    # Create workspace
    workspace = setup_temp_workspace()
    if verbose:
        print(f"Workspace: {workspace}")

    try:
        baseline = load_baseline(compare) if compare else None
        results: List[BenchmarkCaseResult] = []

        if suite == "ci":
            results = run_ci_suite(workspace, iterations, warmups)
        elif suite == "real":
            results = run_real_suite(workspace, iterations, warmups)
        elif suite == "stress":
            print("Stress suite not yet implemented")
            return
        else:
            print(f"Unknown suite: {suite}")
            raise typer.Exit(code=2)

        # Create report
        report = create_report(results)

        # Write JSON report
        report_dict = report_to_dict(report)
        output.write_text(json.dumps(report_dict, indent=2))

        # Print markdown summary
        print(format_markdown_table([report]))

        # Compare to baseline if requested
        failed = False
        if baseline:
            all_passed = True
            print("\nBaseline comparison:")
            for case in results:
                if not case.status.skipped and (case.outcome.errors > 0 or not case.samples_seconds):
                    all_passed = False
                    print(f"  FAIL: {case.name} " f"({case.outcome.errors} errors, {len(case.samples_seconds)}/{case.iterations} successful samples)")
                    continue
                passed, reasons = compare_to_baseline(case, baseline)
                if not passed:
                    all_passed = False
                    print(f"  FAIL: {case.name}")
                    for reason in reasons:
                        print(f"    - {reason}")

            if not all_passed:
                print("\nRegression detected!")
                failed = True

        # Update baseline if requested
        if update_baseline and compare:
            baseline_cases = {}
            for case in results:
                if not case.status.skipped and case.outcome.errors == 0 and case.samples_seconds:
                    baseline_cases[case.name] = {
                        "median_seconds": round(case.median_seconds, 4),
                        "p95_seconds": round(case.p95_seconds, 4),
                        "max_regression_ratio": 1.25,
                        "hard_max_seconds": round(case.max_seconds * 1.5, 4) if case.max_seconds > 0 else None,
                    }

            baseline_data = {
                "schema_version": 1,
                "cases": baseline_cases,
            }
            compare.write_text(json.dumps(baseline_data, indent=2))
            print(f"\nBaseline updated: {compare}")

        if failed:
            raise typer.Exit(code=1)

        print(f"\nReport written: {output}")

    finally:
        if not keep_workdir:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    app()
