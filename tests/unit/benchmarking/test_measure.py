"""Tests for benchmark measurement utilities."""
# mypy: disable-error-code="no-untyped-def"

import json
import time
from pathlib import Path

from benchmarks.measure import (
    BaselineCase,
    BenchmarkCaseResult,
    BenchmarkConfig,
    BenchmarkCorpusInfo,
    BenchmarkOutcome,
    BenchmarkStatus,
    SpeedBaseline,
    calculate_stats,
    compare_to_baseline,
    create_report,
    create_skipped_result,
    load_baseline,
    measure_case,
    measure_case_with_phases,
    report_to_dict,
)


class TestCalculateStats:
    """Tests for statistics calculation."""

    def test_calculate_stats_basic(self):
        """Test basic statistics calculation."""
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = calculate_stats(samples)

        assert stats["median"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        # P95 of 5 elements is index 4 (floor(5 * 0.95) = 4)
        assert stats["p95"] == 5.0

    def test_calculate_stats_empty(self):
        """Test calculation with empty samples."""
        stats = calculate_stats([])

        assert stats["median"] == 0.0
        assert stats["min"] == 0.0
        assert stats["max"] == 0.0

    def test_calculate_stats_single(self):
        """Test calculation with single sample."""
        stats = calculate_stats([42.0])

        assert stats["median"] == 42.0
        assert stats["min"] == 42.0
        assert stats["max"] == 42.0

    def test_calculate_stats_even_count(self):
        """Test calculation with even number of samples."""
        samples = [1.0, 2.0, 3.0, 4.0]
        stats = calculate_stats(samples)

        assert stats["median"] == 2.5  # Average of middle two


class TestMeasureCase:
    """Tests for benchmark case measurement."""

    def test_measure_case_basic(self, tmp_path: Path):
        """Test basic case measurement."""
        corpus = BenchmarkCorpusInfo(
            profile="tiny",
            files=10,
            bytes=1000,
            expected_indexable_files=10,
        )
        config = BenchmarkConfig(workers=1, batch_size=64)

        def case_fn():
            time.sleep(0.01)
            return BenchmarkOutcome(files_seen=10, errors=0)

        result = measure_case(
            name="test.basic",
            suite="ci",
            corpus=corpus,
            config=config,
            fn=case_fn,
            iterations=3,
            warmups=1,
        )

        assert result.name == "test.basic"
        assert result.suite == "ci"
        assert result.iterations == 3
        assert result.warmups == 1
        assert len(result.samples_seconds) == 3
        assert result.median_seconds > 0
        assert result.status.passed is True

    def test_measure_case_handles_errors(self, tmp_path: Path):
        """Test that measure_case handles errors gracefully."""
        corpus = BenchmarkCorpusInfo(
            profile="tiny",
            files=10,
            bytes=1000,
            expected_indexable_files=10,
        )
        config = BenchmarkConfig(workers=1, batch_size=64)

        error_count = [0]

        def case_fn():
            error_count[0] += 1
            if error_count[0] <= 1:
                # First warmup fails
                raise RuntimeError("Warmup fails")
            return BenchmarkOutcome(files_seen=10, errors=0)

        # Should not raise, just return result with errors tracked
        result = measure_case(
            name="test.error",
            suite="ci",
            corpus=corpus,
            config=config,
            fn=case_fn,
            iterations=2,
            warmups=1,
        )

        # Should complete without raising
        assert result.name == "test.error"
        assert result.status.passed is False
        assert result.outcome.errors == 1

    def test_measure_case_counts_warmup_errors(self, tmp_path: Path):
        """Warmup errors must survive subsequent successful iterations."""
        corpus = BenchmarkCorpusInfo(
            profile="tiny",
            files=10,
            bytes=1000,
            expected_indexable_files=10,
        )
        config = BenchmarkConfig(workers=1, batch_size=64)
        failed = [False]

        def case_fn():
            if not failed[0]:
                failed[0] = True
                raise RuntimeError("Warmup fails")
            return BenchmarkOutcome(files_seen=10, errors=0)

        result = measure_case(
            name="test.warmup.error",
            suite="ci",
            corpus=corpus,
            config=config,
            fn=case_fn,
            iterations=3,
            warmups=2,
        )

        assert result.outcome.errors == 1
        assert result.status.passed is False
        assert len(result.samples_seconds) == 3

    def test_measure_case_with_phases_counts_warmup_errors(self, tmp_path: Path):
        """Warmup errors must survive successful iterations in phased measurement."""
        corpus = BenchmarkCorpusInfo(
            profile="tiny",
            files=10,
            bytes=1000,
            expected_indexable_files=10,
        )
        config = BenchmarkConfig(workers=1, batch_size=64)
        failed = [False]

        def case_fn():
            if not failed[0]:
                failed[0] = True
                raise RuntimeError("Warmup fails")
            return BenchmarkOutcome(files_indexed=10), {"plan_seconds": 0.1}

        result = measure_case_with_phases(
            name="test.phases.warmup.error",
            suite="ci",
            corpus=corpus,
            config=config,
            fn=case_fn,
            iterations=3,
            warmups=2,
        )

        assert result.outcome.errors == 1
        assert result.status.passed is False
        assert result.phases is not None
        assert len(result.samples_seconds) == 3

    def test_measure_case_warmups(self, tmp_path: Path):
        """Test that warmups are executed."""
        corpus = BenchmarkCorpusInfo(
            profile="tiny",
            files=10,
            bytes=1000,
            expected_indexable_files=10,
        )
        config = BenchmarkConfig(workers=1, batch_size=64)

        warmup_count = 0
        iteration_count = 0

        def case_fn():
            nonlocal warmup_count, iteration_count
            if warmup_count < 2:  # warmups=2
                warmup_count += 1
            else:
                iteration_count += 1
            return BenchmarkOutcome(files_seen=10, errors=0)

        result = measure_case(
            name="test.warmup",
            suite="ci",
            corpus=corpus,
            config=config,
            fn=case_fn,
            iterations=3,
            warmups=2,
        )

        assert warmup_count == 2
        assert iteration_count == 3
        assert len(result.samples_seconds) == 3


class TestCreateSkippedResult:
    """Tests for skipped result creation."""

    def test_create_skipped_result(self):
        """Test creating a skipped result."""
        corpus = BenchmarkCorpusInfo(
            profile="tiny",
            files=10,
            bytes=1000,
            expected_indexable_files=10,
        )
        config = BenchmarkConfig(workers=1, batch_size=64)

        result = create_skipped_result(
            name="test.skipped",
            suite="ci",
            corpus=corpus,
            config=config,
            reason="Model not cached",
        )

        assert result.name == "test.skipped"
        assert result.suite == "ci"
        assert result.status.skipped is True
        assert result.status.skip_reason == "Model not cached"
        assert result.iterations == 0
        assert len(result.samples_seconds) == 0


class TestBaselineComparison:
    """Tests for baseline comparison."""

    def test_compare_to_baseline_pass(self):
        """Test comparison with passing result."""
        baseline = SpeedBaseline(
            cases={
                "test.case": BaselineCase(
                    median_seconds=1.0,
                    p95_seconds=1.2,
                    max_regression_ratio=1.25,
                )
            }
        )

        result = BenchmarkCaseResult(
            name="test.case",
            suite="ci",
            corpus=BenchmarkCorpusInfo("tiny", 10, 1000, 10),
            config=BenchmarkConfig(1, 64),
            iterations=10,
            warmups=2,
            samples_seconds=[1.0, 1.1, 1.2],
            median_seconds=1.1,  # Within 1.25x of 1.0
            p95_seconds=1.3,  # Within 1.25x of 1.2
            min_seconds=1.0,
            max_seconds=1.2,
            outcome=BenchmarkOutcome(errors=0),
            status=BenchmarkStatus(passed=True),
        )

        passed, reasons = compare_to_baseline(result, baseline)
        assert passed is True
        assert len(reasons) == 0

    def test_compare_to_baseline_median_regression(self):
        """Test comparison with median regression."""
        baseline = SpeedBaseline(
            cases={
                "test.case": BaselineCase(
                    median_seconds=1.0,
                    p95_seconds=1.2,
                    max_regression_ratio=1.25,
                )
            }
        )

        result = BenchmarkCaseResult(
            name="test.case",
            suite="ci",
            corpus=BenchmarkCorpusInfo("tiny", 10, 1000, 10),
            config=BenchmarkConfig(1, 64),
            iterations=10,
            warmups=2,
            samples_seconds=[1.5, 1.6, 1.7],
            median_seconds=1.6,  # Exceeds 1.25x of 1.0
            p95_seconds=1.8,
            min_seconds=1.5,
            max_seconds=1.7,
            outcome=BenchmarkOutcome(errors=0),
            status=BenchmarkStatus(passed=True),
        )

        passed, reasons = compare_to_baseline(result, baseline)
        assert passed is False
        assert len(reasons) > 0

    def test_compare_to_baseline_missing(self):
        """Test comparison with missing baseline."""
        baseline = SpeedBaseline(cases={})

        result = BenchmarkCaseResult(
            name="test.missing",
            suite="ci",
            corpus=BenchmarkCorpusInfo("tiny", 10, 1000, 10),
            config=BenchmarkConfig(1, 64),
            iterations=10,
            warmups=2,
            samples_seconds=[1.0, 1.1, 1.2],
            median_seconds=1.1,
            p95_seconds=1.3,
            min_seconds=1.0,
            max_seconds=1.2,
            outcome=BenchmarkOutcome(errors=0),
            status=BenchmarkStatus(passed=True),
        )

        passed, reasons = compare_to_baseline(result, baseline)
        assert passed is False
        assert "No baseline" in reasons[0]


class TestReportSerialization:
    """Tests for report serialization."""

    def test_report_to_dict(self):
        """Test report to dictionary conversion."""
        report = create_report(
            [
                BenchmarkCaseResult(
                    name="test.case",
                    suite="ci",
                    corpus=BenchmarkCorpusInfo("tiny", 10, 1000, 10),
                    config=BenchmarkConfig(1, 64),
                    iterations=10,
                    warmups=2,
                    samples_seconds=[1.0, 1.1, 1.2],
                    median_seconds=1.1,
                    p95_seconds=1.3,
                    min_seconds=1.0,
                    max_seconds=1.2,
                    outcome=BenchmarkOutcome(errors=0),
                    status=BenchmarkStatus(passed=True),
                )
            ]
        )

        report_dict = report_to_dict(report)

        assert report_dict["schema_version"] == 1
        assert "created_at" in report_dict
        assert len(report_dict["benchmarks"]) == 1
        assert report_dict["benchmarks"][0]["name"] == "test.case"

    def test_report_json_roundtrip(self, tmp_path: Path):
        """Test JSON roundtrip of report."""
        report = create_report(
            [
                BenchmarkCaseResult(
                    name="test.case",
                    suite="ci",
                    corpus=BenchmarkCorpusInfo("tiny", 10, 1000, 10),
                    config=BenchmarkConfig(1, 64),
                    iterations=10,
                    warmups=2,
                    samples_seconds=[1.0, 1.1, 1.2],
                    median_seconds=1.1,
                    p95_seconds=1.3,
                    min_seconds=1.0,
                    max_seconds=1.2,
                    outcome=BenchmarkOutcome(errors=0),
                    status=BenchmarkStatus(passed=True),
                )
            ]
        )

        report_dict = report_to_dict(report)
        json_str = json.dumps(report_dict, indent=2)
        loaded = json.loads(json_str)

        assert loaded["schema_version"] == 1
        assert len(loaded["benchmarks"]) == 1


class TestLoadBaseline:
    """Tests for baseline loading."""

    def test_load_nonexistent_baseline(self, tmp_path: Path):
        """Test loading a non-existent baseline."""
        baseline = load_baseline(tmp_path / "nonexistent.json")

        assert baseline.schema_version == 1
        assert len(baseline.cases) == 0

    def test_load_valid_baseline(self, tmp_path: Path):
        """Test loading a valid baseline file."""
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "cases": {
                        "test.case": {
                            "median_seconds": 1.0,
                            "p95_seconds": 1.2,
                            "max_regression_ratio": 1.25,
                            "hard_max_seconds": 2.0,
                        }
                    },
                }
            )
        )

        baseline = load_baseline(baseline_path)

        assert baseline.schema_version == 1
        assert "test.case" in baseline.cases
        assert baseline.cases["test.case"].median_seconds == 1.0
        assert baseline.cases["test.case"].p95_seconds == 1.2
        assert baseline.cases["test.case"].max_regression_ratio == 1.25
        assert baseline.cases["test.case"].hard_max_seconds == 2.0
