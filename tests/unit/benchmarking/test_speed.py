"""Tests for speed benchmark CI suite and baseline comparison."""
# mypy: disable-error-code="no-untyped-def"

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.corpora import (
    CORPUS_TINY,
    MUTATION_NOOP,
    MUTATION_ONE_CHANGED,
    apply_mutation,
    generate_corpus,
)
from benchmarks.measure import (
    BaselineCase,
    BenchmarkCaseResult,
    BenchmarkConfig,
    BenchmarkCorpusInfo,
    BenchmarkOutcome,
    BenchmarkStatus,
    SpeedBaseline,
    compare_to_baseline,
    create_report,
    measure_case,
    report_to_dict,
)


class TestBaselineComparisonFailures:
    """Tests for baseline comparison regression detection."""

    def test_median_regression_fails(self):
        """Baseline comparison fails when median exceeds threshold."""
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
            median_seconds=1.6,
            p95_seconds=1.3,
            min_seconds=1.5,
            max_seconds=1.7,
            outcome=BenchmarkOutcome(errors=0),
            status=BenchmarkStatus(passed=True),
        )

        passed, reasons = compare_to_baseline(result, baseline)
        assert passed is False
        assert any("Median regression" in r for r in reasons)

    def test_p95_regression_fails(self):
        """Baseline comparison fails when p95 exceeds threshold."""
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
            samples_seconds=[1.0, 1.0, 1.0],
            median_seconds=1.0,
            p95_seconds=1.6,
            min_seconds=1.0,
            max_seconds=1.7,
            outcome=BenchmarkOutcome(errors=0),
            status=BenchmarkStatus(passed=True),
        )

        passed, reasons = compare_to_baseline(result, baseline)
        assert passed is False
        assert any("P95 regression" in r for r in reasons)

    def test_hard_max_fails(self):
        """Baseline comparison fails when hard max seconds exceeded."""
        baseline = SpeedBaseline(
            cases={
                "test.case": BaselineCase(
                    median_seconds=1.0,
                    p95_seconds=1.2,
                    max_regression_ratio=1.25,
                    hard_max_seconds=1.5,
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
            samples_seconds=[1.0, 1.1, 2.0],
            median_seconds=1.1,
            p95_seconds=1.3,
            min_seconds=1.0,
            max_seconds=2.0,
            outcome=BenchmarkOutcome(errors=0),
            status=BenchmarkStatus(passed=True),
        )

        passed, reasons = compare_to_baseline(result, baseline)
        assert passed is False
        assert any("Hard max exceeded" in r for r in reasons)

    def test_hard_max_not_checked_when_none(self):
        """Hard max is not checked when not defined in baseline."""
        baseline = SpeedBaseline(
            cases={
                "test.case": BaselineCase(
                    median_seconds=1.0,
                    p95_seconds=1.2,
                    max_regression_ratio=1.25,
                    hard_max_seconds=None,
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
            samples_seconds=[1.0, 1.1, 100.0],
            median_seconds=1.1,
            p95_seconds=1.3,
            min_seconds=1.0,
            max_seconds=100.0,
            outcome=BenchmarkOutcome(errors=0),
            status=BenchmarkStatus(passed=True),
        )

        passed, reasons = compare_to_baseline(result, baseline)
        assert passed is True


class TestReportFormat:
    """Tests for speed report format and validation."""

    def test_report_contains_files_chunks_errors(self):
        """Report must contain files, chunks, and errors to prevent empty corpus benchmarking."""
        corpus = BenchmarkCorpusInfo(
            profile="tiny",
            files=10,
            bytes=1000,
            expected_indexable_files=10,
        )
        config = BenchmarkConfig(workers=1, batch_size=64)

        def case_fn():
            return BenchmarkOutcome(
                files_seen=10,
                files_indexed=8,
                chunks_indexed=50,
                results_count=5,
                errors=0,
            )

        result = measure_case(
            name="test.case",
            suite="ci",
            corpus=corpus,
            config=config,
            fn=case_fn,
            iterations=3,
            warmups=1,
        )

        report = create_report([result])
        report_dict = report_to_dict(report)

        case_dict = report_dict["benchmarks"][0]
        assert case_dict["outcome"]["files_seen"] == 10
        assert case_dict["outcome"]["files_indexed"] == 8
        assert case_dict["outcome"]["chunks_indexed"] == 50
        assert case_dict["outcome"]["errors"] == 0

    def test_empty_corpus_report_has_zero_counts(self):
        """Empty corpus report shows zero counts."""
        corpus = BenchmarkCorpusInfo(
            profile="tiny",
            files=0,
            bytes=0,
            expected_indexable_files=0,
        )
        config = BenchmarkConfig(workers=1, batch_size=64)

        def case_fn():
            return BenchmarkOutcome(
                files_seen=0,
                files_indexed=0,
                chunks_indexed=0,
                errors=1,
            )

        result = measure_case(
            name="test.empty",
            suite="ci",
            corpus=corpus,
            config=config,
            fn=case_fn,
            iterations=3,
            warmups=1,
        )

        report = create_report([result])
        report_dict = report_to_dict(report)

        case_dict = report_dict["benchmarks"][0]
        assert case_dict["outcome"]["files_seen"] == 0
        assert case_dict["outcome"]["files_indexed"] == 0
        assert case_dict["outcome"]["chunks_indexed"] == 0
        assert case_dict["outcome"]["errors"] == 1

    def test_report_json_is_valid(self, tmp_path: Path):
        """Report JSON can be written and read back."""
        corpus = BenchmarkCorpusInfo(
            profile="tiny",
            files=10,
            bytes=1000,
            expected_indexable_files=10,
        )
        config = BenchmarkConfig(workers=1, batch_size=64)

        def case_fn():
            return BenchmarkOutcome(files_indexed=10, errors=0)

        result = measure_case(
            name="test.case",
            suite="ci",
            corpus=corpus,
            config=config,
            fn=case_fn,
            iterations=3,
            warmups=1,
        )

        report = create_report([result])
        report_dict = report_to_dict(report)

        report_path = tmp_path / "report.json"
        report_path.write_text(json.dumps(report_dict, indent=2))

        loaded = json.loads(report_path.read_text())
        assert loaded["schema_version"] == 1
        assert len(loaded["benchmarks"]) == 1
        assert loaded["benchmarks"][0]["name"] == "test.case"


class TestCorpusGenerator:
    """Tests for corpus generator producing expected indexable file counts."""

    def test_tiny_corpus_indexable_count(self, tmp_path: Path):
        """Tiny corpus produces expected number of indexable files."""
        manifest = generate_corpus(tmp_path / "tiny", CORPUS_TINY)

        assert manifest.indexable_files > 0
        assert manifest.indexable_files < manifest.files_total

    def test_corpus_generator_creates_consistent_indexable_count(self, tmp_path: Path):
        """Same seed produces same indexable file count across runs."""
        path1 = tmp_path / "corpus1"
        path2 = tmp_path / "corpus2"

        manifest1 = generate_corpus(path1, CORPUS_TINY)
        manifest2 = generate_corpus(path2, CORPUS_TINY)

        assert manifest1.indexable_files == manifest2.indexable_files

    def test_scan_files_indexes_expected_count(self, tmp_path: Path):
        """scan_files indexes files that corpus generator marks as indexable."""
        from simgrep.files import ScanOptions, scan_files

        manifest = generate_corpus(tmp_path / "scan_test", CORPUS_TINY)

        scan_options = ScanOptions(
            patterns=("*.java", "*.py", "*.md", "*.yml", "*.yaml", "*.json"),
        )
        entries = scan_files(tmp_path / "scan_test", scan_options)

        assert len(entries) > 0
        assert len(entries) <= manifest.indexable_files


class TestBenchmarkMutation:
    """Tests for benchmark mutation affecting IndexEngine.plan_project()."""

    def test_mutation_preserves_some_files(self, tmp_path: Path):
        """Mutation changes content but keeps files."""
        corpus_path = tmp_path / "mutation_test"
        manifest = generate_corpus(corpus_path, CORPUS_TINY)

        original_files = list(corpus_path.rglob("*.java"))
        assert len(original_files) > 0

        mutated = apply_mutation(corpus_path, manifest, MUTATION_ONE_CHANGED)

        assert mutated.indexable_files > 0

    def test_mutation_adds_files(self, tmp_path: Path):
        """Mutation with add_files creates new files."""
        from benchmarks.corpora import MutationPlan

        corpus_path = tmp_path / "add_test"
        manifest = generate_corpus(corpus_path, CORPUS_TINY)

        mutation = MutationPlan(add_files=3)
        apply_mutation(corpus_path, manifest, mutation)

        added_files = list((corpus_path / "added").rglob("*.java"))
        assert len(added_files) == 3

    def test_mutation_deletes_files(self, tmp_path: Path):
        """Mutation with delete_files removes files."""
        from benchmarks.corpora import MutationPlan

        corpus_path = tmp_path / "delete_test"
        manifest = generate_corpus(corpus_path, CORPUS_TINY)

        mutation = MutationPlan(delete_files=3)
        mutated = apply_mutation(corpus_path, manifest, mutation)

        assert mutated.indexable_files < manifest.indexable_files

    def test_noop_mutation_only_counts_java_python_md(self, tmp_path: Path):
        """NOOP mutation only counts .java, .py, .md files (not yaml/json)."""
        corpus_path = tmp_path / "noop_test"
        manifest = generate_corpus(corpus_path, CORPUS_TINY)

        mutated = apply_mutation(corpus_path, manifest, MUTATION_NOOP)

        java_count = len(list(corpus_path.rglob("*.java")))
        py_count = len(list(corpus_path.rglob("*.py")))
        md_count = len(list(corpus_path.rglob("*.md")))
        expected = java_count + py_count + md_count

        assert mutated.indexable_files == expected
        assert expected < manifest.indexable_files


class TestModelCacheSkip:
    """Tests for model cache skip being machine-readable."""

    def test_skipped_result_has_machine_readable_reason(self):
        """Skipped result contains machine-readable skip reason."""
        result = BenchmarkCaseResult(
            name="test.skip",
            suite="real",
            corpus=BenchmarkCorpusInfo("small", 100, 10000, 100),
            config=BenchmarkConfig(workers=1, batch_size=64, model_mode="real"),
            iterations=0,
            warmups=0,
            samples_seconds=[],
            median_seconds=0.0,
            p95_seconds=0.0,
            min_seconds=0.0,
            max_seconds=0.0,
            outcome=BenchmarkOutcome(),
            status=BenchmarkStatus(skipped=True, skip_reason="Model 'ibm-granite/granite-embedding-30m-english' not cached"),
        )

        assert result.status.skipped is True
        assert result.status.skip_reason is not None
        assert "not cached" in result.status.skip_reason
        assert "Model" in result.status.skip_reason

    def test_report_preserves_skip_reason(self):
        """Report JSON preserves skip reason exactly."""
        result = BenchmarkCaseResult(
            name="test.skip",
            suite="real",
            corpus=BenchmarkCorpusInfo("small", 100, 10000, 100),
            config=BenchmarkConfig(workers=1, batch_size=64, model_mode="real"),
            iterations=0,
            warmups=0,
            samples_seconds=[],
            median_seconds=0.0,
            p95_seconds=0.0,
            min_seconds=0.0,
            max_seconds=0.0,
            outcome=BenchmarkOutcome(),
            status=BenchmarkStatus(skipped=True, skip_reason="Model 'test-model' not cached"),
        )

        report = create_report([result])
        report_dict = report_to_dict(report)

        case_dict = report_dict["benchmarks"][0]
        assert case_dict["status"]["skipped"] is True
        assert case_dict["status"]["skip_reason"] == "Model 'test-model' not cached"

    def test_skip_does_not_fail_comparison(self):
        """Skipped results pass baseline comparison regardless of baseline."""
        baseline = SpeedBaseline(cases={})

        result = BenchmarkCaseResult(
            name="test.skip",
            suite="real",
            corpus=BenchmarkCorpusInfo("small", 100, 10000, 100),
            config=BenchmarkConfig(workers=1, batch_size=64, model_mode="real"),
            iterations=0,
            warmups=0,
            samples_seconds=[],
            median_seconds=0.0,
            p95_seconds=0.0,
            min_seconds=0.0,
            max_seconds=0.0,
            outcome=BenchmarkOutcome(),
            status=BenchmarkStatus(skipped=True, skip_reason="Model not cached"),
        )

        passed, reasons = compare_to_baseline(result, baseline)
        assert passed is True
        assert len(reasons) == 0
