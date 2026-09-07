"""Round-17 E2E edge contracts for `simgrep diff`: empty trees, file operands,
nested paths, config-driven ephemeral scanning, threshold boundaries, and
error/stream hygiene. Complements (never repeats) tests/e2e/test_cli_diff.py."""

from __future__ import annotations

import hashlib
import json
import pathlib
import re

import numpy as np
import pytest

from tests.conftest import FakeTextExtractor, FakeTokenChunker, FakeVectorIndex
from tests.e2e.conftest import (
    assert_success,
    run_simgrep_command,
)


class TokenSetHashingEmbedder:
    """Deterministic fake embedder: unit vector per token, summed and normalized.

    Identical texts collide at cosine 1.0; texts with disjoint token sets are
    near-orthogonal, so a renamed-but-unchanged file matches perfectly while a
    reworded one falls below the diff threshold.
    """

    ndim = 64

    def encode(self, texts: list[str], *, is_query: bool = False, batch_size: int | None = None) -> np.ndarray:
        del is_query, batch_size
        vectors = np.zeros((len(texts), self.ndim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                digest = hashlib.md5(token.encode("utf-8")).digest()
                rng = np.random.default_rng(int.from_bytes(digest[:4], "little"))
                vectors[row] += rng.standard_normal(self.ndim).astype(np.float32)
            norm = float(np.linalg.norm(vectors[row]))
            if norm > 0:
                vectors[row] /= norm
        return vectors


class RoundTripVectorIndex(FakeVectorIndex):
    """FakeVectorIndex whose save/load actually preserves vectors."""

    def save(self, path: pathlib.Path) -> None:
        payload = {str(label): vector.tolist() for label, vector in sorted(self.data.items())}
        path.write_text(json.dumps(payload), encoding="utf-8")

    def load(self, path: pathlib.Path) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.data = {int(label): np.asarray(vector, dtype=np.float32) for label, vector in raw.items()}


class _HashingRuntime:
    def __init__(self) -> None:
        self.extractor = FakeTextExtractor()
        self.chunker = FakeTokenChunker()
        self.embedder = TokenSetHashingEmbedder()

    def new_vector_index(self, ndim: int) -> RoundTripVectorIndex:
        return RoundTripVectorIndex(ndim)


@pytest.fixture(autouse=True)
def hashing_runtime_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the conftest FakeEmbedder runtime with the token-set hashing one."""
    runtime = _HashingRuntime()

    class _Factory:
        def for_app(self, config: object) -> _HashingRuntime:
            del config
            return runtime

        def for_project(self, config: object) -> _HashingRuntime:
            del config
            return runtime

    monkeypatch.setattr("simgrep.execution.RuntimeFactory", _Factory)


_ALPHA = "alpha beta gamma delta epsilon zeta\n" * 4
_BETA = "the quick brown fox jumps over the lazy dog again and again\n" * 3
# Two tokens reworded vs _ALPHA: probe-measured token-set cosine ~0.71
# (above 0.6, below 0.8).
_ALPHA_REWORDED = "alpha beta gamma delta eta theta\n" * 4


def _write(path: pathlib.Path, text: str) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_config(temp_simgrep_home: pathlib.Path, text: str) -> None:
    config_dir = temp_simgrep_home / ".config" / "simgrep"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.toml").write_text(text, encoding="utf-8")


def test_diff_empty_tree_versus_populated_reports_pure_sides(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    root = tmp_path
    populated = root / "populated"
    _write(populated / "a.py", _ALPHA)
    _write(populated / "b.py", _BETA)
    (root / "hollow").mkdir()

    removed = run_simgrep_command(["diff", "--format", "count", "populated", "hollow"], cwd=root)
    assert_success(removed)
    assert removed.stdout.strip() == "0 matched, 0 added, 2 removed"
    assert removed.stderr == ""

    added = run_simgrep_command(["diff", "--format", "count", "hollow", "populated"], cwd=root)
    assert_success(added)
    assert added.stdout.strip() == "0 matched, 2 added, 0 removed"

    payload_result = run_simgrep_command(["diff", "--format", "json", "hollow", "populated"], cwd=root)
    assert_success(payload_result)
    payload = json.loads(payload_result.stdout)
    assert [entry["file_path"] for entry in payload["added"]] == [
        "populated/a.py",
        "populated/b.py",
    ]
    assert payload["removed"] == []
    assert [entry["file_path"] for entry in payload["files"]] == [
        "populated/a.py",
        "populated/b.py",
    ]
    assert [entry["added"] for entry in payload["files"]] == [1, 1]
    assert [entry["removed"] for entry in payload["files"]] == [0, 0]


def test_diff_identical_single_files_report_semantic_identity(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    root = tmp_path
    _write(root / "old.py", _ALPHA)
    _write(root / "new.py", _ALPHA)

    count = run_simgrep_command(["diff", "--format", "count", "old.py", "new.py"], cwd=root)
    assert_success(count)
    assert count.stdout.strip() == "1 matched, 0 added, 0 removed"

    rich = run_simgrep_command(["diff", "old.py", "new.py"], cwd=root)
    assert_success(rich)
    assert rich.stdout == ("Semantic Diff: old.py -> new.py\n" "1 matched, 0 added, 0 removed\n" "Trees are semantically identical.\n")


def test_diff_disjoint_single_files_report_replacement_pair(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    root = tmp_path
    _write(root / "old.py", _ALPHA)
    _write(root / "new.py", _BETA)

    result = run_simgrep_command(["diff", "--format", "jsonl", "old.py", "new.py"], cwd=root)
    assert_success(result)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert [(row["kind"], row["file_path"]) for row in rows] == [
        ("added", "new.py"),
        ("removed", "old.py"),
    ]
    assert all(row["line_start"] == 1 for row in rows)


def test_diff_file_argument_against_directory_matches_counterpart(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    root = tmp_path
    _write(root / "solo.py", _ALPHA)
    _write(root / "tree" / "solo.py", _ALPHA)
    _write(root / "tree" / "extra.py", _BETA)

    count = run_simgrep_command(["diff", "--format", "count", "solo.py", "tree"], cwd=root)
    assert_success(count)
    assert count.stdout.strip() == "1 matched, 1 added, 0 removed"

    jsonl = run_simgrep_command(["diff", "--format", "jsonl", "solo.py", "tree"], cwd=root)
    assert_success(jsonl)
    rows = [json.loads(line) for line in jsonl.stdout.splitlines()]
    assert [(row["kind"], row["file_path"]) for row in rows] == [("added", "tree/extra.py")]


def test_diff_nested_trees_render_relative_paths_and_ordered_rollups(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    root = tmp_path
    _write(root / "tree_a" / "pkg" / "inner" / "mod.py", _ALPHA)
    _write(root / "tree_a" / "docs.md", _BETA)
    _write(root / "tree_b" / "pkg" / "inner" / "mod.py", _ALPHA)
    _write(root / "tree_b" / "pkg" / "newmod.py", _ALPHA)
    result = run_simgrep_command(["diff", "--format", "json", "tree_a", "tree_b"], cwd=root)
    assert_success(result)
    payload = json.loads(result.stdout)
    assert payload["matched"] == 1
    assert [entry["file_path"] for entry in payload["added"]] == ["tree_b/pkg/newmod.py"]
    assert [entry["file_path"] for entry in payload["removed"]] == ["tree_a/docs.md"]
    # Rollups ordered by -(added+removed) descending, then file_path ascending.
    assert [entry["file_path"] for entry in payload["files"]] == [
        "tree_a/docs.md",
        "tree_b/pkg/newmod.py",
        "tree_a/pkg/inner/mod.py",
        "tree_b/pkg/inner/mod.py",
    ]

    absolute = run_simgrep_command(["diff", "--format", "jsonl", "--absolute-paths", "tree_a", "tree_b"], cwd=root)
    assert_success(absolute)
    for line in absolute.stdout.splitlines():
        row = json.loads(line)
        assert row["file_path"].startswith(str(root))


def test_diff_ephemeral_scan_honors_configured_file_patterns(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    _write_config(temp_simgrep_home, 'file_patterns = ["*.md"]\n')
    root = tmp_path
    _write(root / "tree_a" / "keep.txt", _BETA)
    _write(root / "tree_b" / "keep.txt", _BETA + "\n")
    _write(root / "tree_b" / "ignored.py", _ALPHA)
    _write(root / "tree_b" / "seen.md", _ALPHA)

    result = run_simgrep_command(["diff", "--format", "json", "tree_a", "tree_b"], cwd=root)
    assert_success(result)
    payload = json.loads(result.stdout)
    assert payload["chunks_a"] == 0
    assert payload["chunks_b"] == 1
    assert [entry["file_path"] for entry in payload["added"]] == ["tree_b/seen.md"]


def test_diff_ephemeral_scan_applies_max_file_size_boundary_inclusively(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    limit = len(_BETA.encode())
    assert len((_BETA + "x").encode()) > limit
    _write_config(temp_simgrep_home, f"max_file_size_bytes = {limit}\n")
    root = tmp_path
    _write(root / "tree_b" / "at_limit.py", _BETA)
    _write(root / "tree_b" / "over_limit.py", _BETA + "x")
    assert (root / "tree_b" / "at_limit.py").stat().st_size == limit
    (root / "tree_a").mkdir()  # real empty A-side: engines reject non-existent paths (a8c3ab5)

    result = run_simgrep_command(["diff", "--format", "json", "tree_a", "tree_b"], cwd=root)
    assert_success(result)
    payload = json.loads(result.stdout)
    assert payload["chunks_a"] == 0
    assert payload["chunks_b"] == 1
    assert [entry["file_path"] for entry in payload["added"]] == ["tree_b/at_limit.py"]


def test_diff_accepts_threshold_one_and_matches_identical_chunks(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    root = tmp_path
    _write(root / "ta" / "doc.py", _ALPHA)
    _write(root / "tb" / "doc.py", _ALPHA)

    result = run_simgrep_command(["diff", "--threshold", "1.0", "--format", "json", "ta", "tb"], cwd=root)
    assert_success(result)
    payload = json.loads(result.stdout)
    assert payload["threshold"] == 1.0
    assert payload["matched"] == 1
    assert payload["added"] == []
    assert payload["removed"] == []


def test_diff_lowering_threshold_recovers_near_duplicate_chunk(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    root = tmp_path
    _write(root / "ta" / "doc.py", _ALPHA)
    _write(root / "tb" / "doc.py", _ALPHA_REWORDED)

    strict_result = run_simgrep_command(["diff", "--threshold", "0.8", "--format", "json", "ta", "tb"], cwd=root)
    assert_success(strict_result)
    strict = json.loads(strict_result.stdout)
    assert strict["matched"] == 0
    assert len(strict["added"]) == 1

    loose_result = run_simgrep_command(["diff", "--threshold", "0.6", "--format", "json", "ta", "tb"], cwd=root)
    assert_success(loose_result)
    loose = json.loads(loose_result.stdout)
    assert loose["added"] == []
    assert loose["removed"] == []

    assert loose["matched"] >= strict["matched"]
    assert len(loose["added"]) <= len(strict["added"])


def test_diff_missing_input_paths_rejected_with_error_and_hint(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """Pins the typo guard: non-existent operand paths are rejected up front with an
    actionable hint instead of silently diffing empty trees (behavior change in a8c3ab5)."""
    result = run_simgrep_command(["diff", "--format", "count", "does_not_exist_a", "does_not_exist_b"], cwd=tmp_path)
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Path not found:" in result.stderr
    assert "Check the path and try again." in result.stderr


def test_diff_identical_path_arguments_collapse_to_no_changes(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """Pins current behavior: passing the same operand path twice yields a
    single ephemeral corpus, so both sides report zero changes and exit 0."""
    _write(tmp_path / "t", _ALPHA)

    result = run_simgrep_command(["diff", "--format", "count", "t", "t"], cwd=tmp_path)
    assert_success(result)
    assert result.stdout.strip() == "0 matched, 0 added, 0 removed"


def test_diff_max_chunks_guard_fails_with_error_and_hint_on_stderr(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    root = tmp_path
    _write(root / "ta" / "doc.py", _ALPHA)
    _write(root / "tb" / "doc.py", _ALPHA)

    result = run_simgrep_command(["diff", "--max-chunks", "1", "ta", "tb"], cwd=root)
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Error: 2 chunks exceed --max-chunks 1." in result.stderr
    assert "Hint:" in result.stderr
