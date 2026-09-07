"""E2E coverage for `simgrep diff` (CLI surface, output contracts, rename invisibility)."""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
from typing import Sequence

import numpy as np
import pytest

from tests.conftest import FakeTextExtractor, FakeTokenChunker, FakeVectorIndex
from tests.e2e.conftest import (
    assert_failure_contains,
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


_INVOICE_BODY = (
    "def compute_invoice_total(items):\n"
    "    subtotal = sum(item.price * item.quantity for item in items)\n"
    "    tax = subtotal * TAX_RATE\n"
    "    shipping = flat_shipping_fee(items)\n"
    "    return round(subtotal + tax + shipping, 2)\n"
)

_SESSION_BODY = (
    "def refresh_session_token(user):\n" "    token = issue_signed_token(user.id)\n" "    session_cache.store(user.id, token)\n" "    return token\n"
)

_TABLE_BODY = (
    "def render_markdown_table(rows):\n"
    "    header = rows[0]\n"
    "    lines = ['| ' + ' | '.join(header) + ' |']\n"
    "    lines.append('|' + '---|' * len(header))\n"
    "    lines.extend('| ' + ' | '.join(row) + ' |' for row in rows[1:])\n"
    "    return '\\n'.join(lines)\n"
)

_REWORDED_BODY = (
    "def compress_log_archive(directory):\n"
    "    payload = read_binary_stream(directory)\n"
    "    packed = deflate_bytes(payload)\n"
    "    write_output_file(directory, packed)\n"
    "    return len(packed)\n"
)

_PROMO_BODY = (
    "def validate_coupon_code(code):\n"
    "    record = lookup_promotion(code)\n"
    "    if record is None or record.expired:\n"
    "        raise InvalidPromotionError(code)\n"
    "    return record.discount_rate\n"
)


@pytest.fixture
def changed_trees(tmp_path: pathlib.Path) -> pathlib.Path:
    """Root holding tree_a (3 files) and tree_b derived from a copy.

    Mutations from A to B: ``invoice.py`` renamed to ``invoice_v2.py`` with
    byte-identical content (rename-invisibility target), ``session.py``
    reworded into an unrelated body, ``markdown.py`` deleted, and a brand-new
    ``promo.py`` added.
    """
    root = tmp_path / "workspace"
    tree_a = root / "tree_a"
    tree_b = root / "tree_b"
    tree_a.mkdir(parents=True)
    (tree_a / "invoice.py").write_text(_INVOICE_BODY, encoding="utf-8")
    (tree_a / "session.py").write_text(_SESSION_BODY, encoding="utf-8")
    (tree_a / "markdown.py").write_text(_TABLE_BODY, encoding="utf-8")

    import shutil

    shutil.copytree(tree_a, tree_b)
    (tree_b / "invoice.py").rename(tree_b / "invoice_v2.py")  # identical content
    (tree_b / "session.py").write_text(_REWORDED_BODY, encoding="utf-8")
    (tree_b / "markdown.py").unlink()
    (tree_b / "promo.py").write_text(_PROMO_BODY, encoding="utf-8")
    return root


@pytest.fixture
def identical_trees(tmp_path: pathlib.Path) -> pathlib.Path:
    """Root holding two byte-identical trees."""
    import shutil

    root = tmp_path / "same"
    tree_a = root / "tree_a"
    tree_a.mkdir(parents=True)
    (tree_a / "invoice.py").write_text(_INVOICE_BODY, encoding="utf-8")
    (tree_a / "session.py").write_text(_SESSION_BODY, encoding="utf-8")
    shutil.copytree(tree_a, root / "tree_b")
    return root


def test_diff_rename_is_invisible_in_changed_trees(temp_simgrep_home: pathlib.Path, changed_trees: pathlib.Path) -> None:
    """The renamed file must contribute ZERO added/removed entries."""
    result = run_simgrep_command(["diff", "tree_a", "tree_b", "--format", "json"], cwd=changed_trees)
    assert_success(result)
    payload = json.loads(result.stdout)
    added_paths = {entry["file_path"] for entry in payload["added"]}
    removed_paths = {entry["file_path"] for entry in payload["removed"]}
    assert added_paths == {"tree_b/session.py", "tree_b/promo.py"}
    assert removed_paths == {"tree_a/session.py", "tree_a/markdown.py"}
    for side in (payload["added"], payload["removed"]):
        for entry in side:
            assert "invoice" not in entry["file_path"]
    # Only the rename pairs survived matching: 3 chunks per tree, 2+2 unmatched.
    assert payload["matched"] == 1
    assert payload["chunks_a"] == 3
    assert payload["chunks_b"] == 3
    renamed_rollups = [rollup for rollup in payload["files"] if "invoice" in rollup["file_path"]]
    assert renamed_rollups, "renamed file must still appear in the per-file rollup"
    for rollup in renamed_rollups:
        assert rollup["added"] == 0
        assert rollup["removed"] == 0


def test_diff_identical_trees_report_zero_changes(temp_simgrep_home: pathlib.Path, identical_trees: pathlib.Path) -> None:
    count_result = run_simgrep_command(["diff", "tree_a", "tree_b", "--format", "count"], cwd=identical_trees)
    assert_success(count_result)
    assert count_result.stdout.strip() == "2 matched, 0 added, 0 removed"

    rich_result = run_simgrep_command(["diff", "tree_a", "tree_b"], cwd=identical_trees)
    assert_success(rich_result)
    assert "0 added, 0 removed" in rich_result.stdout


def test_diff_json_matches_pinned_schema(temp_simgrep_home: pathlib.Path, changed_trees: pathlib.Path) -> None:
    result = run_simgrep_command(["diff", "tree_a", "tree_b", "--format", "json"], cwd=changed_trees)
    assert_success(result)
    payload = json.loads(result.stdout)
    assert set(payload.keys()) == {
        "matched",
        "chunks_a",
        "chunks_b",
        "threshold",
        "added",
        "removed",
        "files",
    }
    for entry in [*payload["added"], *payload["removed"]]:
        assert set(entry.keys()) == {"label", "file_path", "line_start", "line_end"}
        assert isinstance(entry["label"], int)
        assert entry["line_end"] >= entry["line_start"] >= 1
    for rollup in payload["files"]:
        assert set(rollup.keys()) == {"file_path", "added", "removed", "matched"}
    assert payload["threshold"] == 0.8


def test_diff_jsonl_records_carry_kind(temp_simgrep_home: pathlib.Path, changed_trees: pathlib.Path) -> None:
    result = run_simgrep_command(["diff", "tree_a", "tree_b", "--format", "jsonl"], cwd=changed_trees)
    assert_success(result)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows) == 4  # 2 added + 2 removed chunks
    kinds = {row["kind"] for row in rows}
    assert kinds == {"added", "removed"}
    for row in rows:
        assert set(row.keys()) == {
            "kind",
            "label",
            "file_path",
            "line_start",
            "line_end",
        }
    added_rows = [row for row in rows if row["kind"] == "added"]
    assert {row["file_path"] for row in added_rows} == {
        "tree_b/session.py",
        "tree_b/promo.py",
    }


def test_diff_count_format_single_line(temp_simgrep_home: pathlib.Path, changed_trees: pathlib.Path) -> None:
    result = run_simgrep_command(["diff", "tree_a", "tree_b", "--format", "count"], cwd=changed_trees)
    assert_success(result)
    assert re.fullmatch(r"1 matched, 2 added, 2 removed\n?", result.stdout)


def test_diff_rich_output_sections_and_rollup_table(temp_simgrep_home: pathlib.Path, changed_trees: pathlib.Path) -> None:
    result = run_simgrep_command(["diff", "tree_a", "tree_b"], cwd=changed_trees)
    assert_success(result)
    assert "Semantic Diff:" in result.stdout
    assert "tree_a -> tree_b" in result.stdout
    assert "1 matched, 2 added, 2 removed" in result.stdout
    assert "Added" in result.stdout
    assert "Removed" in result.stdout
    assert not re.search(r"invoice_v2\.py:\d+-\d+", result.stdout)
    assert "markdown.py" in result.stdout
    assert "promo.py" in result.stdout


def test_diff_threshold_validation_rejected_with_hint(temp_simgrep_home: pathlib.Path, changed_trees: pathlib.Path) -> None:
    zero = run_simgrep_command(["diff", "tree_a", "tree_b", "--threshold", "0"], cwd=changed_trees)
    assert_failure_contains(zero, ["--threshold"])
    above = run_simgrep_command(["diff", "tree_a", "tree_b", "--threshold", "1.5"], cwd=changed_trees)
    assert_failure_contains(above, ["--threshold"])


def test_diff_top_and_max_chunks_validation_rejected(temp_simgrep_home: pathlib.Path, changed_trees: pathlib.Path) -> None:
    top_zero = run_simgrep_command(["diff", "tree_a", "tree_b", "--top", "0"], cwd=changed_trees)
    assert top_zero.exit_code != 0
    chunks_zero = run_simgrep_command(["diff", "tree_a", "tree_b", "--max-chunks", "0"], cwd=changed_trees)
    assert chunks_zero.exit_code != 0


def test_diff_unknown_format_rejected(temp_simgrep_home: pathlib.Path, changed_trees: pathlib.Path) -> None:
    result = run_simgrep_command(["diff", "tree_a", "tree_b", "--format", "yaml"], cwd=changed_trees)
    assert_failure_contains(result, ["--format"])


def test_diff_machine_formats_stdout_clean(temp_simgrep_home: pathlib.Path, changed_trees: pathlib.Path) -> None:
    formats: Sequence[str] = ("json", "jsonl", "count")
    for fmt in formats:
        result = run_simgrep_command(["diff", "tree_a", "tree_b", "--format", fmt], cwd=changed_trees)
        assert_success(result)
        assert result.stderr == "", f"--format {fmt} leaked to stderr"


def test_diff_absolute_paths_flag(temp_simgrep_home: pathlib.Path, changed_trees: pathlib.Path) -> None:
    absolute = run_simgrep_command(
        ["diff", "tree_a", "tree_b", "--format", "json", "--absolute-paths"],
        cwd=changed_trees,
    )
    assert_success(absolute)
    payload = json.loads(absolute.stdout)
    all_paths = {entry["file_path"] for entry in [*payload["added"], *payload["removed"]]}
    assert all_paths
    for path in all_paths:
        assert pathlib.Path(path).is_absolute()
        assert path.startswith(str(changed_trees))
