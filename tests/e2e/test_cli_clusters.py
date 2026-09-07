"""E2E coverage for `simgrep clusters` (CLI surface, output contracts, validation)."""

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
    assert_clean_json_list,
    assert_clean_jsonl,
    assert_failure_contains,
    assert_paths_only,
    assert_success,
    run_simgrep_command,
)


class TokenSetHashingEmbedder:
    """Deterministic fake embedder: unit vector per token, summed and normalized.

    Identical texts collide at cosine 1.0; texts with disjoint token sets are
    near-orthogonal. Unlike the length-based FakeEmbedder, unrelated long texts
    stay dissimilar.
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
    """FakeVectorIndex whose save/load actually preserves vectors.

    The base class fills ``np.ones`` on load, which would collapse every chunk
    to cosine 1.0 after the persistent index snapshot reload.
    """

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


_DUPLICATE_BODY = (
    "def transfer_funds(account_src, account_dst, amount):\n"
    "    ledger = open_ledger_connection()\n"
    "    if amount > account_src.balance:\n"
    "        raise InsufficientFundsError(amount)\n"
    "    ledger.debit(account_src, amount)\n"
    "    ledger.credit(account_dst, amount)\n"
    "    ledger.commit()\n"
)

_DISTINCT_BODY = (
    "def parse_weather_report(payload):\n"
    "    stations = decode_station_table(payload)\n"
    "    readings = [harvest_barometer(entry) for entry in stations]\n"
    "    return summarize_meteorology(readings)\n"
)


@pytest.fixture
def duplicate_project(tmp_path: pathlib.Path) -> pathlib.Path:
    """Project with a planted cross-file duplicate pair and a distinct third file."""
    project_dir = tmp_path / "dup_project"
    project_dir.mkdir()
    (project_dir / "alpha.py").write_text(_DUPLICATE_BODY, encoding="utf-8")
    (project_dir / "beta.py").write_text(_DUPLICATE_BODY, encoding="utf-8")
    (project_dir / "gamma.py").write_text(_DISTINCT_BODY, encoding="utf-8")
    assert_success(run_simgrep_command(["init"], cwd=project_dir))
    assert_success(run_simgrep_command(["index", "--rebuild"], cwd=project_dir))
    return project_dir


def test_clusters_finds_duplicate_pair_and_excludes_distinct_file(temp_simgrep_home: pathlib.Path, duplicate_project: pathlib.Path) -> None:
    result = run_simgrep_command(["clusters", "--format", "paths"], cwd=duplicate_project)
    lines = assert_paths_only(result)
    basenames = {pathlib.Path(line).name for line in lines}
    assert basenames == {"alpha.py", "beta.py"}


def test_clusters_json_matches_pinned_schema(temp_simgrep_home: pathlib.Path, duplicate_project: pathlib.Path) -> None:
    payload = assert_clean_json_list(run_simgrep_command(["clusters", "--format", "json"], cwd=duplicate_project))
    assert len(payload) >= 1
    cluster = payload[0]
    assert set(cluster.keys()) == {"score", "duplicated_lines", "members"}
    assert isinstance(cluster["score"], float)
    assert isinstance(cluster["duplicated_lines"], int)
    members = cluster["members"]
    assert len(members) >= 2
    for member in members:
        assert set(member.keys()) == {"label", "file_path", "line_start", "line_end"}
        assert isinstance(member["label"], int)
        assert isinstance(member["file_path"], str)
        assert isinstance(member["line_start"], int)
        assert isinstance(member["line_end"], int)
        assert member["line_end"] >= member["line_start"] >= 1


def test_clusters_jsonl_one_object_per_line(temp_simgrep_home: pathlib.Path, duplicate_project: pathlib.Path) -> None:
    rows = assert_clean_jsonl(run_simgrep_command(["clusters", "--format", "jsonl"], cwd=duplicate_project))
    json_rows = assert_clean_json_list(run_simgrep_command(["clusters", "--format", "json"], cwd=duplicate_project))
    assert rows == json_rows
    assert len(rows) >= 1


def test_clusters_count_prints_total_found(temp_simgrep_home: pathlib.Path, duplicate_project: pathlib.Path) -> None:
    count_result = run_simgrep_command(["clusters", "--format", "count"], cwd=duplicate_project)
    assert_success(count_result)
    assert count_result.stderr == ""
    payload = assert_clean_json_list(run_simgrep_command(["clusters", "--format", "json"], cwd=duplicate_project))
    # default --top 20 is not reached, so shown == total_found
    assert count_result.stdout.strip() == str(len(payload))
    assert int(count_result.stdout.strip()) >= 1


def test_clusters_rich_output_shape(temp_simgrep_home: pathlib.Path, duplicate_project: pathlib.Path) -> None:
    result = run_simgrep_command(["clusters"], cwd=duplicate_project)
    assert_success(result)
    assert "Semantic Clusters (" in result.stdout
    assert "found" in result.stdout
    assert "duplicated lines" in result.stdout
    assert "score=" in result.stdout
    assert "gamma" not in result.stdout


def test_clusters_compact_prefixes_cluster_index_and_score(temp_simgrep_home: pathlib.Path, duplicate_project: pathlib.Path) -> None:
    result = run_simgrep_command(["clusters", "--format", "compact"], cwd=duplicate_project)
    assert_success(result)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines, "expected compact member lines"
    for line in lines:
        assert re.match(r"^\[\d+\] score=\d\.\d{3}  \S+:\d+-\d+$", line)


def test_clusters_absolute_paths_flag(temp_simgrep_home: pathlib.Path, duplicate_project: pathlib.Path) -> None:
    result = run_simgrep_command(["clusters", "--format", "paths", "--absolute-paths"], cwd=duplicate_project)
    lines = assert_paths_only(result)
    assert lines, "expected absolute paths"
    for line in lines:
        assert pathlib.Path(line).is_absolute()


def test_clusters_format_grep_rejected_cleanly(temp_simgrep_home: pathlib.Path, duplicate_project: pathlib.Path) -> None:
    assert_failure_contains(
        run_simgrep_command(["clusters", "--format", "grep"], cwd=duplicate_project),
        ["--format must be one of", "jsonl"],
    )


def test_clusters_threshold_above_one_rejected(temp_simgrep_home: pathlib.Path, duplicate_project: pathlib.Path) -> None:
    assert_failure_contains(
        run_simgrep_command(["clusters", "--threshold", "1.5"], cwd=duplicate_project),
        ["--threshold"],
    )


def test_clusters_min_size_below_two_rejected(temp_simgrep_home: pathlib.Path, duplicate_project: pathlib.Path) -> None:
    result = run_simgrep_command(["clusters", "--min-size", "1"], cwd=duplicate_project)
    assert result.exit_code != 0


def test_clusters_machine_formats_stdout_clean(temp_simgrep_home: pathlib.Path, duplicate_project: pathlib.Path) -> None:
    formats: Sequence[str] = ("json", "jsonl", "count", "paths")
    for fmt in formats:
        result = run_simgrep_command(["clusters", "--format", fmt], cwd=duplicate_project)
        assert_success(result)
        assert result.stderr == "", f"--format {fmt} leaked to stderr"


def test_clusters_ephemeral_on_plain_temp_dir(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    plain_dir = tmp_path / "plain_no_init"
    plain_dir.mkdir()
    (plain_dir / "one.py").write_text(_DUPLICATE_BODY, encoding="utf-8")
    (plain_dir / "two.py").write_text(_DUPLICATE_BODY, encoding="utf-8")
    result = run_simgrep_command(["clusters", str(plain_dir)], cwd=temp_simgrep_home)
    assert_success(result)
    assert "Semantic Clusters (" in result.stdout


def test_clusters_ephemeral_options_flow(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """Caller options must reach ClustersEngine on the ephemeral branch too."""
    plain_dir = tmp_path / "plain_opts"
    plain_dir.mkdir()
    (plain_dir / "one.py").write_text(_DUPLICATE_BODY, encoding="utf-8")
    (plain_dir / "two.py").write_text(_DUPLICATE_BODY, encoding="utf-8")
    default_result = run_simgrep_command(["clusters", "--ephemeral", str(plain_dir)], cwd=plain_dir)
    narrowed = run_simgrep_command(["clusters", "--ephemeral", str(plain_dir), "--min-size", "50"], cwd=plain_dir)
    assert_success(default_result)
    assert re.search(r"Semantic Clusters \([1-9]\d* found", default_result.stdout)
    assert_success(narrowed)
    assert "No duplicate clusters found." in narrowed.stdout
