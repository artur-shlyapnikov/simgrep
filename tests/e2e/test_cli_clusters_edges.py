"""Edge-contract E2E coverage for `simgrep clusters`: routing table, boundaries, config interplay."""

from __future__ import annotations

import hashlib
import json
import pathlib
import re

import numpy as np
import pytest

from tests.conftest import FakeTextExtractor, FakeTokenChunker, FakeVectorIndex
from tests.e2e.conftest import (
    assert_clean_json_list,
    assert_failure_contains,
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


_PAIR_B_BODY = (
    "def refresh_cache_shard(node):\n" "    blob = fetch_manifest(node)\n" "    entries = flatten_segment(blob)\n" "    return prune_stale_entries(entries)\n"
)


def _write_config(temp_simgrep_home: pathlib.Path, text: str) -> None:
    """Overwrite the global simgrep config.toml inside the temp home."""
    config_dir = temp_simgrep_home / ".config" / "simgrep"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.toml").write_text(text, encoding="utf-8")


def _plant(root: pathlib.Path, files: dict[str, str]) -> pathlib.Path:
    for name, body in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    return root


@pytest.fixture
def twopair_project(tmp_path: pathlib.Path) -> pathlib.Path:
    """Project with two disjoint duplicate pairs (p1/p2 and q1/q2), i.e. exactly two clusters."""
    project_dir = tmp_path / "twopair_project"
    project_dir.mkdir()
    _plant(project_dir, {"p1.py": _DUPLICATE_BODY, "p2.py": _DUPLICATE_BODY})
    _plant(project_dir, {"q1.py": _PAIR_B_BODY, "q2.py": _PAIR_B_BODY})
    assert_success(run_simgrep_command(["init"], cwd=project_dir))
    assert_success(run_simgrep_command(["index", "--rebuild"], cwd=project_dir))
    return project_dir


@pytest.fixture
def sized_pair_tree(tmp_path: pathlib.Path) -> pathlib.Path:
    """Ephemeral-scan tree with a 271-byte twin pair and a 472-byte twin pair."""
    tree = tmp_path / "sized_tree"
    tree.mkdir()
    big_body = _DUPLICATE_BODY + "x" * 200 + "\n"  # 472 bytes
    _plant(tree, {"small_a.py": _DUPLICATE_BODY, "small_b.py": _DUPLICATE_BODY})
    _plant(tree, {"big_a.py": big_body, "big_b.py": big_body})
    return tree


def test_clusters_threshold_point_nine_nine_nine_nine_still_binds_identical_chunks(temp_simgrep_home: pathlib.Path, duplicate_project: pathlib.Path) -> None:
    """Cosine of bitwise-identical float32-normalized rows is 0.99999976..., so a 0.9999
    threshold (probe margin ~1e-4) must still cluster the alpha/beta twins with score >= 0.99."""
    payload = assert_clean_json_list(run_simgrep_command(["clusters", "--threshold", "0.9999", "--format", "json"], cwd=duplicate_project))
    assert len(payload) == 1
    assert payload[0]["score"] >= 0.99
    assert {m["file_path"] for m in payload[0]["members"]} == {"alpha.py", "beta.py"}
    assert payload[0]["duplicated_lines"] == 14


def test_clusters_threshold_one_point_zero_accepted_but_finds_nothing(temp_simgrep_home: pathlib.Path, duplicate_project: pathlib.Path) -> None:
    """--threshold 1.0 passes CLI validation ((0.0, 1.0] domain) yet yields zero clusters:
    the measured identical-pair cosine 0.9999997615814209 falls short of exact 1.0 in float32."""
    result = run_simgrep_command(["clusters", "--threshold", "1.0", "--format", "count"], cwd=duplicate_project)
    assert_success(result)
    assert result.stdout.strip() == "0"
    assert result.stderr == ""


def test_clusters_threshold_zero_rejected_with_domain_message(temp_simgrep_home: pathlib.Path, duplicate_project: pathlib.Path) -> None:
    """The lower boundary 0.0 is outside the documented (0.0, 1.0] domain and names the rule."""
    assert_failure_contains(
        run_simgrep_command(["clusters", "--threshold", "0.0"], cwd=duplicate_project),
        ["--threshold must be greater than 0.0"],
    )


def test_clusters_top_cap_reports_precap_total_and_caps_display(temp_simgrep_home: pathlib.Path, twopair_project: pathlib.Path) -> None:
    """With two clusters and --top 1: count prints the PRE-cap total_found (2); rich says
    '(2 found, 1 shown)'; compact/json show exactly one cluster and never the shown-note."""
    count_capped = run_simgrep_command(["clusters", "--top", "1", "--format", "count"], cwd=twopair_project)
    assert_success(count_capped)
    assert count_capped.stdout.strip() == "2"

    rich_result = run_simgrep_command(["clusters", "--top", "1"], cwd=twopair_project)
    assert_success(rich_result)
    assert "Semantic Clusters (2 found, 1 shown)" in rich_result.stdout
    assert rich_result.stdout.count("score=") == 1
    member_lines = [line for line in rich_result.stdout.splitlines() if ".py" in line]
    assert len(member_lines) == 2

    compact_result = run_simgrep_command(["clusters", "--top", "1", "--format", "compact"], cwd=twopair_project)
    assert_success(compact_result)
    compact_lines = [line for line in compact_result.stdout.splitlines() if line.strip()]
    assert len(compact_lines) >= 1
    for line in compact_lines:
        assert re.match(r"^\[\d+\] score=\d\.\d{3}  \S+:\d+-\d+$", line)
        assert line.startswith("[1] ")
    assert "shown" not in compact_result.stdout

    count_uncapped = run_simgrep_command(["clusters", "--format", "count"], cwd=twopair_project)
    assert_success(count_uncapped)
    assert count_uncapped.stdout.strip() == "2"


def test_clusters_min_size_three_drops_pair_clusters_with_format_specific_empty_ux(temp_simgrep_home: pathlib.Path, duplicate_project: pathlib.Path) -> None:
    """Every planted cluster has 2 members, so --min-size 3 empties the outcome: rich prints its
    dedicated sentence while count/json degrade numerically and paths/jsonl go silent."""
    rich_result = run_simgrep_command(["clusters", "--min-size", "3"], cwd=duplicate_project)
    assert_success(rich_result)
    assert "No duplicate clusters found." in rich_result.stdout

    count_result = run_simgrep_command(["clusters", "--min-size", "3", "--format", "count"], cwd=duplicate_project)
    assert_success(count_result)
    assert count_result.stdout.strip() == "0"
    assert count_result.stderr == ""

    json_result = run_simgrep_command(["clusters", "--min-size", "3", "--format", "json"], cwd=duplicate_project)
    assert_success(json_result)
    assert json.loads(json_result.stdout) == []
    assert json_result.stderr == ""

    paths_result = run_simgrep_command(["clusters", "--min-size", "3", "--format", "paths"], cwd=duplicate_project)
    assert_success(paths_result)
    assert paths_result.stdout == ""
    assert paths_result.stderr == ""

    jsonl_result = run_simgrep_command(["clusters", "--min-size", "3", "--format", "jsonl"], cwd=duplicate_project)
    assert_success(jsonl_result)
    assert jsonl_result.stdout == ""
    assert jsonl_result.stderr == ""


def test_clusters_max_chunks_guard_fails_with_hint_and_clean_stdout(temp_simgrep_home: pathlib.Path, duplicate_project: pathlib.Path) -> None:
    """duplicate_project scans 3 chunks; --max-chunks 1 trips the engine guard: exit 1, the
    message names the counts, the hint names the flag, stdout stays empty."""
    result = run_simgrep_command(["clusters", "--max-chunks", "1"], cwd=duplicate_project)
    assert_failure_contains(result, ["Too many chunks to cluster (3 > 1)", "raise --max-chunks"])
    assert result.stdout == ""


def test_clusters_no_path_no_active_project_hints_init_or_path(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """Bare `clusters` in a directory with no .simgrep anywhere up-tree fails with the
    actionable hint instead of a traceback."""
    nowhere = tmp_path / "nowhere"
    nowhere.mkdir()
    result = run_simgrep_command(["clusters"], cwd=nowhere)
    assert_failure_contains(result, ["No active project found.", "Run `simgrep init` or pass a PATH for ephemeral clustering."])


def test_clusters_auto_ephemeral_fallback_for_path_outside_any_project(
    temp_simgrep_home: pathlib.Path,
    duplicate_project: pathlib.Path,
    twopair_project: pathlib.Path,
) -> None:
    """PATH pointing at a sibling directory that was never init'ed/indexed routes to the
    ephemeral branch automatically (no --ephemeral needed) and clusters its twins."""
    outside = duplicate_project.parent / "outside_dup"
    outside.mkdir()
    _plant(outside, {"dup_a.py": _DUPLICATE_BODY, "dup_b.py": _DUPLICATE_BODY})

    fallback_result = run_simgrep_command(["clusters", "--format", "paths", str(outside)], cwd=duplicate_project)
    fallback_lines = [line for line in fallback_result.stdout.splitlines() if line.strip()]
    assert_success(fallback_result)
    basenames = {pathlib.Path(line).name for line in fallback_lines}
    assert basenames == {"dup_a.py", "dup_b.py"}

    twopair_result = run_simgrep_command(["clusters", "--format", "paths", str(twopair_project)], cwd=duplicate_project)
    twopair_lines = [line for line in twopair_result.stdout.splitlines() if line.strip()]
    assert_success(twopair_result)
    assert twopair_lines == ["p1.py", "p2.py", "q1.py", "q2.py"]
    assert twopair_lines == sorted(set(twopair_lines))


def test_clusters_persistent_outside_any_project_rejected_with_message(
    temp_simgrep_home: pathlib.Path, duplicate_project: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """--persistent plus a PATH no project covers is refused up front (the ephemeral fallback
    must not silently win once persistence is demanded)."""
    outside = tmp_path / "outside_dup"
    outside.mkdir()
    _plant(outside, {"dup_a.py": _DUPLICATE_BODY, "dup_b.py": _DUPLICATE_BODY})
    result = run_simgrep_command(["clusters", "--persistent", str(outside)], cwd=duplicate_project)
    assert_failure_contains(result, ["Persistent clustering requires an active project covering the requested path."])


def test_clusters_persistent_flag_routes_through_project_index(temp_simgrep_home: pathlib.Path, duplicate_project: pathlib.Path) -> None:
    """Inside an initialized project, bare `clusters --persistent` takes the clusters_project
    branch and reports the indexed alpha/beta twins."""
    payload = assert_clean_json_list(run_simgrep_command(["clusters", "--persistent", "--format", "json"], cwd=duplicate_project))
    assert len(payload) == 1
    assert {m["file_path"] for m in payload[0]["members"]} == {"alpha.py", "beta.py"}
    assert payload[0]["score"] >= 0.99


def test_clusters_persistent_and_ephemeral_flags_are_mutually_exclusive(temp_simgrep_home: pathlib.Path, duplicate_project: pathlib.Path) -> None:
    """Combining the two routing flags is an argument error before any clustering work."""
    assert_failure_contains(
        run_simgrep_command(["clusters", "--persistent", "--ephemeral"], cwd=duplicate_project),
        ["--persistent and --ephemeral cannot be combined."],
    )


def test_clusters_ephemeral_scan_honors_config_file_patterns(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """Global config.toml file_patterns narrow which files the EPHEMERAL clusters scan sees;
    a .txt twin visible under defaults disappears under patterns = ["*.md"]."""
    tree = tmp_path / "pattern_tree"
    tree.mkdir()
    _plant(tree, {"one.md": _DUPLICATE_BODY, "two.md": _DUPLICATE_BODY, "three.txt": _DUPLICATE_BODY})

    _write_config(temp_simgrep_home, 'file_patterns = ["*.md"]\n')
    narrowed = run_simgrep_command(["clusters", "--format", "paths", str(tree)], cwd=tmp_path)
    narrowed_lines = [line for line in narrowed.stdout.splitlines() if line.strip()]
    assert_success(narrowed)
    assert {pathlib.Path(line).name for line in narrowed_lines} == {"one.md", "two.md"}

    _write_config(temp_simgrep_home, "")
    widened = run_simgrep_command(["clusters", "--format", "paths", str(tree)], cwd=tmp_path)
    widened_lines = [line for line in widened.stdout.splitlines() if line.strip()]
    assert_success(widened)
    assert {pathlib.Path(line).name for line in widened_lines} == {"one.md", "two.md", "three.txt"}


def test_clusters_ephemeral_scan_honors_config_max_file_size_bytes(temp_simgrep_home: pathlib.Path, sized_pair_tree: pathlib.Path) -> None:
    """A 350-byte ceiling drops the 472-byte twin pair but keeps the 271-byte pair; restoring
    the default config brings all four files back."""
    _write_config(temp_simgrep_home, "max_file_size_bytes = 350\n")
    capped = run_simgrep_command(["clusters", "--format", "paths", str(sized_pair_tree)], cwd=sized_pair_tree)
    capped_lines = [line for line in capped.stdout.splitlines() if line.strip()]
    assert_success(capped)
    assert {pathlib.Path(line).name for line in capped_lines} == {"small_a.py", "small_b.py"}

    _write_config(temp_simgrep_home, "")
    restored = run_simgrep_command(["clusters", "--format", "paths", str(sized_pair_tree)], cwd=sized_pair_tree)
    restored_lines = [line for line in restored.stdout.splitlines() if line.strip()]
    assert_success(restored)
    assert {pathlib.Path(line).name for line in restored_lines} == {
        "big_a.py",
        "big_b.py",
        "small_a.py",
        "small_b.py",
    }


def test_clusters_same_file_flag_accepted_without_disturbing_cross_file_pairs(temp_simgrep_home: pathlib.Path, duplicate_project: pathlib.Path) -> None:
    """--same-file widens pairing eligibility; with the one-chunk-per-file fake chunker no
    intra-file pair can exist, so the cross-file alpha/beta cluster must be unchanged."""
    payload = assert_clean_json_list(run_simgrep_command(["clusters", "--same-file", "--format", "json"], cwd=duplicate_project))
    assert len(payload) == 1
    assert {m["file_path"] for m in payload[0]["members"]} == {"alpha.py", "beta.py"}
    for member in payload[0]["members"]:
        assert member["line_start"] == 1
        assert member["line_end"] == 7


def test_clusters_compact_rendering_is_deterministic_and_format_token_case_insensitive(
    temp_simgrep_home: pathlib.Path, duplicate_project: pathlib.Path
) -> None:
    """Repeated compact runs are byte-identical, and the format token matches without regard
    to case ('COMPACT' equals 'compact')."""
    compact_a = run_simgrep_command(["clusters", "--format", "compact"], cwd=duplicate_project)
    compact_b = run_simgrep_command(["clusters", "--format", "compact"], cwd=duplicate_project)
    upper = run_simgrep_command(["clusters", "--format", "COMPACT"], cwd=duplicate_project)
    assert_success(compact_a)
    assert_success(compact_b)
    assert_success(upper)
    assert compact_a.stdout == compact_b.stdout == upper.stdout
    assert compact_a.stdout != ""
    assert "[1] score=" in compact_a.stdout
