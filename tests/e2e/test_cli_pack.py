"""E2E coverage for `simgrep pack` (budgeted context assembly).

Self-sufficient mirror of tests/e2e/test_cli_related.py: designed temp repo +
token-set hashing fake runtime patched onto ``simgrep.main.RuntimeFactory``
directly (no shared conftest fixtures). Covers payload shapes, exit codes
0/1/2, budget-never-exceeded, citations, and determinism.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
from typing import Sequence

import numpy as np
import pytest
from typer.testing import CliRunner, Result

from simgrep.main import app
from simgrep.models import Chunk

try:
    runner = CliRunner(mix_stderr=False)  # type: ignore[call-arg]
except TypeError:
    runner = CliRunner()


def run_simgrep_command(args: Sequence[str]) -> Result:
    """In-process CLI invocation with a wide terminal for stable wrapping."""
    return runner.invoke(app, list(args), env={"COLUMNS": "200"})


class TokenSetHashingEmbedder:
    """Unit vector per token, summed and normalized: shared vocab => ~0.9."""

    ndim = 256

    def encode(self, texts: list[str], *, is_query: bool = False, batch_size: int | None = None) -> np.ndarray:
        del is_query, batch_size
        vectors = np.zeros((len(texts), self.ndim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                digest = hashlib.md5(token.encode("utf-8")).digest()
                rng = np.random.default_rng(int.from_bytes(digest[:4], "little"))
                vector = rng.standard_normal(self.ndim).astype(np.float32)
                vectors[row] += vector / float(np.linalg.norm(vector))
            norm = float(np.linalg.norm(vectors[row]))
            if norm > 0:
                vectors[row] /= norm
        return vectors


class FakeTextExtractor:
    def extract(self, path: pathlib.Path) -> str:
        sample = path.read_bytes()[:8192]
        if b"\x00" in sample:
            return ""
        return path.read_text(encoding="utf-8")


class WholeTextChunker:
    def chunk(self, text: str) -> Sequence[Chunk]:
        if not text.strip():
            return []
        return [Chunk(id=-1, file_id=-1, text=text, start=0, end=len(text), tokens=max(1, len(text.split())))]


class RoundTripVectorIndex:
    def __init__(self, ndim: int = 256) -> None:
        self.ndim = ndim
        self.data: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.data)

    def add(
        self,
        labels: np.ndarray | None = None,
        vectors: np.ndarray | None = None,
        *,
        keys: np.ndarray | None = None,
        vecs: np.ndarray | None = None,
    ) -> None:
        actual_labels = labels if labels is not None else keys
        actual_vectors = vectors if vectors is not None else vecs
        assert actual_labels is not None
        assert actual_vectors is not None
        for label, vector in zip(actual_labels, actual_vectors):
            self.data[int(label)] = np.asarray(vector, dtype=np.float32)

    def remove(self, labels: np.ndarray | None = None, *, keys: np.ndarray | None = None) -> None:
        actual = labels if labels is not None else keys
        if actual is None:
            return
        for label in np.asarray(actual).tolist():
            self.data.pop(int(label), None)

    def search(self, vector: np.ndarray, k: int) -> list[object]:
        del vector, k
        return []

    def save(self, path: pathlib.Path) -> None:
        payload = {str(label): vector.tolist() for label, vector in sorted(self.data.items())}
        path.write_text(json.dumps(payload), encoding="utf-8")

    def load(self, path: pathlib.Path) -> None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.data = {int(label): np.asarray(vector, dtype=np.float32) for label, vector in raw.items()}

    @property
    def keys(self) -> np.ndarray:
        return np.array(sorted(self.data), dtype=np.int64)

    def vectors(self, keys: np.ndarray | None = None) -> np.ndarray:
        actual_keys = self.keys if keys is None else np.asarray(keys, dtype=np.int64)
        rows = [self.data[int(key)] for key in actual_keys.tolist()]
        return np.stack(rows).astype(np.float32, copy=False)


class _HashingRuntime:
    def __init__(self) -> None:
        self.extractor = FakeTextExtractor()
        self.chunker = WholeTextChunker()
        self.embedder = TokenSetHashingEmbedder()

    def require_bulk(self) -> None:
        """No-op: the token-set hashing embedder is already bulk-friendly."""

    def new_vector_index(self, ndim: int) -> RoundTripVectorIndex:
        return RoundTripVectorIndex(ndim)


@pytest.fixture(autouse=True)
def hashing_runtime_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch RuntimeFactory on simgrep.main directly (self-sufficient, foreign-safe)."""
    runtime = _HashingRuntime()

    class _Factory:
        def for_app(self, config: object) -> _HashingRuntime:
            del config
            return runtime

        def for_project(self, config: object) -> _HashingRuntime:
            del config
            return runtime

    monkeypatch.setattr("simgrep.execution.RuntimeFactory", _Factory)


_PAYMENT_BODY = "payment gateway charges the customer card\n" "refund flow reverses a settled payment charge\n" "invoice billing total includes tax payment\n"
_REFUND_BODY = (
    "refund gateway reverses the customer charge\n" "payment refund edge cases mock timeout retries\n" "settlement batch reconciles invoice billing totals\n"
)
_LEDGER_BODY = "quantum blockchain ledger consensus\n" "distributed ledger nodes validate blocks\n"


@pytest.fixture
def pack_repo(tmp_path: pathlib.Path) -> pathlib.Path:
    repo = tmp_path / "proj"
    (repo / "src").mkdir(parents=True)
    (repo / ".git").mkdir()
    (repo / "src" / "payment.py").write_text(_PAYMENT_BODY, encoding="utf-8")
    (repo / "src" / "refund.py").write_text(_REFUND_BODY, encoding="utf-8")
    (repo / "src" / "ledger.py").write_text(_LEDGER_BODY, encoding="utf-8")
    return repo


_LONG_BODY = "payment gateway charges the customer card for the settled invoice amount\n" * 20


@pytest.fixture
def fat_repo(tmp_path: pathlib.Path) -> pathlib.Path:
    """Every chunk far exceeds the minimum 100-token budget when chars/4 estimated."""
    repo = tmp_path / "fat"
    (repo / "src").mkdir(parents=True)
    (repo / ".git").mkdir()
    (repo / "src" / "payment.py").write_text(_LONG_BODY, encoding="utf-8")
    (repo / "src" / "refund.py").write_text(_LONG_BODY.replace("card", "card refund"), encoding="utf-8")
    return repo


def _run(repo: pathlib.Path, *extra: str) -> Result:
    return run_simgrep_command(["pack", "gateway payment", "refund charge", str(repo), "--ephemeral", *extra])


def test_rich_lists_selections_and_budget_line(pack_repo: pathlib.Path) -> None:
    result = _run(pack_repo)
    assert result.exit_code == 0
    assert re.search(r"src/\w+\.py:\d+-\d+ \(score=0\.\d{3}", result.stdout)
    assert re.search(r"packaged \d+/\d+ tokens, \d+ of \d+ chunks", result.stdout)


def test_json_payload_matches_pinned_shape(pack_repo: pathlib.Path) -> None:
    result = _run(pack_repo, "--format", "json")
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert set(payload) == {"queries", "budget_tokens", "used_tokens", "pool_size", "dropped", "selections"}
    assert payload["queries"] == ["gateway payment", "refund charge"]
    assert set(payload["selections"][0]) == {"path", "line_start", "line_end", "score", "tokens", "truncated", "text"}
    assert payload["used_tokens"] <= payload["budget_tokens"]
    assert payload["dropped"] == payload["pool_size"] - len(payload["selections"])
    for selection in payload["selections"]:
        assert selection["path"].startswith("src/")
        assert isinstance(selection["line_start"], int)
        assert isinstance(selection["line_end"], int)


def test_label_dedup_collapses_shared_hits_across_queries(pack_repo: pathlib.Path) -> None:
    result = _run(pack_repo, "--format", "json")
    payload = json.loads(result.stdout)
    # Both queries retrieve the two shared-vocabulary files; without
    # label-dedup the union would double-count them.
    paths = [selection["path"] for selection in payload["selections"]]
    allowed = {"src/payment.py", "src/refund.py", "src/ledger.py"}
    assert set(paths) <= allowed
    assert len(paths) == len(set(paths))
    assert payload["pool_size"] == len(paths)
    assert {"src/payment.py", "src/refund.py"} <= set(paths)


def test_budget_never_exceeded_across_sizes(pack_repo: pathlib.Path) -> None:
    for budget in (100, 150, 1000, 200000):
        payload = json.loads(_run(pack_repo, "--format", "json", "--budget", str(budget)).stdout)
        assert payload["used_tokens"] <= budget
        assert payload["budget_tokens"] == budget


def test_oversized_fallback_truncates_single_best(fat_repo: pathlib.Path) -> None:
    result = _run(fat_repo, "--format", "json", "--budget", "100")
    payload = json.loads(result.stdout)
    assert len(payload["selections"]) == 1
    selection = payload["selections"][0]
    assert selection["truncated"] is True
    assert selection["tokens"] == 100
    assert selection["text"].endswith("…")
    assert payload["used_tokens"] == 100
    assert payload["dropped"] == payload["pool_size"] - 1


def test_markdown_renders_citation_headers_and_separators(pack_repo: pathlib.Path) -> None:
    result = _run(pack_repo, "--format", "markdown")
    assert result.exit_code == 0
    assert re.search(r"### src/payment\.py:\d+-\d+ \(score=0\.\d+, ~\d+ tok\)", result.stdout)
    assert "```" in result.stdout
    assert "---" in result.stdout
    assert re.search(r"packaged \d+/\d+ tokens, \d+ of \d+ chunks", result.stdout)


def test_deterministic_output_across_runs(pack_repo: pathlib.Path) -> None:
    first = _run(pack_repo, "--format", "json").stdout
    second = _run(pack_repo, "--format", "json").stdout
    assert first == second


def test_empty_corpus_exits_one(tmp_path: pathlib.Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / ".git").mkdir()
    (empty / "data.bin").write_bytes(b"\x00\x01binary")
    result = run_simgrep_command(["pack", "anything", str(empty), "--ephemeral"])
    assert result.exit_code == 1


def test_missing_queries_exits_two() -> None:
    result = run_simgrep_command(["pack"])
    assert result.exit_code == 2


def test_budget_below_bounds_exits_two(pack_repo: pathlib.Path) -> None:
    result = run_simgrep_command(["pack", "q", str(pack_repo), "--ephemeral", "--budget", "50"])
    assert result.exit_code == 2


def test_budget_above_bounds_exits_two(pack_repo: pathlib.Path) -> None:
    result = run_simgrep_command(["pack", "q", str(pack_repo), "--ephemeral", "--budget", "300000"])
    assert result.exit_code == 2


def test_per_query_out_of_bounds_exits_two(pack_repo: pathlib.Path) -> None:
    low = run_simgrep_command(["pack", "q", str(pack_repo), "--ephemeral", "--per-query", "0"])
    high = run_simgrep_command(["pack", "q", str(pack_repo), "--ephemeral", "--per-query", "51"])
    assert low.exit_code == 2
    assert high.exit_code == 2


def test_lam_out_of_bounds_exits_two(pack_repo: pathlib.Path) -> None:
    result = run_simgrep_command(["pack", "q", str(pack_repo), "--ephemeral", "--lam", "1.5"])
    assert result.exit_code == 2


def test_persistent_ephemeral_conflict_exits_two(pack_repo: pathlib.Path) -> None:
    result = run_simgrep_command(["pack", "q", str(pack_repo), "--persistent", "--ephemeral"])
    assert result.exit_code == 2


def test_invalid_format_exits_two(pack_repo: pathlib.Path) -> None:
    result = run_simgrep_command(["pack", "q", str(pack_repo), "--ephemeral", "--format", "bogus"])
    assert result.exit_code == 2


def test_lam_extremes_accepted(pack_repo: pathlib.Path) -> None:
    pure_diversity = _run(pack_repo, "--format", "json", "--lam", "0.0")
    pure_relevance = _run(pack_repo, "--format", "json", "--lam", "1.0")
    assert pure_diversity.exit_code == 0
    assert pure_relevance.exit_code == 0


def test_cwd_without_project_is_typed_error_like_similar(pack_repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No TARGET_DIR and no active project -> typed SearchError, mirroring `similar`."""
    monkeypatch.chdir(pack_repo)  # repo exists but was never `simgrep init`ed
    result = run_simgrep_command(["pack", "gateway payment", "--ephemeral", "--format", "json"])
    assert result.exit_code == 1
    assert "No active project found." in (result.stderr or "")


def test_no_active_project_is_typed_error_not_crash(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fresh dir, no TARGET_DIR, no project: typed SearchError per sibling convention."""
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    monkeypatch.chdir(fresh)
    result = run_simgrep_command(["pack", "anything"])
    assert result.exit_code == 1
    assert "AssertionError" not in (result.stderr or "")
    assert "No active project found." in (result.stderr or "")
