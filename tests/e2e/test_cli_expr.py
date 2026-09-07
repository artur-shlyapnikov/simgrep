"""E2E coverage for `simgrep search --expr` (boolean semantic queries).

Uses a deterministic token-set hashing embedder (same pattern as
test_cli_diff.py): unit vector per token, summed and normalized. Files whose
token sets share tokens with an expression leaf score high on that leaf;
disjoint files sit near cosine zero. This makes boolean algebra observable:
AND excludes, OR unions, NOT inverts, quoted phrases act as single leaves.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re

import numpy as np
import pytest

from tests.conftest import FakeTextExtractor, FakeTokenChunker, FakeVectorIndex, VectorHit
from tests.e2e.conftest import (
    assert_failure_contains,
    assert_success,
    run_simgrep_command,
)


class TokenSetHashingEmbedder:
    """Deterministic fake embedder: unit vector per token, summed and normalized."""

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

    def search(self, vector: np.ndarray, k: int) -> list[VectorHit]:
        """True cosine ranking over stored vectors (base fake returns a constant)."""
        if not self.data:
            return []
        keys = self.keys
        matrix = self.vectors(keys)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        query = np.asarray(vector, dtype=np.float32).ravel()
        qnorm = float(np.linalg.norm(query))
        if qnorm > 0:
            query = query / qnorm
        sims = (matrix / norms) @ query
        order = np.argsort(-sims)[:k]
        return [VectorHit(label=int(keys[i]), score=float(sims[i])) for i in order]


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


_AUTH_BODY = "auth login user password\n" "auth credential grant ticket\n" "auth session principal login\n" "auth user password ticket\n"

_OAUTH_BODY = "oauth authorization code exchange\n" "oauth access token refresh\n" "oauth consent scope access token\n" "oauth bearer grant refresh\n"

_RETRY_BODY = "retry backoff transient failure\n" "retry exponential jitter attempt\n" "retry deadline reset retry\n" "retry attempt failure backoff\n"

_CACHE_BODY = "cache eviction lru memoize\n" "cache invalidate ttl stale\n" "cache entry hit miss cache\n" "cache ttl eviction stale\n"


@pytest.fixture
def indexed_repo(tmp_path: pathlib.Path, temp_simgrep_home: pathlib.Path) -> pathlib.Path:
    """Repo with auth/oauth/retry/cache files, initialized and indexed.

    The positional PATH slot belongs to `query` now, so expression searches
    target an indexed project via cwd instead of a path argument (a bare path
    with --expr is the pinned mutual-exclusion error).
    """
    del temp_simgrep_home
    repo = tmp_path / "expr-repo"
    repo.mkdir()
    (repo / "auth.py").write_text(_AUTH_BODY, encoding="utf-8")
    (repo / "oauth.py").write_text(_OAUTH_BODY, encoding="utf-8")
    (repo / "retry.py").write_text(_RETRY_BODY, encoding="utf-8")
    (repo / "cache.py").write_text(_CACHE_BODY, encoding="utf-8")
    assert_success(run_simgrep_command(["init"], cwd=repo))
    assert_success(run_simgrep_command(["index", "--rebuild"], cwd=repo))
    return repo


def _search_paths(repo: pathlib.Path, expr: str, *extra: str) -> list[str]:
    result = run_simgrep_command(["search", "--expr", expr, "--format", "paths", *extra], cwd=repo)
    assert_success(result)
    return [line for line in result.stdout.splitlines() if line.strip()]


def test_expr_and_not_excludes_negated_topic(indexed_repo: pathlib.Path) -> None:
    """`retry AND NOT cache` surfaces the retry file and never the cache file.

    A tight candidate pool (--candidates 2, --top 1 keeps k small) leaves the
    retry file absent from the negated leaf's dict, so it passes NOT at full
    credit while the cache file itself is inverted towards zero.
    """
    paths = _search_paths(indexed_repo, "retry AND NOT cache", "--top", "1", "--candidates", "2")
    assert paths == ["retry.py"]


def test_expr_or_unions_both_topics(indexed_repo: pathlib.Path) -> None:
    paths = _search_paths(indexed_repo, "retry OR oauth", "--top", "2", "--candidates", "4")
    assert sorted(paths) == ["oauth.py", "retry.py"]


def test_expr_quoted_phrase_is_single_leaf(indexed_repo: pathlib.Path) -> None:
    paths = _search_paths(indexed_repo, '"access token"', "--top", "1", "--candidates", "2")
    assert paths == ["oauth.py"]


def test_expr_implicit_and_adjacent_atoms(indexed_repo: pathlib.Path) -> None:
    paths = _search_paths(indexed_repo, '"access token" oauth', "--top", "1", "--candidates", "2")
    assert paths == ["oauth.py"]


def test_expr_query_and_expr_are_mutually_exclusive(tmp_path: pathlib.Path) -> None:
    repo = tmp_path / "mutex-repo"
    repo.mkdir()
    (repo / "notes.txt").write_text("placeholder text body\n", encoding="utf-8")
    result = run_simgrep_command(["search", "plain query", "--expr", "retry OR cache", str(repo)])
    assert_failure_contains(result, ["mutually exclusive"])


def test_expr_rejects_lexical_top_flag(indexed_repo: pathlib.Path) -> None:
    result = run_simgrep_command(["search", "--expr", "retry", "--lexical-top", "3"], cwd=indexed_repo)
    assert_failure_contains(result, ["lexical options are not supported with --expr"])


def test_expr_rejects_lexical_weight_flag(indexed_repo: pathlib.Path) -> None:
    result = run_simgrep_command(["search", "--expr", "retry", "--lexical-weight", "0.5"], cwd=indexed_repo)
    assert_failure_contains(result, ["lexical options are not supported with --expr"])


def test_expr_missing_query_errors_when_both_absent(indexed_repo: pathlib.Path) -> None:
    result = run_simgrep_command(["search"], cwd=indexed_repo)
    assert_failure_contains(result, ["query cannot be empty"])


def test_expr_json_format_shape_and_ranking(indexed_repo: pathlib.Path) -> None:
    result = run_simgrep_command(
        ["search", "--expr", "retry OR oauth", "--format", "json", "--top", "4", "--candidates", "4"],
        cwd=indexed_repo,
    )
    assert_success(result)
    payload = json.loads(result.stdout)
    assert isinstance(payload, list) and len(payload) == 4
    top_two = {record["path"] for record in payload[:2]}
    assert top_two == {"retry.py", "oauth.py"}
    for record in payload:
        assert {"path", "score", "text"} <= set(record)
        assert record["score"] is not None


def test_expr_count_format(indexed_repo: pathlib.Path) -> None:
    result = run_simgrep_command(["search", "--expr", "retry OR oauth", "--format", "count"], cwd=indexed_repo)
    assert_success(result)
    assert re.fullmatch(r"[1-9]\d*\n?", result.stdout)


def test_expr_empty_result_set_exits_zero(indexed_repo: pathlib.Path) -> None:
    """A contradiction under a high min-score yields zero matches, exit 0."""
    result = run_simgrep_command(
        ["search", "--expr", "retry AND cache", "--format", "count", "--min-score", "0.99"],
        cwd=indexed_repo,
    )
    assert_success(result)
    assert result.stdout.strip() == "0"


def test_expr_hybrid_flag_is_ignored(indexed_repo: pathlib.Path) -> None:
    """--no-hybrid must neither error nor change --expr's pure semantic path."""
    without = _search_paths(indexed_repo, "retry OR oauth", "--top", "2", "--no-hybrid")
    with_hybrid = _search_paths(indexed_repo, "retry OR oauth", "--top", "2")
    assert sorted(without) == sorted(with_hybrid) == ["oauth.py", "retry.py"]


def test_expr_lowercase_operators_are_plain_words(indexed_repo: pathlib.Path) -> None:
    """Lucene convention: lowercase `and`/`or`/`not` are ordinary leaf words."""
    result = run_simgrep_command(
        ["search", "--expr", "retry and cache", "--format", "count", "--top", "1", "--candidates", "2"],
        cwd=indexed_repo,
    )
    assert_success(result)
    # `and` is just another leaf word; no syntax error either way.
    assert re.fullmatch(r"\d+\n?", result.stdout)
