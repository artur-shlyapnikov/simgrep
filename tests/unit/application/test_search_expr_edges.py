"""Edge-case tests for the `--expr` boolean semantic query branch of SearchEngine.

Replicates the shipped scripted stack locally (the shipped fixtures are
module-private and cross-test imports are fragile): unit-vector vocabulary
encoding plus a cosine index with a relevance floor give closed-form leaf
scores, so per-leaf top-k cuts, `min_score` boundaries, empty universes,
single-chunk universes, top=0 and cross-engine determinism can be asserted
end-to-end through `search_path`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from simgrep.models import AppConfig, DiversityMode, SearchOptions
from simgrep.search import SearchEngine
from tests.conftest import FakeTextExtractor, FakeTokenChunker

VOCAB = ("auth", "login", "oauth", "retry", "backoff", "cache")
RELEVANCE_FLOOR = 0.3


def _unit(dim: int, index: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    vec[index] = 1.0
    return vec


class ScriptedEmbedder:
    """Encodes text as the normalized sum of unit vectors for vocabulary hits."""

    def __init__(self) -> None:
        self.ndim = len(VOCAB)

    def encode(self, texts: list[str], *, is_query: bool = False, batch_size: int | None = None) -> np.ndarray:
        del is_query, batch_size
        out = np.zeros((len(texts), self.ndim), dtype=np.float32)
        for row, text in enumerate(texts):
            lowered = f" {text.lower()} "
            acc = np.zeros(self.ndim, dtype=np.float32)
            for index, entry in enumerate(VOCAB):
                if f" {entry} " in lowered:
                    acc = acc + _unit(self.ndim, index)
            norm = float(np.linalg.norm(acc))
            if norm > 0:
                out[row] = acc / norm
        return out


class ScriptedVectorIndex:
    """Cosine-similarity index that only surfaces hits above RELEVANCE_FLOOR."""

    def __init__(self, ndim: int) -> None:
        self.ndim = ndim
        self.data: dict[int, np.ndarray] = {}

    def add(
        self,
        labels: np.ndarray | None = None,
        vectors: np.ndarray | None = None,
        *,
        keys: np.ndarray | None = None,
    ) -> None:
        actual_labels = labels if labels is not None else keys
        assert actual_labels is not None and vectors is not None
        flat = np.asarray(actual_labels, dtype=np.int64).reshape(-1)
        rows = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
        for label, vector in zip(flat, rows):
            self.data[int(label)] = vector

    def remove(self, labels: np.ndarray | None = None, *, keys: np.ndarray | None = None) -> None:
        actual = labels if labels is not None else keys
        if actual is not None:
            for label in actual:
                self.data.pop(int(label), None)

    def search(self, vector: np.ndarray, k: int) -> list[Any]:
        from simgrep.models import VectorHit

        query = np.asarray(vector, dtype=np.float32)
        scored: list[tuple[float, int]] = []
        for label, stored in self.data.items():
            denom = float(np.linalg.norm(query) * np.linalg.norm(stored))
            similarity = 0.0 if denom == 0.0 else float(np.dot(query, stored) / denom)
            if similarity >= RELEVANCE_FLOOR:
                scored.append((similarity, label))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [VectorHit(label=label, score=similarity) for similarity, label in scored[:k]]

    @property
    def keys(self) -> np.ndarray:
        return np.array(sorted(self.data), dtype=np.int64)

    def vectors(self, keys: np.ndarray | None = None) -> np.ndarray:
        actual_keys = self.keys if keys is None else np.asarray(keys, dtype=np.int64)
        rows = [self.data[int(label)] for label in actual_keys]
        if not rows:
            return np.zeros((0, self.ndim), dtype=np.float32)
        return np.stack(rows).astype(np.float32, copy=False)


class ScriptedRuntime:
    def __init__(self) -> None:
        self.embedder = ScriptedEmbedder()
        self.extractor: Any = FakeTextExtractor()
        self.chunker: Any = FakeTokenChunker()

    def new_vector_index(self, ndim: int) -> ScriptedVectorIndex:
        return ScriptedVectorIndex(ndim)


def _seed_corpus(tmp_path: Path, files: dict[str, str]) -> ScriptedRuntime:
    """Materialize the corpus on disk (the scanner walks the real tree)."""
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return ScriptedRuntime()


def _options(expr: str, **overrides: Any) -> SearchOptions:
    defaults: dict[str, Any] = {
        "query": expr,
        "expr": expr,
        "lexical_top": 0,
        "lexical_weight": 0.0,
        "min_score": 0.05,
        "diversity": DiversityMode.none,
    }
    defaults.update(overrides)
    return SearchOptions(**defaults)


# --- A1 per-leaf top-k cut before fusion -------------------------------------------------

FILES = {
    "pure_auth.py": "auth",
    "mid_auth.py": "auth login",
    "diluted_auth.py": "auth login cache oauth",
    "back_one.py": "backoff",
    "back_two.py": "backoff retry",
}


def test_per_leaf_top_k_cut_removes_or_candidates_before_fusion(tmp_path: Path) -> None:
    """Each leaf fetches its own top-k BEFORE evaluate; a chunk ranked below k for its
    ONLY matching leaf is fused as absent even though OR semantics would admit it."""
    engine = SearchEngine(_seed_corpus(tmp_path, FILES))
    ctrl = engine.search_path(tmp_path, AppConfig(model="fake"), _options("auth OR backoff", top=2))
    cut = engine.search_path(tmp_path, AppConfig(model="fake"), _options("auth OR backoff", top=2, candidate_top=2))
    assert ctrl.semantic_candidates == 5
    assert cut.semantic_candidates == 4  # diluted_auth.py cut from auth's top-2
    # and it is genuinely the per-leaf cut: with a wide budget it surfaces via auth alone
    wide = engine.search_path(tmp_path, AppConfig(model="fake"), _options("auth", top=5))
    assert [r.file_path.name for r in wide.results][:3] == ["pure_auth.py", "mid_auth.py", "diluted_auth.py"]


# --- A2 inclusive min_score boundary on the combined score -------------------------------


def test_min_score_boundary_on_combined_score_is_inclusive(tmp_path: Path) -> None:
    """rank_candidates keeps final_score == min_score (>=) and drops the next float up."""
    files = {"a_one.py": "auth", "a_two.py": "auth login"}
    engine = SearchEngine(_seed_corpus(tmp_path, files))
    baseline = engine.search_path(tmp_path, AppConfig(model="fake"), _options("auth", min_score=0.0))
    names = [r.file_path.name for r in baseline.results]
    assert set(names) == {"a_one.py", "a_two.py"}
    top_score = baseline.results[0].score
    kept = engine.search_path(tmp_path, AppConfig(model="fake"), _options("auth", min_score=top_score))
    assert [r.file_path.name for r in kept.results] == names  # == survives
    dropped = engine.search_path(
        tmp_path,
        AppConfig(model="fake"),
        _options("auth", min_score=math.nextafter(top_score, math.inf)),
    )
    assert dropped.results == []  # one ulp above empties


# --- A3 unscored leaf collapses the universe ----------------------------------------------


def test_unscored_leaf_yields_empty_universe_even_under_not(tmp_path: Path) -> None:
    """Non-empty index but a leaf with zero hits: universe collapses to {} — even pure-NOT
    returns nothing (no labels to invert); a positive AND NOT unscored keeps everything."""
    files = {"auth.py": "auth login"}
    engine = SearchEngine(_seed_corpus(tmp_path, files))
    miss = engine.search_path(tmp_path, AppConfig(model="fake"), _options("cache"))
    assert miss.results == [] and miss.semantic_candidates == 0 and miss.chunks_searched == 1
    pure_not = engine.search_path(tmp_path, AppConfig(model="fake"), _options("NOT cache"))
    assert pure_not.results == [] and pure_not.semantic_candidates == 0
    guarded = engine.search_path(tmp_path, AppConfig(model="fake"), _options("auth AND NOT cache"))
    assert [r.file_path.name for r in guarded.results] == ["auth.py"]


# --- A4 single-chunk universe with unscored negation ---------------------------------------


def test_single_chunk_universe_with_unscored_negation(tmp_path: Path) -> None:
    """Single-candidate universe: NOT over a leaf with an empty dict never dominates,
    so the lone chunk survives an OR with the negation at full score."""
    engine = SearchEngine(_seed_corpus(tmp_path, {"only.py": "auth"}))
    outcome = engine.search_path(tmp_path, AppConfig(model="fake"), _options("auth OR NOT oauth"))
    assert [r.file_path.name for r in outcome.results] == ["only.py"]
    assert outcome.results[0].score == pytest.approx(1.0)
    assert outcome.semantic_candidates == 1


# --- A5 top=0 short-circuits results but still counts candidates ----------------------------


def test_top_zero_returns_no_results_but_counts_candidates(tmp_path: Path) -> None:
    """options.top == 0 short-circuits rank_candidates to []; the expr branch still
    reports how many fused candidates existed."""
    engine = SearchEngine(_seed_corpus(tmp_path, {"auth.py": "auth login"}))
    outcome = engine.search_path(tmp_path, AppConfig(model="fake"), _options("auth", top=0))
    assert outcome.results == []
    assert outcome.semantic_candidates == 1


# --- A6 identical scores order deterministically across engines -----------------------------


def test_identical_scores_order_deterministically_across_engines(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Equal combined scores fall back to (-score, label) ordering; two independent
    engines built over identical corpora produce byte-identical result orders."""
    files = {"x1.py": "auth login", "x2.py": "auth login", "x3.py": "auth login"}
    orders = []
    for tmp_path in tmp_path_factory.mktemp("tie_a"), tmp_path_factory.mktemp("tie_b"):
        engine = SearchEngine(_seed_corpus(tmp_path, files))
        outcome = engine.search_path(tmp_path, AppConfig(model="fake"), _options("auth"))
        orders.append([r.file_path.name for r in outcome.results])
    assert orders[0] == orders[1] == ["x1.py", "x2.py", "x3.py"]
