"""Unit tests for the `--expr` boolean semantic query branch of SearchEngine.

Fakes are deliberately scripted: the embedder maps a fixed vocabulary onto
orthogonal unit directions, and the vector index returns cosine similarity for
every stored chunk above a relevance floor. That makes leaf scores predictable
so AND (elementwise min), OR (max) and NOT (1 - s) semantics can be asserted
end-to-end through `search_path` / `search_project`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from simgrep.errors import SimgrepError
from simgrep.models import SCHEMA_VERSION, AppConfig, DiversityMode, FreshnessMode, ProjectConfig, SearchOptions
from simgrep.search import SearchEngine
from tests.conftest import FakeTextExtractor, FakeTokenChunker

VOCAB = ("auth", "login", "oauth", "retry", "backoff", "token", "cache", "connection pool")
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

    def add(self, labels: np.ndarray | None = None, vectors: np.ndarray | None = None, *, keys: np.ndarray | None = None) -> None:
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

    def save(self, path: Path) -> None:
        import json

        payload = {str(label): self.data[label].tolist() for label in sorted(self.data)}
        path.write_text(json.dumps(payload), encoding="utf-8")

    def load(self, path: Path) -> None:
        import json

        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.data = {int(label): np.asarray(vector, dtype=np.float32) for label, vector in payload.items()}

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


FILES = {
    "auth.py": "handles auth login flow",
    "oauth.py": "oauth token refresh",
    "retry.py": "retry with backoff",
    "mixed.py": "auth login plus retry backoff",
}


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


def _paths(results: list[Any]) -> list[str]:
    return [result.file_path.name for result in results]


# --- AND excludes -----------------------------------------------------------------


def test_and_excludes_partial_matches(tmp_path: Path) -> None:
    engine = SearchEngine(_seed_corpus(tmp_path, FILES))
    # Chunks whose combined score is exactly 0.0 (a conjunct leaf misses them
    # entirely) are dropped before ranking; only mixed.py carries both topics.
    outcome = engine.search_path(tmp_path, AppConfig(model="fake"), _options("auth AND retry", min_score=0.7))
    assert _paths(outcome.results) == ["mixed.py"]
    # candidate pairs keep only nonzero combined scores: just mixed.py
    assert outcome.semantic_candidates == 1


# --- OR unions --------------------------------------------------------------------


def test_or_unions_leaf_candidate_sets(tmp_path: Path) -> None:
    engine = SearchEngine(_seed_corpus(tmp_path, FILES))
    outcome = engine.search_path(tmp_path, AppConfig(model="fake"), _options("auth OR retry"))
    names = set(_paths(outcome.results))
    assert names == {"auth.py", "retry.py", "mixed.py"}
    # OR keeps the stronger leaf per label: single-topic files outrank the blend.
    assert outcome.results[0].file_path.name in {"auth.py", "retry.py"}


def test_quoted_phrase_is_single_leaf(tmp_path: Path) -> None:
    files = {
        "pool.txt": "the connection pool is exhausted",
        "decoy.txt": "pool cleanup duty",
        "other.txt": "unrelated prose about caching",
    }
    engine = SearchEngine(_seed_corpus(tmp_path, files))
    outcome = engine.search_path(tmp_path, AppConfig(model="fake"), _options('"connection pool"'))
    # The phrase matches as one atom; the decoy lacks the full phrase.
    assert _paths(outcome.results) == ["pool.txt"]


# --- NOT excludes via dominance; mere relatedness is kept --------------------------


def test_not_excludes_docs_dominated_by_negated_leaf(tmp_path: Path) -> None:
    engine = SearchEngine(_seed_corpus(tmp_path, FILES))
    outcome = engine.search_path(tmp_path, AppConfig(model="fake"), _options("retry AND NOT oauth", min_score=0.0))
    names = _paths(outcome.results)
    # Dominance NOT: oauth.py is more similar to the negated leaf than to every
    # positive leaf, so its combined score is exactly 0.0 -> dropped from the
    # results even at min_score 0. auth.py never surfaces under the retry leaf,
    # so the AND zeroes it out and the pipeline drops it too.
    assert names == ["retry.py", "mixed.py"]
    by_name = {result.file_path.name: result for result in outcome.results}
    assert by_name["retry.py"].score == pytest.approx(1.0)
    # mixed.py is absent from the oauth hit set, so NOT grants full credit and
    # only its retry similarity ranks it below the pure hit.
    assert by_name["mixed.py"].score == pytest.approx(0.945, abs=1e-3)
    assert outcome.semantic_candidates == 2


def test_not_keeps_mere_relatedness_below_true_match(tmp_path: Path) -> None:
    files = {
        "exact.py": "pure login implementation",
        "related.py": "auth login token refresh",
        "oauth.py": "oauth token refresh",
    }
    engine = SearchEngine(_seed_corpus(tmp_path, files))
    outcome = engine.search_path(
        tmp_path,
        AppConfig(model="fake"),
        _options("(auth OR login) AND NOT oauth", min_score=0.0),
    )
    names = _paths(outcome.results)
    # Regression guard from the real-model smoke: related.py shares the `token`
    # direction with the negated oauth leaf (cos ~0.41), but its positive
    # similarity (~0.58) dominates, so NOT grants full credit instead of the old
    # fuzzy 1 - s penalty that dragged it to ~0.70 below weaker pure hits.
    assert names == ["exact.py", "related.py"]
    by_name = {result.file_path.name: result for result in outcome.results}
    assert by_name["related.py"].score == pytest.approx(0.9659, abs=1e-3)
    # The dominated doc is literally absent, not surfaced at a neutral floor.
    assert "oauth.py" not in names


def test_pure_not_falls_back_to_fuzzy_inversion(tmp_path: Path) -> None:
    files = {
        "pure.py": "oauth handling notes",
        "diluted.py": "oauth cache notes",
    }
    engine = SearchEngine(_seed_corpus(tmp_path, files))
    outcome = engine.search_path(tmp_path, AppConfig(model="fake"), _options("NOT oauth"))
    names = _paths(outcome.results)
    # No positive leaves -> fuzzy fallback 1 - s over the negated leaf's own hit
    # set: the diluted oauth overlap outranks the exact one, and the exact hit
    # inverts to exactly 0.0 so the zero-drop filter excludes it entirely.
    assert names == ["diluted.py"]
    assert outcome.results[0].score == pytest.approx(0.6191, abs=1e-3)
    assert "pure.py" not in names


def test_not_exclusion_propagates_to_whole_file(tmp_path: Path) -> None:
    files = {
        "straddle.py": "auth login flow\n\noauth token refresh\n",
        "pure.py": "pure login notes",
    }
    runtime = _seed_corpus(tmp_path, files)

    from simgrep.models import Chunk

    class ParagraphChunker:
        def chunk(self, text: str) -> list[Any]:
            parts = [part for part in text.split("\n\n") if part.strip()]
            return [Chunk(id=-1, file_id=-1, text=part, start=0, end=len(part), tokens=max(1, len(part.split()))) for part in parts]

    runtime.chunker = ParagraphChunker()
    engine = SearchEngine(runtime)
    outcome = engine.search_path(
        tmp_path,
        AppConfig(model="fake"),
        _options("(auth OR login) AND NOT oauth", min_score=0.0),
    )
    names = _paths(outcome.results)
    # The oauth chunk of straddle.py is dominance-zeroed, which excludes the
    # WHOLE file: its auth/login sibling straddles (positive similarity beats
    # the negated leaf, so chunk-level zero-drop alone would keep it) but must
    # not resurface the excluded concept.
    assert names == ["pure.py"]
    assert outcome.semantic_candidates == 1
    by_name = {result.file_path.name: result for result in outcome.results}
    assert by_name["pure.py"].score == pytest.approx(1.0)


# --- min_score filter ---------------------------------------------------------------


def test_min_score_filters_combined_scores(tmp_path: Path) -> None:
    engine = SearchEngine(_seed_corpus(tmp_path, FILES))
    outcome = engine.search_path(
        tmp_path,
        AppConfig(model="fake"),
        _options("auth OR retry OR oauth", min_score=0.97),
    )
    # Single-topic files cap at exactly 1.0; every multi-topic blend lands lower
    # (~0.94) because each topic direction is diluted. The high bar keeps only pure hits.
    assert set(_paths(outcome.results)) == {"auth.py", "retry.py", "oauth.py"}


# --- top / diversity ----------------------------------------------------------------


def test_top_caps_result_count(tmp_path: Path) -> None:
    engine = SearchEngine(_seed_corpus(tmp_path, FILES))
    outcome = engine.search_path(tmp_path, AppConfig(model="fake"), _options("auth OR retry OR oauth", top=2))
    assert len(outcome.results) == 2


def test_diversity_file_mode_limits_one_per_file(tmp_path: Path) -> None:
    files = {
        "big.py": "auth login\n\nmore auth login notes\n\neven more auth login",
        "small.py": "auth login here too",
    }
    runtime = _seed_corpus(tmp_path, files)

    from simgrep.models import Chunk

    class ParagraphChunker:
        def chunk(self, text: str) -> list[Any]:
            paragraphs = [part for part in text.split("\n\n") if part.strip()]
            return [Chunk(id=-1, file_id=-1, text=part, start=0, end=len(part), tokens=max(1, len(part.split()))) for part in paragraphs]

    runtime.chunker = ParagraphChunker()
    engine = SearchEngine(runtime)
    outcome = engine.search_path(
        tmp_path,
        AppConfig(model="fake"),
        _options("auth", diversity=DiversityMode.file, top=10),
    )
    names = _paths(outcome.results)
    assert names.count("big.py") == 1
    assert "small.py" in names


# --- ephemeral + persistent paths -----------------------------------------------------


def test_ephemeral_outcome_shape(tmp_path: Path) -> None:
    engine = SearchEngine(_seed_corpus(tmp_path, FILES))
    outcome = engine.search_path(tmp_path, AppConfig(model="fake"), _options("auth AND retry"))
    assert outcome.base_path == tmp_path
    assert outcome.files_seen == 4
    assert outcome.chunks_searched == 4


def test_persistent_project_path(tmp_path: Path) -> None:
    project = ProjectConfig(SCHEMA_VERSION, "p", tmp_path, (tmp_path,), "fake", 128, 20)
    app_config = AppConfig(model="fake")
    engine = SearchEngine(_seed_corpus(tmp_path, FILES))
    outcome = engine.search_project(project, app_config, _options("auth AND retry", min_score=0.7), FreshnessMode.auto)
    assert _paths(outcome.results) == ["mixed.py"]


# --- empty index ----------------------------------------------------------------------


def test_empty_index_returns_empty_outcome_without_error(tmp_path: Path) -> None:
    empty_files: dict[str, str] = {}
    engine = SearchEngine(_seed_corpus(tmp_path, empty_files))
    outcome = engine.search_path(tmp_path, AppConfig(model="fake"), _options("auth AND retry"))
    assert outcome.results == []
    assert outcome.semantic_candidates == 0


# --- parse errors bubble as SimgrepError ----------------------------------------------


def test_invalid_expr_raises_simgrep_error(tmp_path: Path) -> None:
    engine = SearchEngine(_seed_corpus(tmp_path, FILES))
    with pytest.raises(SimgrepError):
        engine.search_path(tmp_path, AppConfig(model="fake"), _options("AND auth"))
