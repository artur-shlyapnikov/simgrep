"""Deep corpus access boundary: one logical dataset from DuckDB metadata + USearch vectors.

The corpus layer owns everything callers previously had to reconstruct themselves:

- label/vector/metadata alignment (:class:`CorpusReader.snapshot` returns an already-aligned
  :class:`ChunkBatch`; missing-metadata labels are dropped defensively, never misaligned),
- lifecycle and cleanup (readers are context managers; stores close, ephemeral corpora vanish),
- freshness sequencing and the project's singleton index lock (:meth:`CorpusAccess.open_project`
  holds the lock for the whole session, so an indexing writer can never mutate artifacts under
  an active reader),
- persistent-vs-ephemeral mode selection and scan-option reconstruction
  (:func:`ephemeral_options`).

``Store``, ``IndexEngine`` and ``SearchEngine`` remain implementation pieces underneath this
boundary; analysis engines consume ``StoredChunk``/``ChunkBatch`` and never touch storage.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from filelock import FileLock

from simgrep.errors import SearchError
from simgrep.models import (
    AppConfig,
    EphemeralIndexOptions,
    FileRole,
    FreshnessMode,
    ProjectConfig,
    ProjectStatus,
    ResultFilters,
    ScanOptions,
    VectorHit,
)
from simgrep.store import Store

__all__ = [
    "ChunkBatch",
    "CorpusAccess",
    "CorpusReader",
    "StoredChunk",
    "ephemeral_options",
]


@dataclass(frozen=True)
class StoredChunk:
    """Typed view of one indexed chunk; no storage column names or aliases."""

    label: int
    file_id: int
    file_path: Path
    text: str
    start_char: int
    end_char: int
    token_count: int
    line_start: int | None
    line_end: int | None
    role: FileRole
    language: str


@dataclass(frozen=True)
class ChunkBatch:
    """Chunks aligned row-for-row with their embedding vectors.

    ``vectors[i]`` is the embedding of ``chunks[i]``. ``indexed_count`` reports how many
    labels the vector index held before metadata alignment, so callers can distinguish
    "empty index" from "vectors whose metadata rows went missing".
    """

    chunks: tuple[StoredChunk, ...]
    vectors: np.ndarray
    indexed_count: int

    @property
    def labels(self) -> tuple[int, ...]:
        return tuple(chunk.label for chunk in self.chunks)

    def __len__(self) -> int:
        return len(self.chunks)


def _role_of(raw: Any) -> FileRole:
    try:
        return FileRole(str(raw))
    except ValueError:
        return FileRole.unknown


def _chunk_from_row(row: dict[str, Any]) -> StoredChunk:
    return StoredChunk(
        label=int(row["label"]),
        file_id=int(row["file_id"]),
        file_path=Path(row["file_path"]),
        text=str(row["chunk_text"]),
        start_char=int(row["start_char"]),
        end_char=int(row["end_char"]),
        token_count=int(row["token_count"]),
        line_start=None if row.get("line_start") is None else int(row["line_start"]),
        line_end=None if row.get("line_end") is None else int(row["line_end"]),
        role=_role_of(row["file_role"]),
        language=str(row["language"]),
    )


def ephemeral_options(
    app_config: AppConfig,
    *,
    patterns: tuple[str, ...] | None = None,
    include_globs: tuple[str, ...] = (),
    exclude_globs: tuple[str, ...] = (),
    max_workers: int = 4,
) -> EphemeralIndexOptions:
    """Scan settings for a one-shot corpus, defaulting to the app's patterns."""
    return EphemeralIndexOptions(
        scan=ScanOptions(
            patterns=patterns or app_config.file_patterns,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
            max_file_size_bytes=app_config.max_file_size_bytes,
            follow_symlinks=app_config.follow_symlinks,
        ),
        max_workers=max_workers,
    )


class CorpusReader:
    """Read-side view over one opened corpus (persistent project or ephemeral build).

    Exposes typed, already-aligned data. Raw store/index objects stay private; the only
    escape hatches are vector search hits and lexical candidates, which are returned in
    domain terms as well.
    """

    def __init__(self, store: Any, index: Any, base_path: Path) -> None:
        self._store = store
        self._index = index
        self._base_path = Path(base_path)

    @property
    def base_path(self) -> Path:
        return self._base_path

    @property
    def chunk_count(self) -> int:
        """Labels currently resident in the vector index."""
        return len(self._index)

    @property
    def ndim(self) -> int:
        return int(getattr(self._index, "ndim", 1))

    def snapshot(self, filters: ResultFilters | None = None) -> ChunkBatch:
        """Every indexed chunk aligned with its vector; missing metadata rows dropped."""
        keys = [int(value) for value in np.asarray(self._index.keys)]
        empty = ChunkBatch(chunks=(), vectors=np.zeros((0, self.ndim), dtype=np.float32), indexed_count=0)
        if not keys:
            return empty
        vectors = np.asarray(self._index.vectors(np.asarray(keys, dtype=np.int64)), dtype=np.float32)
        by_label = {int(row["label"]): row for row in self._store.lookup_chunks(keys, filters or ResultFilters())}
        chunks: list[StoredChunk] = []
        aligned: list[np.ndarray] = []
        for position, key in enumerate(keys):
            row = by_label.get(key)
            if row is None:
                continue
            chunks.append(_chunk_from_row(row))
            aligned.append(vectors[position])
        if not chunks:
            return ChunkBatch(chunks=(), vectors=np.zeros((0, self.ndim), dtype=np.float32), indexed_count=len(keys))
        matrix = np.stack(aligned).astype(np.float32, copy=False)
        return ChunkBatch(chunks=tuple(chunks), vectors=matrix, indexed_count=len(keys))

    def lookup(self, labels: Sequence[int], filters: ResultFilters | None = None) -> list[StoredChunk]:
        """Typed lookup preserving the caller's label rank order (missing labels dropped)."""
        rows = self._store.lookup_chunks([int(label) for label in labels], filters or ResultFilters())
        return [_chunk_from_row(row) for row in rows]

    def search(self, vector: np.ndarray, k: int) -> list[VectorHit]:
        """kNN against the corpus vectors."""
        hits: list[VectorHit] = self._index.search(vector, k)
        return hits

    def lexical(self, query_terms: list[str], limit: int, filters: ResultFilters | None = None) -> list[tuple[StoredChunk, float]]:
        rows = self._store.lexical_candidates(query_terms, limit, filters or ResultFilters())
        return [(_chunk_from_row(row), float(row.get("lexical_score", 0.0))) for row in rows]

    def counts(self, project_name: str = "") -> ProjectStatus:
        status: ProjectStatus = self._store.counts(project_name)
        return status

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> "CorpusReader":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class CorpusAccess:
    """Opens corpora with lifecycle, locking and freshness owned here."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        from simgrep.indexing import IndexEngine

        self._indexing = IndexEngine(runtime)

    @contextmanager
    def open_project(
        self,
        project: ProjectConfig,
        app_config: AppConfig,
        *,
        freshness: FreshnessMode = FreshnessMode.skip,
    ) -> Iterator[CorpusReader]:
        """Hold the project's singleton index lock for the whole session.

        Freshness runs first under the lock (so it can repair/rebuild safely), then the
        read-only reader opens; writers are excluded until the session ends.
        """
        # Local import keeps indexing <-> corpus acyclic at module level.
        lock = FileLock(str(project.index_lock_path), is_singleton=True)
        with lock:
            self._ensure_ready(project, app_config, freshness)
            with _open_persistent_reader(self.runtime, project) as reader:
                yield reader

    @contextmanager
    def open_ephemeral(
        self,
        paths: Sequence[Path],
        app_config: AppConfig,
        options: EphemeralIndexOptions | None = None,
    ) -> Iterator[CorpusReader]:
        """Build a throwaway corpus over ``paths`` and clean it up afterwards."""
        reader = self._indexing.build_ephemeral(list(paths), app_config, options or ephemeral_options(app_config))
        try:
            yield reader
        finally:
            reader.close()

    def _ensure_ready(self, project: ProjectConfig, app_config: AppConfig, freshness: FreshnessMode) -> None:
        """Guarantee fresh, complete artifacts before readers open (caller holds the lock)."""
        db_exists = project.metadata_db_path.exists()
        vector_exists = project.vector_index_path.exists()
        if freshness == FreshnessMode.auto and (not db_exists or not vector_exists):
            self._indexing.index_project(project, app_config, self._index_options(rebuild=True))
            return
        if not db_exists and not vector_exists:
            raise SearchError("Persistent index not found.", hint="Run `simgrep index` first.")
        if not db_exists:
            raise SearchError(f"Persistent database not found: {project.metadata_db_path}", hint="Run `simgrep index --rebuild`.")
        if not vector_exists:
            store = Store.open(project.metadata_db_path, read_only=True)
            try:
                counts = store.counts(project.name)
            finally:
                store.close()
            if counts.chunks_count > 0:
                raise SearchError(f"Vector index not found: {project.vector_index_path}", hint="Run `simgrep index --rebuild`.")
            return
        store = Store.open(project.metadata_db_path)
        try:
            state = store.get_meta("index_state")
            if state == "indexing":
                # We hold the singleton index lock, so no builder can be alive:
                # a lingering "indexing" flag is a crash artifact. Repair instead of failing.
                self._indexing.index_project(project, app_config, self._index_options())
                return
            if state == "failed":
                raise SearchError("Last indexing run failed.", hint=store.get_meta("last_index_error") or "Run `simgrep index --rebuild`.")
            if freshness == FreshnessMode.skip:
                return
            plan = self._indexing.plan_project(project, app_config, self._index_options())
            if plan.has_mutations:
                if freshness == FreshnessMode.check:
                    raise SearchError("Index is stale.", hint="Run `simgrep index` or use `--freshness auto`.")
                self._indexing.index_project(project, app_config, self._index_options())
        finally:
            store.close()

    @staticmethod
    def _index_options(rebuild: bool = False) -> Any:
        from simgrep.models import IndexOptions

        return IndexOptions(rebuild=rebuild)


@contextmanager
def _open_persistent_reader(runtime: Any, project: ProjectConfig) -> Iterator[CorpusReader]:
    store = Store.open(project.metadata_db_path, read_only=True)
    index = runtime.new_vector_index(runtime.embedder.ndim)
    try:
        if not project.vector_index_path.exists():
            counts = store.counts(project.name)
            if counts.chunks_count == 0:
                yield CorpusReader(store, index, project.root)
                return
            raise SearchError(f"Vector index not found: {project.vector_index_path}", hint="Run `simgrep index --rebuild`.")
        index.load(project.vector_index_path)
        yield CorpusReader(store, index, project.root)
    finally:
        store.close()
