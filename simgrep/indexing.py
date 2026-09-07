from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Iterable, Iterator

import numpy as np
from filelock import FileLock

from simgrep.files import build_file_plan, build_project_file_plan, calculate_file_hash, classify_file, extract_chunk_terms, scan_files
from simgrep.models import (
    AppConfig,
    ChangeDetectionMode,
    Chunk,
    ChunkRecord,
    EphemeralIndexOptions,
    FilePlan,
    FilePlanEntry,
    FileRecord,
    FileRole,
    IndexOptions,
    IndexState,
    IndexStats,
    ProjectConfig,
    ScanOptions,
    TermRecord,
)
from simgrep.runtime import Runtime
from simgrep.store import Store
from simgrep.text import compute_line_starts, offset_to_line

_ProgressReporter = Any  # duck-typed: optional ``phase``/``tick`` methods
_active_reporter: _ProgressReporter | None = None


@contextmanager
def progress_scope(reporter: _ProgressReporter | None) -> Iterator[None]:
    """Install ``reporter`` as the process-wide index progress receiver."""
    global _active_reporter
    previous = _active_reporter
    _active_reporter = reporter
    try:
        yield
    finally:
        _active_reporter = previous


def _current_reporter() -> _ProgressReporter | None:
    return _active_reporter


def _report(kind: str, message: str) -> None:
    reporter = _active_reporter
    if reporter is None:
        return
    method = getattr(reporter, kind, None)
    if method is None:
        return
    try:
        method(message)
    except Exception:  # noqa: BLE001 - progress display must never fail an index run
        pass


def _phase(message: str) -> None:
    _report("phase", message)


def _tick(message: str) -> None:
    _report("tick", message)


if TYPE_CHECKING:
    from simgrep.corpus import CorpusReader


@dataclass(frozen=True)
class _PreparedFile:
    entry: FilePlanEntry
    text: str
    chunks: tuple[Chunk, ...]


def _require_bulk(runtime: Any) -> None:
    """Materialize eager runtime components (torch embedder, tokenizer chunker).

    Real Runtimes implement ``require_bulk``; duck-typed test fakes predate the
    seam and have nothing eager to materialize, so they are skipped.
    """
    require = getattr(runtime, "require_bulk", None)
    if require is not None:
        require()


def _load_on_disk_label_count(runtime: Any, project: ProjectConfig) -> int:
    index = runtime.new_vector_index(runtime.embedder.ndim)
    index.load(project.vector_index_path)
    return len(index)


class IndexEngine:
    def __init__(self, runtime: Runtime | Any) -> None:
        self.runtime = runtime

    def plan_project(self, project: ProjectConfig, app_config: AppConfig, options: IndexOptions | None = None) -> FilePlan:
        opts = options or IndexOptions()
        scan_options = self.scan_options(app_config, opts)
        existing: dict[Path, FileRecord] = {}
        if not opts.rebuild and project.metadata_db_path.exists():
            store = Store.open(project.metadata_db_path)
            try:
                existing = store.get_files()
            finally:
                store.close()
        return build_project_file_plan(project, existing, scan_options=scan_options, change_detection=opts.change_detection)

    def index_project(self, project: ProjectConfig, app_config: AppConfig, options: IndexOptions) -> IndexStats:
        t0 = time.perf_counter()
        stats = IndexStats()
        if options.dry_run and not project.metadata_db_path.exists():
            scan_options = self.scan_options(app_config, options)
            plan = build_project_file_plan(project, {}, scan_options=scan_options, change_detection=options.change_detection)
            self._copy_plan_counts(stats, plan)
            stats.files_indexed = plan.new_count + plan.changed_count
            stats.total_seconds = time.perf_counter() - t0
            return stats
        lock = FileLock(str(project.index_lock_path), is_singleton=True)
        with lock:
            store = Store.open(project.metadata_db_path)
            index: Any | None = None
            inserted_file_ids: list[int] = []
            try:
                rebuild = bool(options.rebuild)
                chunks_count = store.counts(project.name).chunks_count
                if (
                    not rebuild
                    and not options.dry_run
                    and chunks_count > 0
                    and (not project.vector_index_path.exists() or chunks_count > _load_on_disk_label_count(self.runtime, project))
                ):
                    # Divergence (crash recovery) must flip rebuild BEFORE planning:
                    # a rebuild plans against an empty store, forcing full re-encode.
                    rebuild = True
                scan_options = self.scan_options(app_config, options)
                t_plan = time.perf_counter()
                existing = {} if rebuild else store.get_files()
                plan = build_project_file_plan(project, existing, scan_options=scan_options, change_detection=options.change_detection)
                stats.plan_seconds = time.perf_counter() - t_plan
                self._copy_plan_counts(stats, plan)
                _phase(f"Scanned {stats.files_seen} file(s): {plan.new_count} new, {plan.changed_count} changed")
                if options.dry_run:
                    # Dry-run reports the plan; chunk counts are unknowable without the chunker.
                    stats.files_indexed = plan.new_count + plan.changed_count
                    stats.total_seconds = time.perf_counter() - t0
                    return stats
                has_work = plan.new_count + plan.changed_count > 0 or any(e.status == "deleted" for e in plan.entries)
                if not rebuild and not has_work:
                    # Nothing to index: leave torch unloaded for query-only flows.
                    if chunks_count == 0 and not project.vector_index_path.exists():
                        # Empty project: persist an empty vector index so search
                        # never finds a missing vector artifact.
                        index = self.runtime.new_vector_index(self.runtime.embedder.ndim)
                        self._atomic_save_index(index, project.vector_index_path)
                    # A lingering "indexing" flag is a crash artifact; the plan and
                    # the divergence probe above say the index is complete.
                    store.set_meta("index_state", IndexState.ready.value)
                    stats.total_seconds = time.perf_counter() - t0
                    return stats
                # Rebuild wipes everything, so swap the database file instead of
                # issuing DELETE scans. Equivalent end state (fresh schema, empty
                # tables, meta re-seeded by ensure_schema), and it avoids the
                # macOS nano-malloc heap-guard abort: torch's libomp corrupts
                # the malloc zone, and any large duckdb DELETE near it dies with
                if rebuild:
                    store.close()
                    project.metadata_db_path.unlink(missing_ok=True)
                    Path(str(project.metadata_db_path) + ".wal").unlink(missing_ok=True)
                    store = Store.open(project.metadata_db_path)
                # Bulk chunk+encode follows: materialize torch BEFORE the USearchIndex
                # so torch's libomp wins the process (OpenMP segfault guard).
                # Deletion-only plans never encode: keeping torch unloaded drops
                # the ~1.3s import from flows that have no use for it.
                if plan.new_count + plan.changed_count > 0:
                    _require_bulk(self.runtime)
                index = self.runtime.new_vector_index(self.runtime.embedder.ndim)
                if project.vector_index_path.exists() and not rebuild:
                    index.load(project.vector_index_path)
                store.set_meta("index_state", IndexState.indexing.value)
                for entry in plan.entries:
                    if entry.status == "deleted" and entry.existing_file_id is not None:
                        removed = store.delete_file(entry.existing_file_id)
                        if removed:
                            index.remove(np.array(removed, dtype=np.int64))
                            stats.vectors_removed += len(removed)
                            stats.index_mutated = True
                for entry in plan.entries:
                    if entry.status == "changed" and entry.existing_file_id is not None:
                        removed = store.delete_file(entry.existing_file_id)
                        if removed:
                            index.remove(np.array(removed, dtype=np.int64))
                            stats.vectors_removed += len(removed)
                            stats.index_mutated = True
                to_embed = [e for e in plan.entries if e.status in {"new", "changed"}]
                _phase(f"Indexing {len(to_embed)} file(s)")
                prepared_iter = self._prepare_entries(to_embed, max_workers=options.max_workers)
                self._flush_with_progress(prepared_iter, len(to_embed), store, index, app_config, stats, inserted_file_ids)
                if stats.index_mutated:
                    store.refresh_term_stats()
                if stats.index_mutated or rebuild or not project.vector_index_path.exists():
                    _phase("Saving index")
                    self._atomic_save_index(index, project.vector_index_path)
                store.set_meta("index_state", IndexState.ready.value)
                store.set_meta("last_index_completed_at", str(time.time()))
                store.set_meta("last_index_error", "")
                stats.total_seconds = time.perf_counter() - t0
                return stats
            except Exception as exc:
                for file_id in inserted_file_ids:
                    try:
                        store.delete_file(file_id)
                    except Exception:  # noqa: BLE001 - compensation must never mask the original failure
                        pass
                if not options.dry_run:
                    store.set_meta("index_state", IndexState.failed.value)
                    store.set_meta("last_index_error", str(exc))
                raise
            finally:
                store.close()

    def build_ephemeral(self, files: list[Path], app_config: AppConfig, options: EphemeralIndexOptions | None = None) -> "CorpusReader":
        # Bulk chunk+encode follows: materialize torch before the ephemeral index.
        _require_bulk(self.runtime)
        opts = options or EphemeralIndexOptions()
        scan_options = opts.scan
        discovered = []
        for path in files:
            discovered.extend(scan_files(path, scan_options))
        plan = build_file_plan(discovered, {}, options=scan_options, change_detection=ChangeDetectionMode.stat)
        store = Store.memory()
        index = self.runtime.new_vector_index(self.runtime.embedder.ndim)
        stats = IndexStats()
        try:
            inserted_file_ids: list[int] = []
            to_embed = [e for e in plan.entries if e.status == "new"]
            _phase(f"Indexing {len(to_embed)} file(s)")
            prepared = self._prepare_entries(to_embed, max_workers=opts.max_workers)
            self._flush_with_progress(prepared, len(to_embed), store, index, app_config, stats, inserted_file_ids)
            base = files[0].resolve() if len(files) == 1 else Path.cwd()
            if base.is_file():
                base = base.parent
            # Local import: the corpus boundary sits on top of indexing, never under it.
            from simgrep.corpus import CorpusReader

            return CorpusReader(store=store, index=index, base_path=base)
        except Exception:
            store.close()
            raise

    @staticmethod
    def scan_options(app_config: AppConfig, options: IndexOptions) -> ScanOptions:
        """Canonical scan configuration for one indexing pass (public: the
        daemon fingerprints the corpus through it instead of private hooks)."""
        patterns = options.patterns or app_config.file_patterns
        return ScanOptions(
            patterns=patterns,
            include_globs=options.include_globs,
            exclude_globs=options.exclude_globs,
            max_file_size_bytes=app_config.max_file_size_bytes,
            follow_symlinks=app_config.follow_symlinks,
        )

    @staticmethod
    def _copy_plan_counts(stats: IndexStats, plan: FilePlan) -> None:
        stats.files_seen = len([e for e in plan.entries if e.status != "deleted"])
        stats.files_skipped_unchanged = plan.unchanged_count
        stats.files_skipped_too_large = plan.too_large_count
        stats.ignored_count = plan.ignored_count
        stats.unreadable_count = plan.unreadable_count
        stats.files_pruned_deleted = plan.deleted_count

    def _prepare_entries(self, entries: list[FilePlanEntry], *, max_workers: int) -> Iterable[_PreparedFile]:
        """Yield prepared files lazily so extraction overlaps the encode pipeline.

        ``pool.map`` preserves input order, so label assignment matches the old
        materialized path exactly; the executor lives until the consumer drains
        (or abandons) this generator.
        """

        def _prepare(entry: FilePlanEntry) -> _PreparedFile:
            text = self.runtime.extractor.extract(entry.path)
            chunks = tuple(self._chunk_with_lines(text))
            return _PreparedFile(entry=entry, text=text, chunks=chunks)

        workers = max(1, int(max_workers))
        if workers <= 1 or len(entries) <= 1:
            for entry in entries:
                yield _prepare(entry)
            return
        with ThreadPoolExecutor(max_workers=workers) as pool:
            yield from pool.map(_prepare, entries)

    def _chunk_with_lines(self, text: str) -> list[Chunk]:
        chunks = self.runtime.chunker.chunk(text)
        if not chunks:
            return []
        starts = compute_line_starts(text)
        out: list[Chunk] = []
        for chunk in chunks:
            line_start = offset_to_line(starts, max(chunk.start, 0))
            line_end = offset_to_line(starts, max(chunk.end - 1, chunk.start))
            out.append(
                Chunk(
                    id=chunk.id,
                    file_id=chunk.file_id,
                    text=chunk.text,
                    start=chunk.start,
                    end=chunk.end,
                    tokens=chunk.tokens,
                    line_start=line_start,
                    line_end=line_end,
                )
            )
        return out

    def _flush_with_progress(
        self,
        prepared: Iterable[_PreparedFile],
        total: int,
        store: Store,
        index: Any,
        app_config: AppConfig,
        stats: IndexStats,
        inserted_file_ids: list[int],
    ) -> None:
        """Run :meth:`_flush_prepared`, ticking per embedded file."""

        if total <= 0:
            self._flush_prepared(prepared, store, index, app_config, stats, inserted_file_ids)
            return

        def tracked() -> Iterable[_PreparedFile]:
            for position, item in enumerate(prepared, start=1):
                yield item
                _tick(f"Embedding {position}/{total} file(s)")

        self._flush_prepared(tracked(), store, index, app_config, stats, inserted_file_ids)

    def _flush_prepared(
        self,
        prepared: Iterable[_PreparedFile],
        store: Store,
        index: Any,
        app_config: AppConfig,
        stats: IndexStats,
        inserted_file_ids: list[int],
    ) -> None:
        """Pipeline encode (GPU) against writes (DuckDB/usearch): while flush N+1
        encodes, flush N's store and index writes run on the writer thread.

        The writer owns EVERY store mutation for the duration of the pipeline, so
        the DuckDB connection is never used from two threads concurrently. Flushes
        are consumed FIFO, which keeps label assignment identical to the serial
        path. Failure semantics match the serial flow: on any failure the already
        written file ids land in ``inserted_file_ids`` for the caller's
        compensation pass, partial counters land in ``stats``, and the original
        error propagates.
        """
        flush_chunk_count = max(app_config.batch_size * 4, 256)
        jobs: queue.Queue[tuple[list[_PreparedFile], np.ndarray] | None] = queue.Queue(maxsize=2)
        abort = threading.Event()
        writer_errors: list[BaseException] = []
        writer_file_ids: list[int] = []
        written = {"chunks": 0, "vectors": 0, "mutated": False}

        def _writer() -> None:
            try:
                while True:
                    job = jobs.get()
                    if job is None or abort.is_set():
                        return
                    batch, vectors = job
                    try:
                        self._write_flush(batch, vectors, store, index, writer_file_ids, written)
                    except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread below
                        writer_errors.append(exc)
                        abort.set()
            except BaseException as exc:  # noqa: BLE001 - defensive: never die silently
                writer_errors.append(exc)
                abort.set()

        writer = threading.Thread(target=_writer, name="simgrep-index-writer", daemon=True)
        writer.start()
        pending: list[_PreparedFile] = []
        pending_count = 0
        try:
            for item in prepared:
                stats.files_processed += 1
                stats.files_indexed += 1
                # Zero-chunk files flow through so _write_flush records them;
                # otherwise plans re-report them as new on every freshness run.
                pending.append(item)
                pending_count += len(item.chunks)
                if pending_count >= flush_chunk_count:
                    jobs.put((pending, self.runtime.embedder.encode(self._flush_texts(pending), is_query=False, batch_size=app_config.batch_size)))
                    pending, pending_count = [], 0
            if pending:
                jobs.put((pending, self.runtime.embedder.encode(self._flush_texts(pending), is_query=False, batch_size=app_config.batch_size)))
        except BaseException:
            abort.set()
            jobs.put(None)
            writer.join()
            inserted_file_ids.extend(writer_file_ids)
            self._merge_written(stats, written)
            raise
        jobs.put(None)
        writer.join()
        inserted_file_ids.extend(writer_file_ids)
        self._merge_written(stats, written)
        if writer_errors:
            raise writer_errors[0]

    @staticmethod
    def _flush_texts(batch: list[_PreparedFile]) -> list[str]:
        return [chunk.text for item in batch for chunk in item.chunks]

    @staticmethod
    def _merge_written(stats: IndexStats, written: dict[str, Any]) -> None:
        stats.chunks_indexed += int(written["chunks"])
        stats.vectors_added += int(written["vectors"])
        if written["mutated"]:
            stats.index_mutated = True

    def _insert_prepared(
        self,
        prepared: list[_PreparedFile],
        store: Store,
        index: Any,
        app_config: AppConfig,
        stats: IndexStats,
        inserted_file_ids: list[int],
    ) -> None:
        chunk_texts = [chunk.text for item in prepared for chunk in item.chunks]
        if not chunk_texts:
            return
        vectors = self.runtime.embedder.encode(chunk_texts, is_query=False, batch_size=app_config.batch_size)
        written: dict[str, Any] = {"chunks": 0, "vectors": 0, "mutated": False}
        self._write_flush(prepared, vectors, store, index, inserted_file_ids, written)
        self._merge_written(stats, written)

    def _write_flush(
        self,
        prepared: list[_PreparedFile],
        vectors: np.ndarray,
        store: Store,
        index: Any,
        inserted_file_ids: list[int],
        written: dict[str, Any],
    ) -> None:
        """Persist one encoded flush: reserve labels, upsert files, write chunks,
        terms, and vectors. Runs on the pipeline writer thread; ``written``
        accumulates counters for the caller to merge into ``stats``."""
        # Record every processed file (even chunk-less ones) so later plans see
        # them as unchanged instead of re-reporting them as new forever.
        features_by_item: list[Any] = []
        file_records: list[FileRecord] = []
        for item in prepared:
            try:
                path_hash = item.entry.new_hash if item.entry.new_hash is not None else calculate_file_hash(item.entry.path)
            except OSError:
                path_hash = None
            features = classify_file(item.entry.path)
            features_by_item.append(features)
            if item.entry.size_bytes and item.entry.mtime_ns:
                size_bytes, mtime_ns = item.entry.size_bytes, item.entry.mtime_ns
            else:
                # One stat call: the previous code stated the path twice.
                file_stat = item.entry.path.stat()
                size_bytes = item.entry.size_bytes or file_stat.st_size
                mtime_ns = item.entry.mtime_ns or file_stat.st_mtime_ns
            file_records.append(
                FileRecord(
                    id=0,
                    path=item.entry.path,
                    size_bytes=size_bytes,
                    mtime_ns=mtime_ns,
                    sha256=path_hash,
                    role=features.file_role,
                    language=features.language,
                    is_test=features.is_test,
                    is_generated=features.is_generated,
                )
            )
        # One batched upsert (single MAX + chunked multi-row INSERT) instead of
        # three round-trips per file.
        file_ids = store.upsert_files(file_records)
        inserted_file_ids.extend(file_ids)
        chunk_count = sum(len(item.chunks) for item in prepared)
        if chunk_count == 0:
            return
        labels = store.reserve_labels(chunk_count)
        chunk_records: list[ChunkRecord] = []
        term_records: list[TermRecord] = []
        label_offset = 0
        for item, features, file_id in zip(prepared, features_by_item, file_ids):
            kind = "text" if features.file_role in {FileRole.docs, FileRole.config, FileRole.dependency_metadata, FileRole.build_metadata} else "mixed"
            for chunk in item.chunks:
                label = labels[label_offset]
                label_offset += 1
                chunk_records.append(
                    ChunkRecord(
                        label=label,
                        file_id=file_id,
                        text=chunk.text,
                        start_char=chunk.start,
                        end_char=chunk.end,
                        token_count=chunk.tokens,
                        line_start=chunk.line_start,
                        line_end=chunk.line_end,
                        kind=kind,
                    )
                )
                for term, field, tf, weight in extract_chunk_terms(chunk.text, item.entry.path, features.language):
                    term_records.append(TermRecord(label=label, term=term, field=field, tf=tf, weight=weight))
        store.insert_chunks(chunk_records)
        store.insert_terms(term_records)
        index.add(np.array(labels, dtype=np.int64), np.asarray(vectors, dtype=np.float32))
        written["chunks"] += len(chunk_records)
        written["vectors"] += len(chunk_records)
        written["mutated"] = written["mutated"] or bool(chunk_records)

    @staticmethod
    def _atomic_save_index(index: Any, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = destination.parent / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(prefix="vectors-", suffix=".tmp", dir=tmp_dir, delete=False) as handle:
            tmp_path = Path(handle.name)
        try:
            index.save(tmp_path)
            tmp_path.replace(destination)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
