"""Unit tests for the warm search daemon (simgrep/daemon.py)."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import tempfile
import threading
import time
from itertools import count
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from simgrep import daemon
from simgrep.models import (
    DiversityMode,
    FileRole,
    FreshnessMode,
    LexicalFallbackMode,
    PathBoost,
    ResultFilters,
    SearchOptions,
    SearchOutcome,
    SearchResult,
)


def _full_options() -> SearchOptions:
    return SearchOptions(
        query="duckdb wal recovery",
        top=7,
        min_score=0.5,
        candidate_top=11,
        lexical_top=13,
        lexical_weight=0.3,
        diversity=DiversityMode.file,
        scope_path=Path("/tmp/sg"),
        file_filter=("a.py",),
        keyword_filter="kw",
        include_globs=("src/**",),
        exclude_globs=("docs/**",),
        path_boosts=(PathBoost("tests/**", 0.4),),
        lexical_fallback=LexicalFallbackMode.off,
        expr="python AND NOT test",
    )


_SOCKET_SEQ = count()


def _short_socket() -> Path:
    """Unix sockets need short paths; pytest tmp_path nests too deeply."""
    return Path(tempfile.gettempdir()) / f"simgrep-daemon-test-{os.getpid()}-{next(_SOCKET_SEQ)}.sock"


def test_serialize_options_roundtrip() -> None:
    options = _full_options()
    restored = daemon._deserialize_options(daemon._serialize_options(options))
    assert restored == options


def test_serialize_outcome_roundtrip() -> None:
    outcome = SearchOutcome(
        results=[
            SearchResult(
                label=42,
                score=0.875,
                file_path=Path("/tmp/sg/a.py"),
                chunk_text="text",
                start_char=3,
                end_char=9,
                line_start=1,
                line_end=2,
                file_role=FileRole.test,
                language="python",
                why={"semantic": 0.9},
            )
        ],
        base_path=Path("/tmp/sg"),
        files_seen=3,
        chunks_searched=30,
        semantic_candidates=30,
    )
    restored = daemon._deserialize_outcome(daemon._serialize_outcome(outcome))
    assert restored == outcome


def test_socket_path_is_deterministic_per_db(tmp_path: Path) -> None:
    db = tmp_path / ".simgrep" / "metadata.duckdb"
    first = daemon.socket_path_for(db)
    second = daemon.socket_path_for(db)
    other = daemon.socket_path_for(tmp_path / "other.duckdb")
    assert first == second
    assert first != other


def _fake_project(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(metadata_db_path=tmp_path / ".simgrep" / "metadata.duckdb", root=tmp_path)


def test_try_search_skipped_without_auto_freshness(tmp_path: Path) -> None:
    project = _fake_project(tmp_path)
    options = SearchOptions(query="q")
    assert daemon.try_search(project, None, options, FreshnessMode.skip) is None
    assert daemon.try_search(project, None, options, FreshnessMode.check) is None


def test_try_search_falls_back_on_stale_socket(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIMGREP_NO_DAEMON", "0")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    sock_path = _short_socket()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))  # bound then closed: stale socket file
    server.close()
    spawned = False

    def _no_spawn(sock: Path, db: Path) -> None:
        nonlocal spawned
        spawned = True

    monkeypatch.setattr(daemon, "_spawn", _no_spawn)
    project = _fake_project(tmp_path)
    out = daemon.try_search(project, None, SearchOptions(query="q"), FreshnessMode.auto)
    assert out is None
    assert not sock_path.exists() or spawned or (time.time() - sock_path.stat().st_mtime) < 5.0


def test_roundtrip_proto_mismatch_unlinks_socket(tmp_path: Path) -> None:
    sock_path = _short_socket()
    responses = []

    def _bad_server() -> None:
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(sock_path))
        server.listen(1)
        responses.append("listening")
        conn, _ = server.accept()
        with conn:
            buffer = bytearray()
            while not buffer.endswith(b"\n"):
                chunk = conn.recv(65536)
                if not chunk:
                    return
                buffer.extend(chunk)
            conn.sendall(json.dumps({"proto": 999, "ok": True, "outcome": {}}).encode() + b"\n")
        server.close()
        responses.append("done")

    thread = threading.Thread(target=_bad_server)
    thread.start()
    while not responses:
        time.sleep(0.01)
    response = daemon._roundtrip(sock_path, {"proto": daemon.PROTO_VERSION})
    thread.join(timeout=5)
    assert response is None
    assert not sock_path.exists()


def test_spawn_throttled_by_fresh_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sock_path = _short_socket()
    sock_path.touch()
    calls: list[tuple[Path, Path]] = []

    def _fail_popen(*args: object, **kwargs: object) -> None:
        calls.append((sock_path, sock_path))  # pragma: no cover - must not run

    monkeypatch.setattr(subprocess, "Popen", _fail_popen)
    daemon._spawn(sock_path, tmp_path / "db.duckdb")
    assert not calls


# ------------------------------------------------------- _CachedStore tests ----


def _seed_store(db_path: Path) -> tuple[int, int]:
    """Build a small persistent index: one file, two chunks, lexical terms."""
    from simgrep.models import ChunkRecord, FileRecord, TermRecord
    from simgrep.store import Store

    store = Store.open(db_path)
    try:
        file_id = store.upsert_file(FileRecord(id=0, path=Path("mod1/a.py"), size_bytes=64, mtime_ns=1, language="python"))
        first, second = store.reserve_labels(2)
        store.insert_chunks(
            [
                ChunkRecord(
                    label=first, file_id=file_id, text="duckdb wal recovery internals", start_char=0, end_char=29, token_count=4, line_start=1, line_end=1
                ),
                ChunkRecord(
                    label=second, file_id=file_id, text="vector index save load path", start_char=30, end_char=57, token_count=5, line_start=2, line_end=2
                ),
            ]
        )
        terms = [
            TermRecord(label=label, term=term, field="chunk", tf=1, weight=1.0)
            for label, text in ((first, "duckdb wal recovery internals"), (second, "vector index save load path"))
            for term in text.split()
        ]
        store.insert_terms(terms)
        return first, second
    finally:
        store.close()


def _labels(rows: list[dict[str, Any]]) -> list[int]:
    return [int(row["label"]) for row in rows]


def test_cached_store_lookup_preserves_caller_order(tmp_path: Path) -> None:
    from simgrep.store import Store

    db = tmp_path / "meta.duckdb"
    first, second = _seed_store(db)
    real = Store.open(db, read_only=True)
    try:
        expected = real.lookup_chunks([second, first], ResultFilters())
    finally:
        real.close()
    cached = daemon._CachedStore(db)
    assert _labels(cached.lookup_chunks([second, first], ResultFilters())) == _labels(expected) == [second, first]


def test_cached_store_serves_warm_rows_without_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from simgrep.store import Store

    db = tmp_path / "meta.duckdb"
    _seed_store(db)
    cached = daemon._CachedStore(db)
    assert len(cached.lookup_chunks([1], ResultFilters())) == 1

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("Store.open must not be called for cached labels")

    monkeypatch.setattr(Store, "open", _boom)
    assert _labels(cached.lookup_chunks([1, 1], ResultFilters())) == [1, 1]


def test_cached_store_invalidates_on_db_change(tmp_path: Path) -> None:
    from simgrep.models import ChunkRecord, FileRecord
    from simgrep.store import Store

    db = tmp_path / "meta.duckdb"
    _seed_store(db)
    cached = daemon._CachedStore(db)
    assert cached.lookup_chunks([1], ResultFilters())

    store = Store.open(db)
    try:
        file_id = store.upsert_file(FileRecord(id=0, path=Path("mod1/b.py"), size_bytes=32, mtime_ns=2, language="python"))
        store.insert_chunks(
            [ChunkRecord(label=3, file_id=file_id, text="freshly added marker chunk", start_char=0, end_char=26, token_count=4, line_start=1, line_end=1)]
        )
    finally:
        store.close()

    assert _labels(cached.lookup_chunks([1, 3], ResultFilters())) == [1, 3]
    fresh = cached.lookup_chunks([3], ResultFilters())
    assert "freshly added marker chunk" in str(fresh[0]["chunk_text"])


def test_cached_store_filtered_queries_delegate(tmp_path: Path) -> None:
    from simgrep.store import Store

    db = tmp_path / "meta.duckdb"
    _seed_store(db)
    real = Store.open(db, read_only=True)
    try:
        expected = real.lookup_chunks([1, 2], ResultFilters(include_globs=("mod1/**",)))
    finally:
        real.close()
    cached = daemon._CachedStore(db)
    assert _labels(cached.lookup_chunks([1, 2], ResultFilters(include_globs=("mod1/**",)))) == _labels(expected)


def test_cached_store_lexical_cache_and_invalidation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from simgrep.store import Store

    db = tmp_path / "meta.duckdb"
    _seed_store(db)
    cached = daemon._CachedStore(db)
    first = cached.lexical_candidates(["duckdb", "wal"], 10, ResultFilters())
    assert first

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("Store.open must not be called for cached lexical queries")

    monkeypatch.setattr(Store, "open", _boom)
    assert _labels(cached.lexical_candidates(["duckdb", "wal"], 10, ResultFilters())) == _labels(first)

    monkeypatch.undo()
    store = Store.open(db)
    try:
        store.delete_file(1)
    finally:
        store.close()
    assert cached.lexical_candidates(["duckdb", "wal"], 10, ResultFilters()) == []


def test_cached_store_counts_invalidation(tmp_path: Path) -> None:
    from simgrep.store import Store

    db = tmp_path / "meta.duckdb"
    _seed_store(db)
    cached = daemon._CachedStore(db)
    assert cached.counts("p").chunks_count == 2

    store = Store.open(db)
    try:
        store.delete_file(1)
    finally:
        store.close()
    assert cached.counts("p").chunks_count == 0


class _CountingEmbedder:
    """Stub embedder recording every encode call."""

    ndim = 7

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], bool, int | None]] = []

    def encode(self, texts: list[str], *, is_query: bool = False, batch_size: int | None = None) -> list[float]:
        self.calls.append((tuple(texts), is_query, batch_size))
        return [float(len(t)) for t in texts]


def test_caching_embedder_memoizes_single_text_encodes() -> None:
    inner = _CountingEmbedder()
    cached = daemon._CachingEmbedder(inner)
    first = cached.encode(["abc"], is_query=True)
    second = cached.encode(["abc"], is_query=True)
    assert first == second == [3.0]
    assert len(inner.calls) == 1
    assert cached.encode(["abc"], is_query=False) == [3.0]
    assert cached.encode(["abcd"], is_query=True) == [4.0]
    assert len(inner.calls) == 3


def test_caching_embedder_delegates_batches_and_exposes_attrs() -> None:
    inner = _CountingEmbedder()
    cached = daemon._CachingEmbedder(inner)
    assert cached.encode(["aa", "b"], is_query=False, batch_size=8) == [2.0, 1.0]
    assert len(inner.calls) == 1
    assert inner.calls[0] == (("aa", "b"), False, 8)
    assert cached.ndim == 7


def test_caching_embedder_evicts_lru(monkeypatch: pytest.MonkeyPatch) -> None:
    inner = _CountingEmbedder()
    cached = daemon._CachingEmbedder(inner)
    monkeypatch.setattr(cached, "_CAP", 2)
    cached.encode(["a"], is_query=True)
    cached.encode(["bb"], is_query=True)
    cached.encode(["ccc"], is_query=True)
    assert len(inner.calls) == 3
    cached.encode(["a"], is_query=True)
    assert len(inner.calls) == 4
    cached.encode(["ccc"], is_query=True)
    assert len(inner.calls) == 4


class _FakeIndex:
    """Load-only index stub: refuses missing files like the real one."""

    ndim = 4

    def __init__(self) -> None:
        self.loaded: Path | None = None

    def load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        self.loaded = path


def _fp_project(tmp_path: Path) -> Any:
    from simgrep.project import init_project

    project = init_project(tmp_path, yes=True)
    project.metadata_db_path.parent.mkdir(parents=True, exist_ok=True)
    _seed_store(project.metadata_db_path)
    project.vector_index_path.write_text("1,2", encoding="utf-8")
    return project


def _fp_engine(plans: list[Any]) -> Any:
    """Engine fake whose always-clean plan_project records every call."""

    from types import SimpleNamespace

    def _plan(project: Any, config: Any, options: Any) -> Any:
        plans.append(project)
        return SimpleNamespace(new_count=0, changed_count=0, deleted_count=0)

    indexing = SimpleNamespace(
        scan_options=lambda config, options: SimpleNamespace(
            patterns=("*",), include_globs=(), exclude_globs=(), max_file_size_bytes=None, follow_symlinks=False
        ),
        plan_project=_plan,
    )
    runtime = SimpleNamespace(embedder=SimpleNamespace(ndim=4), new_vector_index=lambda ndim: _FakeIndex())
    return SimpleNamespace(runtime=runtime, indexing=indexing, search_reader=lambda reader, options: ([], 0))


def _fp_handle(project: Any, engine: Any) -> dict[str, Any]:
    from contextlib import nullcontext

    daemon._STORES.clear()
    request = {"root": str(project.root), "options": daemon._serialize_options(SearchOptions(query="warm"))}
    from simgrep.models import AppConfig as _AppConfig

    return daemon._handle(request, engine, daemon._IndexCache(), _AppConfig(model="fake"), lambda path, is_singleton=True: nullcontext(), engine.indexing)


def _reset_fp_state() -> None:
    daemon._FINGERPRINTS.clear()
    daemon._FP_LOADED.clear()


def test_fingerprint_sidecar_skips_plan_after_restart(tmp_path: Path) -> None:
    project = _fp_project(tmp_path)
    plans: list[Any] = []
    engine = _fp_engine(plans)

    _reset_fp_state()
    assert _fp_handle(project, engine)["ok"] is True
    assert len(plans) == 1

    _reset_fp_state()
    assert _fp_handle(project, engine)["ok"] is True
    assert len(plans) == 1


def test_fingerprint_sidecar_voided_by_db_rewrite(tmp_path: Path) -> None:
    project = _fp_project(tmp_path)
    plans: list[Any] = []
    engine = _fp_engine(plans)
    _reset_fp_state()
    assert _fp_handle(project, engine)["ok"] is True
    assert len(plans) == 1

    _seed_store(project.metadata_db_path)
    _reset_fp_state()
    assert _fp_handle(project, engine)["ok"] is True
    assert len(plans) == 2

    _reset_fp_state()
    assert _fp_handle(project, engine)["ok"] is True
    assert len(plans) == 2


def test_fingerprint_sidecar_replans_when_disk_changes(tmp_path: Path) -> None:
    project = _fp_project(tmp_path)
    plans: list[Any] = []
    engine = _fp_engine(plans)
    _reset_fp_state()
    assert _fp_handle(project, engine)["ok"] is True

    (project.root / "new_file.py").write_text("x = 1", encoding="utf-8")
    assert _fp_handle(project, engine)["ok"] is True
    assert len(plans) == 2


def test_fingerprint_sidecar_corrupt_or_absent_is_ignored(tmp_path: Path) -> None:
    project = _fp_project(tmp_path)
    plans: list[Any] = []
    engine = _fp_engine(plans)
    _reset_fp_state()
    assert _fp_handle(project, engine)["ok"] is True
    cache = daemon._fp_cache_path(project)
    assert cache.exists()

    cache.write_text("{not json", encoding="utf-8")
    _reset_fp_state()
    assert _fp_handle(project, engine)["ok"] is True
    assert len(plans) == 2

    cache.unlink()
    _reset_fp_state()
    assert _fp_handle(project, engine)["ok"] is True
    assert len(plans) == 3
    assert cache.exists()


def test_fingerprint_sidecar_roundtrip_preserves_tuples(tmp_path: Path) -> None:
    project = _fp_project(tmp_path)
    scan_key = (("*",), ("a/**",), (), None, False)
    fingerprint = (("", 1, 2, 3), ("sub/", 1, 3, 4), ("sub/a.py", 1, 5, 6, 7))
    daemon._persist_fingerprint_cache(project, scan_key, fingerprint)
    daemon._FINGERPRINTS.pop(project.metadata_db_path, None)
    daemon._FP_LOADED.clear()
    daemon._load_fingerprint_cache(project)
    restored = daemon._FINGERPRINTS[project.metadata_db_path]
    assert restored == (scan_key, fingerprint)
    assert isinstance(restored[1][0], tuple)
    assert isinstance(restored[0][0], tuple)


def test_disk_fingerprint_records_tree_and_prunes(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    (root / "pkg" / "nested").mkdir(parents=True)
    (root / "pkg" / "a.py").write_text("x")
    (root / "pkg" / "nested" / "b.txt").write_text("y")
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("z")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "a.pyc").write_text("z")
    (root / "loop").symlink_to(root)

    opts = SimpleNamespace(follow_symlinks=False)
    fp = daemon._disk_fingerprint(root, opts)
    assert fp is not None
    rels = [entry[0] for entry in fp]
    assert rels[0] == ""
    assert len(fp) == 1 + 2 + 2  # root dir, pkg/, pkg/nested/, two files
    assert "pkg/" in rels and "pkg/a.py" in rels and "pkg/nested/" in rels
    assert not any(r.startswith((".git", "__pycache__", "loop")) for r in rels)
    assert daemon._disk_fingerprint(root, opts) == fp

    (root / "pkg" / "a.py").write_text("yy")
    assert daemon._disk_fingerprint(root, opts) != fp


def test_disk_fingerprint_follows_symlinks_without_looping(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "a.py").write_text("x")
    (root / "loop").symlink_to(root)

    fp = daemon._disk_fingerprint(root, SimpleNamespace(follow_symlinks=True))
    assert fp is not None
    assert "pkg/a.py" in [entry[0] for entry in fp]
    assert not any(entry[0].startswith("loop") for entry in fp)


def test_disk_fingerprint_none_on_missing_root(tmp_path: Path) -> None:
    assert daemon._disk_fingerprint(tmp_path / "absent", SimpleNamespace(follow_symlinks=False)) is None
