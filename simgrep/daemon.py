"""Warm search daemon: unix-socket proxy that keeps the ORT session and
usearch index resident across CLI invocations.

A `simgrep search` costs ~440ms wall today; roughly 250ms of that is process
setup the daemon amortizes: python imports, ONNX session build, tokenizer
load. The client hook lives in ``SearchEngine.search_project``: it tries the
daemon first and falls back to the normal in-process path on any failure
(missing daemon, stale index, protocol mismatch, timeout). The daemon never
mutates the index: it re-plans freshness under the project FileLock per query
and answers ``stale`` when work is found, letting the client index and search
in-process. The metadata store is opened read-only per query and closed, so a
resident daemon never blocks a concurrent ``simgrep index`` writer.

Protocol: one JSON request line -> one JSON response line, ``proto`` field
guards version skew (mismatch = client fallback + socket removal).
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from simgrep.models import (
        FreshnessMode,
        SearchOptions,
        SearchOutcome,
    )
    from simgrep.store import Store

PROTO_VERSION = 1
IDLE_TIMEOUT_SECONDS = 600.0
CONNECT_TIMEOUT_SECONDS = 0.3
RESPONSE_TIMEOUT_SECONDS = 15.0

_SPAWN_GUARD = threading.Lock()


def socket_path_for(metadata_db_path: Path) -> Path:
    digest = hashlib.sha1(str(Path(metadata_db_path).resolve()).encode(), usedforsecurity=False).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / "simgrep" / "daemons" / f"{digest}.sock"


# ---------------------------------------------------------------- client ----


def try_search(project: Any, app_config: Any, options: SearchOptions, freshness: FreshnessMode) -> SearchOutcome | None:
    """Serve ``search_project`` via the daemon; ``None`` means fall back."""
    from simgrep.models import FreshnessMode

    if os.environ.get("SIMGREP_NO_DAEMON") == "1" or "PYTEST_CURRENT_TEST" in os.environ or freshness is not FreshnessMode.auto:
        return None
    sock_path = socket_path_for(project.metadata_db_path)
    request = {
        "proto": PROTO_VERSION,
        "op": "search",
        "root": str(project.root),
        "options": _serialize_options(options),
    }
    response = _roundtrip(sock_path, request)
    if os.environ.get("SIMGREP_DAEMON_DEBUG"):
        print(f"[daemon] response={'none' if response is None else response.get('ok')} error={response.get('error') if response else None}", file=sys.stderr)
    if response is None:
        # Dead or starting daemon: ensure a spawn. The marker file written by
        # _spawn throttles respawn storms while the daemon boots.
        _spawn(sock_path, project.metadata_db_path)
        return None
    if not response.get("ok"):
        return None
    try:
        return _deserialize_outcome(response["outcome"])
    except Exception:
        return None


def _roundtrip(sock_path: Path, request: dict[str, Any]) -> dict[str, Any] | None:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(CONNECT_TIMEOUT_SECONDS)
            sock.connect(str(sock_path))
            sock.settimeout(RESPONSE_TIMEOUT_SECONDS)
            sock.sendall(json.dumps(request).encode() + b"\n")
            buffer = bytearray()
            while not buffer.endswith(b"\n"):
                chunk = sock.recv(65536)
                if not chunk:
                    return None
                buffer.extend(chunk)
        response = json.loads(bytes(buffer))
    except (OSError, ValueError):
        return None
    if response.get("proto") != PROTO_VERSION:
        try:
            sock_path.unlink()
        except OSError:
            pass
        return None
    return dict(response)


def _spawn(sock_path: Path, metadata_db_path: Path) -> None:
    with _SPAWN_GUARD:
        try:
            if time.time() - sock_path.stat().st_mtime < 5.0:
                return  # a daemon was spawned recently; let it boot
        except OSError:
            pass
        try:
            sock_path.parent.mkdir(parents=True, exist_ok=True)
            sock_path.touch(exist_ok=True)  # respawn-throttle marker
            import subprocess

            subprocess.Popen(
                [sys.executable, "-m", "simgrep.daemon", "--socket", str(sock_path), "--db", str(metadata_db_path)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=open(Path(tempfile.gettempdir()) / "simgrep_daemon_err.txt", "ab") if os.environ.get("SIMGREP_DAEMON_DEBUG") else subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            pass


def _serialize_options(options: SearchOptions) -> dict[str, Any]:
    return {
        "query": options.query,
        "top": options.top,
        "min_score": options.min_score,
        "candidate_top": options.candidate_top,
        "lexical_top": options.lexical_top,
        "lexical_weight": options.lexical_weight,
        "diversity": options.diversity.value,
        "scope_path": str(options.scope_path) if options.scope_path else None,
        "file_filter": list(options.file_filter),
        "keyword_filter": options.keyword_filter,
        "include_globs": list(options.include_globs),
        "exclude_globs": list(options.exclude_globs),
        "path_boosts": [{"pattern": b.pattern, "weight": b.weight} for b in options.path_boosts],
        "lexical_fallback": options.lexical_fallback.value,
        "expr": options.expr,
    }


def _deserialize_options(raw: dict[str, Any]) -> SearchOptions:
    from simgrep.models import DiversityMode, LexicalFallbackMode, PathBoost, SearchOptions

    return SearchOptions(
        query=raw["query"],
        top=raw["top"],
        min_score=raw["min_score"],
        candidate_top=raw["candidate_top"],
        lexical_top=raw["lexical_top"],
        lexical_weight=raw["lexical_weight"],
        diversity=DiversityMode(raw["diversity"]),
        scope_path=Path(raw["scope_path"]) if raw["scope_path"] else None,
        file_filter=tuple(raw["file_filter"]),
        keyword_filter=raw["keyword_filter"],
        include_globs=tuple(raw["include_globs"]),
        exclude_globs=tuple(raw["exclude_globs"]),
        path_boosts=tuple(PathBoost(b["pattern"], b["weight"]) for b in raw["path_boosts"]),
        lexical_fallback=LexicalFallbackMode(raw["lexical_fallback"]),
        expr=raw["expr"],
    )


def _serialize_outcome(outcome: SearchOutcome) -> dict[str, Any]:
    results = [
        {
            "label": r.label,
            "score": r.score,
            "file_path": str(r.file_path),
            "chunk_text": r.chunk_text,
            "start_char": r.start_char,
            "end_char": r.end_char,
            "line_start": r.line_start,
            "line_end": r.line_end,
            "file_role": r.file_role.value,
            "language": r.language,
            "why": r.why,
        }
        for r in outcome.results
    ]
    return {
        "results": results,
        "base_path": str(outcome.base_path),
        "files_seen": outcome.files_seen,
        "chunks_searched": outcome.chunks_searched,
        "semantic_candidates": outcome.semantic_candidates,
    }


def _deserialize_outcome(raw: dict[str, Any]) -> SearchOutcome:
    from simgrep.models import FileRole, SearchOutcome, SearchResult

    results = [
        SearchResult(
            label=r["label"],
            score=r["score"],
            file_path=Path(r["file_path"]),
            chunk_text=r["chunk_text"],
            start_char=r["start_char"],
            end_char=r["end_char"],
            line_start=r["line_start"],
            line_end=r["line_end"],
            file_role=FileRole(r["file_role"]),
            language=r["language"],
            why=r["why"],
        )
        for r in raw["results"]
    ]
    return SearchOutcome(
        results=results,
        base_path=Path(raw["base_path"]),
        files_seen=raw["files_seen"],
        chunks_searched=raw["chunks_searched"],
        semantic_candidates=raw["semantic_candidates"],
    )


# ---------------------------------------------------------------- server ----


class _IndexCache:
    """Resident usearch index keyed by file identity; reloaded on change."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[tuple[int, int], Any, int]] = {}

    def get(self, path: Path, ndim: int) -> Any | None:
        try:
            stat = path.stat()
        except OSError:
            self._entries.pop(str(path), None)
            return None
        identity = (stat.st_mtime_ns, stat.st_size)
        entry = self._entries.get(str(path))
        if entry and entry[0] == identity and entry[2] == ndim:
            return entry[1]
        return None

    def put(self, path: Path, ndim: int, index: Any) -> None:
        try:
            stat = path.stat()
        except OSError:
            return
        self._entries[str(path)] = ((stat.st_mtime_ns, stat.st_size), index, ndim)


_STORES: dict[str, _CachedStore] = {}
"""Resident per-project metadata caches, keyed by DB path (single-threaded serve loop)."""


_FINGERPRINTS: dict[Path, tuple[tuple[Any, ...], tuple[Any, ...]]] = {}
"""Per-project ``(scan key, disk fingerprint)`` from the last clean plan."""

_FP_CACHE_PROTO = 1
_FP_CACHE_NAME = "fingerprint_cache.json"
_FP_LOADED: set[Path] = set()
"""DB paths whose side-car fingerprint cache was already hydrated this process."""


def _fp_cache_path(project: Any) -> Path:
    return Path(project.metadata_db_path).parent / _FP_CACHE_NAME


def _as_tuples(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_as_tuples(item) for item in value)
    return value


def _load_fingerprint_cache(project: Any) -> None:
    """Hydrate ``_FINGERPRINTS`` from the side-car a previous daemon process
    wrote after its clean plan, so a restart on an unchanged project skips
    the ~200ms freshness plan on its first served query. Guarded by the DB
    file identity captured at persist time — any index rebuild rewrites the
    DB and voids the cache. Trust model is identical to the in-process
    cache, one process wide: the per-query disk fingerprint still gates
    every serve, so later edits re-plan exactly as before.
    """
    db_path = project.metadata_db_path
    if db_path in _FP_LOADED:
        return
    _FP_LOADED.add(db_path)
    try:
        stat = db_path.stat()
        payload = json.loads(_fp_cache_path(project).read_text())
    except (OSError, ValueError):
        return
    if not isinstance(payload, dict) or payload.get("proto") != _FP_CACHE_PROTO:
        return
    identity = payload.get("db_identity")
    if not isinstance(identity, list) or tuple(identity) != (stat.st_mtime_ns, stat.st_size):
        return
    raw_scan = payload.get("scan_key")
    raw_fingerprint = payload.get("fingerprint")
    if isinstance(raw_scan, list) and isinstance(raw_fingerprint, list):
        _FINGERPRINTS[db_path] = (_as_tuples(raw_scan), _as_tuples(raw_fingerprint))


def _persist_fingerprint_cache(project: Any, scan_key: tuple[Any, ...], fingerprint: tuple[Any, ...]) -> None:
    """Atomically record the last clean-plan ``(scan key, fingerprint)`` next
    to the metadata DB. Pure accelerator: every failure is swallowed and the
    next daemon simply re-plans.
    """
    db_path = project.metadata_db_path
    tmp: Path | None = None
    try:
        stat = db_path.stat()
        path = _fp_cache_path(project)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        payload = {
            "proto": _FP_CACHE_PROTO,
            "db_identity": [stat.st_mtime_ns, stat.st_size],
            "scan_key": scan_key,
            "fingerprint": fingerprint,
        }
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, path)
        tmp = None
    except (OSError, TypeError, ValueError):
        pass
    finally:
        if tmp is not None:
            try:
                tmp.unlink()
            except OSError:
                pass


def _disk_fingerprint(root: Path, scan_options: Any) -> tuple[Any, ...] | None:
    """Fingerprint every filesystem input ``scan_files`` reads.

    Traverses in the same sorted pre-order as ``simgrep.files.scan_files`` and
    prunes the same directories (ignore specs, ``IGNORED_DIR_NAMES``, symlink
    rules), but records ``(rel, dev, ino, size, mtime_ns)`` for files and
    ``(rel, dev, ino, mtime_ns)`` for dirs without any ``realpath`` work.
    Superset-safe: files the scan would filter out or dedup away are still
    fingerprinted, so any change the scan could observe flips the tuple.
    Returns ``None`` on any OSError; callers must then run the full plan.
    """
    from simgrep.files import IGNORED_DIR_NAMES, _read_ignore_lines, _rewrite_ignore_lines

    try:
        root_resolved = root.resolve()
        root_stat = root_resolved.stat()
    except OSError:
        return None
    parts: list[tuple[Any, ...]] = [("", root_stat.st_dev, root_stat.st_ino, root_stat.st_mtime_ns)]
    ignore_lines: list[str] = []
    spec_cache: tuple[int, Any] | None = None
    follow = scan_options.follow_symlinks
    root_prefix = str(root_resolved)
    root_prefix = "" if root_prefix == "/" else root_prefix.rstrip("/")

    def _resolved_str(path: Path) -> str:
        # With symlinks unfollowed, every traversed path string already sits in
        # resolved space (root resolved once, symlink entries pruned), so skip
        # the per-directory realpath syscalls entirely.
        return str(path) if not follow else str(path.resolve())

    visited: set[str] = set()

    def _rel(item_path: str) -> str:
        return item_path[len(root_prefix) + 1 :] if root_prefix else item_path[1:]

    def _accumulated_spec() -> Any:
        nonlocal spec_cache
        if not ignore_lines:
            return None
        if spec_cache is None or spec_cache[0] != len(ignore_lines):
            import pathspec

            try:
                built: Any = pathspec.PathSpec.from_lines("gitwildmatch", ignore_lines)
            except Exception:
                built = None
            spec_cache = (len(ignore_lines), built)
        return spec_cache[1]

    def _walk(current: Path, dir_rel: str) -> bool:
        try:
            real = _resolved_str(current)
            if real in visited:
                return True
            visited.add(real)
            ignore_lines.extend(_rewrite_ignore_lines(_read_ignore_lines(current), dir_rel))
            entries = sorted(os.scandir(current), key=lambda item: item.name)
        except OSError:
            return False
        for item in entries:
            rel = f"{dir_rel}/{item.name}" if dir_rel else item.name
            try:
                if item.is_symlink() and not follow:
                    continue
                if item.is_dir(follow_symlinks=follow):
                    resolved_str = _resolved_str(Path(item.path))
                    if resolved_str in visited or item.name in IGNORED_DIR_NAMES:
                        continue
                    spec = _accumulated_spec()
                    if spec is not None:
                        if follow:
                            try:
                                child_rel = Path(resolved_str).relative_to(root_resolved).as_posix()
                            except ValueError:
                                child_rel = ""
                        else:
                            child_rel = _rel(item.path)
                        if child_rel and spec.match_file(child_rel + "/"):
                            continue
                    try:
                        st = item.stat(follow_symlinks=follow)
                    except OSError:
                        return False
                    parts.append((rel + "/", st.st_dev, st.st_ino, st.st_mtime_ns))
                    if not _walk(Path(item.path), rel):
                        return False
                    continue
                if not item.is_file(follow_symlinks=follow):
                    continue
                try:
                    st = item.stat(follow_symlinks=follow)
                except OSError:
                    return False
            except OSError:
                return False
            parts.append((rel, st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns))
        return True

    if not _walk(root_resolved, ""):
        return None
    parts.sort()
    return tuple(parts)


class _CachingEmbedder:
    """LRU memo for single-text encodes; multi-text batches pass through.

    The warm handler is dominated by the ONNX query encode (~6ms); users
    re-run the same queries, so memoize ``(text, is_query)``. Vectors are
    returned shared — every caller treats embeddings as read-only (they are
    only fed to usearch search). Batch encodes (indexing) always delegate.
    """

    _CAP = 256

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._cache: OrderedDict[tuple[str, bool], Any] = OrderedDict()

    @property
    def ndim(self) -> int:
        return int(self._inner.ndim)

    def encode(self, texts: list[str], *, is_query: bool = False, batch_size: int | None = None) -> Any:
        if len(texts) != 1:
            return self._inner.encode(texts, is_query=is_query, batch_size=batch_size)
        key = (texts[0], bool(is_query))
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        vec = self._inner.encode(texts, is_query=is_query, batch_size=batch_size)
        self._cache[key] = vec
        while len(self._cache) > self._CAP:
            self._cache.popitem(last=False)
        return vec

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def serve(socket_path: Path, metadata_db_path: Path) -> int:
    from dataclasses import replace

    from filelock import FileLock

    from simgrep.config import load_app_config
    from simgrep.indexing import IndexEngine
    from simgrep.runtime import RuntimeFactory
    from simgrep.search import SearchEngine

    try:
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        if socket_path.exists():
            socket_path.unlink()
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(socket_path))
        os.chmod(socket_path, 0o600)
        server.listen(4)
    except OSError:
        return 1

    def _cleanup(_signum: int, _frame: Any) -> None:
        try:
            socket_path.unlink()
        except OSError:
            pass
        raise SystemExit(0)

    import signal

    signal.signal(signal.SIGTERM, _cleanup)
    signal.signal(signal.SIGINT, _cleanup)

    factory = RuntimeFactory()
    runtime = factory.for_app(load_app_config())
    runtime = replace(
        runtime,
        embedder=_CachingEmbedder(runtime.embedder),
        query_embedder=_CachingEmbedder(runtime.query_embedder),
    )
    engine = SearchEngine(runtime)
    indexing = IndexEngine(runtime)  # public plan/scan surface; fingerprinting needs it
    app_config = load_app_config()
    index_cache = _IndexCache()
    server.settimeout(IDLE_TIMEOUT_SECONDS)
    try:
        while True:
            try:
                conn, _ = server.accept()
            except (socket.timeout, OSError):
                break
            with conn:
                conn.settimeout(RESPONSE_TIMEOUT_SECONDS)
                try:
                    request = _read_request(conn)
                    response = (
                        {"proto": PROTO_VERSION, "ok": False, "error": "bad-request"}
                        if request is None
                        else _handle(request, engine, index_cache, app_config, FileLock, indexing)
                    )
                except Exception:
                    response = {"proto": PROTO_VERSION, "ok": False, "error": "internal"}
                try:
                    conn.sendall(json.dumps(response).encode() + b"\n")
                except OSError:
                    pass
    finally:
        server.close()
        try:
            socket_path.unlink()
        except OSError:
            pass
    return 0


def _read_request(conn: socket.socket) -> dict[str, Any] | None:
    buffer = bytearray()
    while not buffer.endswith(b"\n"):
        chunk = conn.recv(65536)
        if not chunk:
            return None
        buffer.extend(chunk)
        if len(buffer) > 1 << 20:
            return None
    request = json.loads(bytes(buffer))
    if request.get("proto") != PROTO_VERSION:
        raise ValueError("proto mismatch")
    return dict(request)


class _CachedStore:
    """Resident read cache over a project's metadata DB.

    Serves ``lookup_chunks`` rows, ``lexical_candidates`` results, and
    ``counts`` from memory while the DB file identity is unchanged; the real
    store opens lazily and only for cache misses, so a resident daemon never
    blocks a concurrent ``simgrep index`` writer. Rows are cached only for
    the default (empty) result filters; filtered queries always delegate.
    All access happens under the project FileLock held by ``_handle``, so a
    DB rewrite can never race a cached serve.
    """

    _MAX_ROWS = 65536
    _MAX_LEXICAL = 256

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._identity: tuple[int, int] | None = None
        self._rows: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._counts: dict[str, Any] = {}
        self._lexical: OrderedDict[tuple[Any, ...], list[dict[str, Any]]] = OrderedDict()

    def _ensure_current(self) -> bool:
        """Drop caches when the DB file changed; ``False`` if unreadable."""
        try:
            stat = self._db_path.stat()
        except OSError:
            return False
        identity = (stat.st_mtime_ns, stat.st_size)
        if identity != self._identity:
            self._identity = identity
            self._rows.clear()
            self._counts.clear()
            self._lexical.clear()
        return True

    @staticmethod
    def _unfiltered(filters: Any) -> bool:
        return not (filters.scope_path or filters.file_filter or filters.keyword_filter or filters.include_globs or filters.exclude_globs)

    @contextmanager
    def _real(self) -> Iterator[Store]:
        from simgrep.store import Store

        store = Store.open(self._db_path, read_only=True)
        try:
            yield store
        finally:
            store.close()

    def lookup_chunks(self, labels: list[int], filters: Any) -> list[dict[str, Any]]:
        if not labels:
            return []
        if not self._unfiltered(filters) or not self._ensure_current():
            with self._real() as store:
                return store.lookup_chunks(labels, filters)
        missing = [label for label in labels if label not in self._rows]
        if missing:
            with self._real() as store:
                for row in store.lookup_chunks(missing, filters):
                    self._rows[int(row["label"])] = row
            while len(self._rows) > self._MAX_ROWS:
                self._rows.popitem(last=False)
        return [self._rows[label] for label in labels if label in self._rows]

    def lexical_candidates(self, query_terms: list[str], limit: int, filters: Any) -> list[dict[str, Any]]:
        if not query_terms or limit <= 0:
            return []
        key: tuple[Any, ...] = (tuple(query_terms), limit)
        if self._unfiltered(filters) and self._ensure_current():
            cached = self._lexical.get(key)
            if cached is not None:
                self._lexical.move_to_end(key)
                return cached
        with self._real() as store:
            details = store.lexical_candidates(query_terms, limit, filters)
        if self._unfiltered(filters) and self._ensure_current():
            self._lexical[key] = details
            while len(self._lexical) > self._MAX_LEXICAL:
                self._lexical.popitem(last=False)
        return details

    def counts(self, project_name: str = "") -> Any:
        if self._ensure_current() and project_name in self._counts:
            return self._counts[project_name]
        with self._real() as store:
            status = store.counts(project_name)
        if self._ensure_current():
            self._counts[project_name] = status
        return status

    def close(self) -> None:
        """No persistent connection; nothing to release."""


def _handle(
    request: dict[str, Any],
    engine: Any,
    index_cache: _IndexCache,
    app_config: Any,
    file_lock: Any,
    indexing: Any = None,
) -> dict[str, Any]:
    from simgrep.indexing import IndexOptions
    from simgrep.models import SearchOutcome
    from simgrep.project import load_project_config

    project = load_project_config(Path(request["root"]))
    if not project.metadata_db_path.exists():
        return {"proto": PROTO_VERSION, "ok": False, "error": "no-index"}
    options = _deserialize_options(request["options"])
    lock = file_lock(str(project.index_lock_path), is_singleton=True)
    with lock:
        _load_fingerprint_cache(project)
        if indexing is None:
            from simgrep.indexing import IndexEngine as _IE

            indexing = _IE(engine.runtime)
        scan_options = indexing.scan_options(app_config, IndexOptions())
        scan_key = (
            scan_options.patterns,
            scan_options.include_globs,
            scan_options.exclude_globs,
            scan_options.max_file_size_bytes,
            scan_options.follow_symlinks,
        )
        fingerprint = _disk_fingerprint(project.root, scan_options)
        cached_fp = _FINGERPRINTS.get(project.metadata_db_path)
        if cached_fp is None or fingerprint is None or fingerprint != cached_fp[1] or scan_key != cached_fp[0]:
            plan = indexing.plan_project(project, app_config, IndexOptions())
            if bool(plan.new_count or plan.changed_count or plan.deleted_count):
                return {"proto": PROTO_VERSION, "ok": False, "error": "stale"}
            if fingerprint is not None:
                _FINGERPRINTS[project.metadata_db_path] = (scan_key, fingerprint)
                _persist_fingerprint_cache(project, scan_key, fingerprint)
        cached = index_cache.get(project.vector_index_path, engine.runtime.embedder.ndim)
        store = _STORES.get(str(project.metadata_db_path))
        if store is None:
            store = _CachedStore(project.metadata_db_path)
            _STORES[str(project.metadata_db_path)] = store
        if cached is None:
            cached = engine.runtime.new_vector_index(engine.runtime.embedder.ndim)
            cached.load(project.vector_index_path)
            index_cache.put(project.vector_index_path, engine.runtime.embedder.ndim, cached)
        # Real reader over the resident index and the cached metadata source.
        from simgrep.corpus import CorpusReader

        reader = CorpusReader(store=store, index=cached, base_path=project.root)
        results, semantic_count = engine.search_reader(reader, options)
        counts = store.counts(project.name)
    outcome = SearchOutcome(
        results=results,
        base_path=project.root,
        files_seen=counts.files_count,
        chunks_searched=counts.chunks_count,
        semantic_candidates=semantic_count,
    )
    return {"proto": PROTO_VERSION, "ok": True, "outcome": _serialize_outcome(outcome)}


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    sock = Path(args[args.index("--socket") + 1]) if "--socket" in args else None
    db = Path(args[args.index("--db") + 1]) if "--db" in args else None
    if sock is None or db is None:
        return 2
    return serve(sock, db)


if __name__ == "__main__":
    raise SystemExit(main())
