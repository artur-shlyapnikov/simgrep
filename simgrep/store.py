from __future__ import annotations

import fnmatch
import math
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, cast

import duckdb

from simgrep.errors import MetadataError
from simgrep.models import SCHEMA_VERSION, ChunkRecord, FileRecord, FileRole, IndexState, ProjectStatus, ResultFilters, SearchResult, TermRecord

SCHEMA_HINT = "Run `simgrep index --rebuild`."


def _neuter_missing_frame_libraries() -> None:
    """Pin confirmed-missing optional DuckDB frame libraries to ``None`` in ``sys.modules``.

    DuckDB probes ``import pandas`` (and friends) on every query. When pandas is
    genuinely absent each probe pays a full ``sys.path`` scan (~10 stats) before
    failing; a ``None`` entry makes the same ``ImportError`` raise immediately
    without touching the filesystem. Only applied when the module is truly
    unimportable, so behavior is identical -- the import still fails.
    """
    for name in ("pandas", "pyarrow"):
        if name in sys.modules:
            continue
        try:
            if find_spec(name) is not None:
                continue
        except (ImportError, ValueError):
            continue
        sys.modules[name] = None  # type: ignore[assignment]


_neuter_missing_frame_libraries()


def _sql_literal(value: Any) -> str:
    """Render a Python value as a DuckDB SQL literal (strings escaped by doubling quotes)."""
    typ = type(value)
    if typ is int:
        return str(value)
    if typ is str:
        return "'" + cast(str, value).replace("'", "''") + "'"
    if typ is float:
        return repr(value) if math.isfinite(value) else f"'{value}'::DOUBLE"
    if value is None:
        return "NULL"
    if typ is bool:
        return "TRUE" if value else "FALSE"
    raise MetadataError(f"Unsupported literal type for bulk insert: {type(value).__name__}")


class Store:
    def __init__(self, conn: duckdb.DuckDBPyConnection, *, read_only: bool = False, path: Path | None = None) -> None:
        self._conn = conn
        self._read_only = read_only
        self.path = path

    @classmethod
    def open(cls, path: Path, read_only: bool = False) -> "Store":
        if read_only and not path.exists():
            raise MetadataError(f"Persistent database not found: {path}", hint="Run `simgrep index` first.")
        if not read_only:
            path.parent.mkdir(parents=True, exist_ok=True)
        try:
            conn = duckdb.connect(str(path), read_only=read_only)
        except duckdb.Error as exc:
            raise MetadataError(
                f"Failed to open metadata database: {path}",
                hint="The database file may be corrupted. Run `simgrep reset` to remove local artifacts, then `simgrep index` to rebuild.",
            ) from exc
        store = cls(conn, read_only=read_only, path=path)
        store.ensure_schema()
        return store

    @classmethod
    def memory(cls) -> "Store":
        try:
            conn = duckdb.connect(":memory:")
        except duckdb.Error as exc:
            raise MetadataError("Failed to create in-memory metadata database") from exc
        store = cls(conn)
        store.ensure_schema()
        return store

    def close(self) -> None:
        self._conn.close()

    def ensure_schema(self) -> None:
        try:
            table_names = {str(row[0]) for row in self._conn.execute("SHOW TABLES").fetchall()}
            if self._read_only:
                if "meta" not in table_names:
                    raise MetadataError("Missing simgrep metadata schema.", hint=SCHEMA_HINT)
                self._validate_schema_version()
                return
            if "indexed_files" in table_names or "text_chunks" in table_names:
                raise MetadataError("Old simgrep metadata schema detected.", hint=SCHEMA_HINT)
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS files (
                    id BIGINT PRIMARY KEY,
                    path VARCHAR UNIQUE NOT NULL,
                    size_bytes BIGINT NOT NULL,
                    mtime_ns BIGINT NOT NULL,
                    sha256 VARCHAR,
                    role VARCHAR NOT NULL,
                    language VARCHAR NOT NULL,
                    is_test BOOLEAN NOT NULL DEFAULT FALSE,
                    is_generated BOOLEAN NOT NULL DEFAULT FALSE,
                    indexed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    label BIGINT PRIMARY KEY,
                    file_id BIGINT NOT NULL,
                    text TEXT NOT NULL,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL,
                    token_count INTEGER NOT NULL,
                    line_start INTEGER,
                    line_end INTEGER,
                    kind VARCHAR NOT NULL DEFAULT 'mixed'
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS terms (
                    label BIGINT NOT NULL,
                    term VARCHAR NOT NULL,
                    field VARCHAR NOT NULL,
                    tf INTEGER NOT NULL,
                    weight DOUBLE NOT NULL,
                    PRIMARY KEY (label, term, field)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS term_stats (
                    term VARCHAR PRIMARY KEY,
                    chunk_df BIGINT NOT NULL,
                    path_df BIGINT NOT NULL,
                    symbol_df BIGINT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key VARCHAR PRIMARY KEY,
                    value VARCHAR NOT NULL
                )
                """
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS chunks_file_id_idx ON chunks(file_id)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS terms_term_idx ON terms(term)")
            self._validate_schema_version()
        except MetadataError:
            raise
        except duckdb.Error as exc:
            raise MetadataError("Failed to initialize metadata schema") from exc

    def _validate_schema_version(self) -> None:
        value = self.get_meta("schema_version")
        if value is None:
            if self._read_only:
                raise MetadataError("Missing schema version.", hint=SCHEMA_HINT)
            self.set_meta("schema_version", str(SCHEMA_VERSION))
            return
        try:
            version = int(value)
        except ValueError:
            raise MetadataError(f"Invalid schema version value: {value}.", hint=SCHEMA_HINT)
        if version != SCHEMA_VERSION:
            raise MetadataError(f"Unsupported schema version {value}.", hint=SCHEMA_HINT)

    def clear(self) -> None:
        self._conn.execute("DELETE FROM terms")
        self._conn.execute("DELETE FROM chunks")
        self._conn.execute("DELETE FROM files")
        self._conn.execute("DELETE FROM term_stats")
        self.set_meta("max_label", "-1")

    def get_files(self) -> dict[Path, FileRecord]:
        rows = self._conn.execute("SELECT id, path, size_bytes, mtime_ns, sha256, role, language, is_test, is_generated FROM files").fetchall()
        out: dict[Path, FileRecord] = {}
        for row in rows:
            # upsert_file persists ``record.path.resolve()``: paths are already
            # canonical, so re-resolving each row only burns syscalls per search.
            path = Path(str(row[1]))
            out[path] = FileRecord(
                id=int(row[0]),
                path=path,
                size_bytes=int(row[2]),
                mtime_ns=int(row[3]),
                sha256=None if row[4] is None else str(row[4]),
                role=FileRole(str(row[5])),
                language=str(row[6]),
                is_test=bool(row[7]),
                is_generated=bool(row[8]),
            )
        return out

    def reserve_labels(self, count: int) -> list[int]:
        if count <= 0:
            return []
        current = int(self.get_meta("max_label") or "-1")
        labels = list(range(current + 1, current + 1 + count))
        self.set_meta("max_label", str(labels[-1]))
        return labels

    def _next_file_id(self) -> int:
        row = self._conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM files").fetchone()
        assert row is not None
        return int(row[0])

    def upsert_file(self, record: FileRecord) -> int:
        return self.upsert_files([record])[0]

    def upsert_files(self, records: list[FileRecord]) -> list[int]:
        """Batch upsert preserving :meth:`upsert_file` semantics.

        One ``MAX(id)`` probe (only when some row needs an id) plus chunked
        multi-values ``INSERT ... ON CONFLICT DO UPDATE ... RETURNING id, path``
        statements. The writer thread is serial and the index lock is held for
        the whole run, so client-side id assignment cannot collide.
        """
        if not records:
            return []
        resolved = [str(record.path.resolve()) for record in records]
        ids: list[int] = []
        if any(record.id <= 0 for record in records):
            row = self._conn.execute("SELECT COALESCE(MAX(id), 0) FROM files").fetchone()
            assert row is not None
            nxt = int(row[0])
            for record in records:
                if record.id > 0:
                    ids.append(record.id)
                else:
                    nxt += 1
                    ids.append(nxt)
        else:
            ids = [int(record.id) for record in records]
        prefix = "INSERT INTO files (id, path, size_bytes, mtime_ns, sha256, role, language, is_test, is_generated) VALUES "
        suffix = (
            " ON CONFLICT(path) DO UPDATE SET"
            " size_bytes = excluded.size_bytes,"
            " mtime_ns = excluded.mtime_ns,"
            " sha256 = excluded.sha256,"
            " role = excluded.role,"
            " language = excluded.language,"
            " is_test = excluded.is_test,"
            " is_generated = excluded.is_generated"
            " RETURNING id, path"
        )
        by_path: dict[str, int] = {}
        start = 0
        while start < len(records):
            parts: list[str] = []
            end = start
            budget = 2_000_000
            while end < len(records) and end - start < 256:
                record = records[end]
                literals = (
                    _sql_literal(ids[end]),
                    _sql_literal(resolved[end]),
                    _sql_literal(record.size_bytes),
                    _sql_literal(record.mtime_ns),
                    _sql_literal(record.sha256),
                    _sql_literal(record.role.value),
                    _sql_literal(record.language),
                    _sql_literal(record.is_test),
                    _sql_literal(record.is_generated),
                )
                candidate = sum(len(item) for item in literals) + 2 * 9
                if parts and candidate > budget:
                    break
                budget -= candidate
                parts.append("(" + ",".join(literals) + ")")
                end += 1
            for row_id, row_path in self._conn.execute(prefix + ",".join(parts) + suffix).fetchall():
                by_path[str(row_path)] = int(row_id)
            start = end
        return [by_path[path] for path in resolved]

    def delete_file(self, file_id: int) -> list[int]:
        labels = [int(row[0]) for row in self._conn.execute("SELECT label FROM chunks WHERE file_id = ?", [file_id]).fetchall()]
        if labels:
            self._conn.execute("DELETE FROM terms WHERE label IN (SELECT UNNEST(?))", [labels])
            self._conn.execute("DELETE FROM chunks WHERE label IN (SELECT UNNEST(?))", [labels])
        self._conn.execute("DELETE FROM files WHERE id = ?", [file_id])
        return labels

    def insert_chunks(self, records: list[ChunkRecord]) -> None:
        if not records:
            return
        self._insert_rows(
            "INSERT INTO chunks (label, file_id, text, start_char, end_char, token_count, line_start, line_end, kind)",
            9,
            [(r.label, r.file_id, r.text, r.start_char, r.end_char, r.token_count, r.line_start, r.line_end, r.kind) for r in records],
        )

    def insert_terms(self, records: list[TermRecord]) -> None:
        """Insert term rows for fresh labels, replacing any rows already stored for those labels.

        DuckDB's ON CONFLICT / executemany paths are pathologically slow (~ms per row), so
        replacement is implemented as a set-based DELETE plus one batched multi-values INSERT,
        which is equivalent on every real write path (labels are always freshly reserved).
        """
        if not records:
            return
        by_key: dict[tuple[int, str, str], TermRecord] = {}
        for record in records:
            by_key[(record.label, record.term, record.field)] = record
        rows = [(r.label, r.term, r.field, r.tf, r.weight) for r in by_key.values()]
        labels = sorted({row[0] for row in rows})
        self._conn.execute("DELETE FROM terms WHERE label IN (SELECT UNNEST(?))", [labels])
        self._insert_rows("INSERT INTO terms (label, term, field, tf, weight)", 5, rows)

    def _insert_rows(
        self,
        sql_prefix: str,
        num_columns: int,
        rows: list[tuple[Any, ...]],
        *,
        max_values_per_statement: int = 512,
        max_sql_chars: int = 2_000_000,
    ) -> None:
        """Insert many rows with literal multi-values statements.

        DuckDB's `?`-parameter binding and executemany cost ~0.3-2ms per row; inlined
        literals are ~50x faster. Strings are escaped by doubling single quotes (verified:
        DuckDB single-quoted strings have no backslash escapes). Statements are split both
        by row count and by rendered size to bound parser memory.
        """
        if num_columns <= 0 or not rows:
            return
        start = 0
        while start < len(rows):
            parts: list[str] = []
            end = start
            budget = max_sql_chars
            while end < len(rows) and end - start < max_values_per_statement:
                literals = [_sql_literal(value) for value in rows[end]]
                candidate = len(literals) + sum(len(item) for item in literals) + 2 * num_columns
                if parts and candidate > budget:
                    break
                budget -= candidate
                parts.append("(" + ",".join(literals) + ")")
                end += 1
            self._conn.execute(f"{sql_prefix} VALUES {','.join(parts)}")
            start = end

    def refresh_term_stats(self) -> None:
        self._conn.execute("DELETE FROM term_stats")
        self._conn.execute(
            """
            INSERT INTO term_stats (term, chunk_df, path_df, symbol_df)
            SELECT
                term,
                COUNT(DISTINCT CASE WHEN field = 'chunk' THEN label END),
                COUNT(DISTINCT CASE WHEN field = 'path' THEN label END),
                COUNT(DISTINCT CASE WHEN field = 'symbol' THEN label END)
            FROM terms
            GROUP BY term
            """
        )

    def lookup_chunks(self, labels: list[int], filters: ResultFilters) -> list[dict[str, Any]]:
        if not labels:
            return []
        rows = self._conn.execute(
            """
            SELECT c.label, c.file_id, f.path, c.text, c.start_char, c.end_char, c.token_count,
                   c.line_start, c.line_end, f.role, f.language
            FROM chunks c
            JOIN files f ON f.id = c.file_id
            WHERE c.label IN (SELECT UNNEST(?))
            """,
            [labels],
        ).fetchall()
        by_rank = {label: idx for idx, label in enumerate(labels)}
        details = [self._row_to_detail(row) for row in rows]
        prepared = _PreparedFilters.from_result_filters(filters)
        details = [row for row in details if _row_passes_filters(row, prepared)]
        return sorted(details, key=lambda row: by_rank.get(int(row["label"]), 10**12))

    def lexical_candidates(self, query_terms: list[str], limit: int, filters: ResultFilters) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT c.label, c.file_id, f.path, c.text, c.start_char, c.end_char, c.token_count,
                   c.line_start, c.line_end, f.role, f.language, SUM(t.tf * t.weight) AS lexical_score
            FROM terms t
            JOIN chunks c ON c.label = t.label
            JOIN files f ON f.id = c.file_id
            WHERE t.term IN (SELECT UNNEST(?))
            GROUP BY c.label, c.file_id, f.path, c.text, c.start_char, c.end_char, c.token_count, c.line_start, c.line_end, f.role, f.language
            ORDER BY lexical_score DESC, c.label ASC
            LIMIT ?
            """,
            [query_terms, limit * 3],
        ).fetchall()
        details = [self._row_to_detail(row, lexical_score=float(row[11])) for row in rows]
        prepared = _PreparedFilters.from_result_filters(filters)
        details = [row for row in details if _row_passes_filters(row, prepared)]
        return details[:limit]

    def counts(self, project_name: str = "") -> ProjectStatus:
        files_row = self._conn.execute("SELECT COUNT(*) FROM files").fetchone()
        chunks_row = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        assert files_row is not None
        assert chunks_row is not None
        files = int(files_row[0])
        chunks = int(chunks_row[0])
        state_value = self.get_meta("index_state")
        state = IndexState(state_value) if state_value else None
        return ProjectStatus(project_name=project_name, files_count=files, chunks_count=chunks, index_exists=chunks > 0, index_state=state)

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", [key]).fetchone()
        return None if row is None else str(row[0])

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            """
            INSERT INTO meta (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            [key, value],
        )

    @staticmethod
    def _row_to_detail(row: tuple[Any, ...], lexical_score: float | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "label": int(row[0]),
            "usearch_label": int(row[0]),
            "file_id": int(row[1]),
            "file_path": Path(str(row[2])),
            "chunk_text": str(row[3]),
            "start_char": int(row[4]),
            "end_char": int(row[5]),
            "start_char_offset": int(row[4]),
            "end_char_offset": int(row[5]),
            "token_count": int(row[6]),
            "line_start": None if row[7] is None else int(row[7]),
            "line_end": None if row[8] is None else int(row[8]),
            "file_role": str(row[9]),
            "language": str(row[10]),
        }
        if lexical_score is not None:
            out["lexical_score"] = lexical_score
        return out

    def to_search_results(self, labels: list[int], filters: ResultFilters) -> list[SearchResult]:
        rows = self.lookup_chunks(labels, filters)
        return [
            SearchResult(
                label=int(row["label"]),
                score=0.0,
                file_path=Path(row["file_path"]),
                chunk_text=str(row["chunk_text"]),
                start_char=int(row["start_char"]),
                end_char=int(row["end_char"]),
                line_start=row["line_start"],
                line_end=row["line_end"],
                file_role=FileRole(str(row["file_role"])),
                language=str(row["language"]),
            )
            for row in rows
        ]


@dataclass(frozen=True)
class _PreparedFilters:
    """Per-query filter state with every syscall hoisted out of the row loop.

    Stored file paths are canonical (resolved at upsert time), so rows need no
    ``resolve()``; the scope is resolved once here instead of once per row.
    """

    scope: Path | None
    scope_is_file: bool
    file_filter: tuple[str, ...] | None
    include_globs: tuple[str, ...] | None
    exclude_globs: tuple[str, ...] | None
    keyword: str | None

    @staticmethod
    def from_result_filters(filters: ResultFilters) -> "_PreparedFilters":
        scope = filters.scope_path.resolve() if filters.scope_path is not None else None
        return _PreparedFilters(
            scope=scope,
            scope_is_file=scope.is_file() if scope is not None else False,
            file_filter=filters.file_filter,
            include_globs=filters.include_globs,
            exclude_globs=filters.exclude_globs,
            keyword=filters.keyword_filter.lower() if filters.keyword_filter else None,
        )


def _row_passes_filters(row: dict[str, Any], prepared: _PreparedFilters) -> bool:
    path = row["file_path"]
    if not isinstance(path, Path):
        path = Path(str(path))
    if prepared.scope is not None:
        if prepared.scope_is_file:
            if path != prepared.scope:
                return False
        else:
            try:
                path.relative_to(prepared.scope)
            except ValueError:
                return False
    path_str = path.as_posix()
    path_name = path.name
    if prepared.file_filter and not any(fnmatch.fnmatch(path_str, pat) or fnmatch.fnmatch(path_name, pat) for pat in prepared.file_filter):
        return False
    if prepared.include_globs and not any(fnmatch.fnmatch(path_str, pat) or fnmatch.fnmatch(path_name, pat) for pat in prepared.include_globs):
        return False
    if prepared.exclude_globs and any(fnmatch.fnmatch(path_str, pat) or fnmatch.fnmatch(path_name, pat) for pat in prepared.exclude_globs):
        return False
    if prepared.keyword and prepared.keyword not in str(row["chunk_text"]).lower():
        return False
    return True
