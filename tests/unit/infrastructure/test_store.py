from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import duckdb
import pytest

from simgrep.errors import MetadataError
from simgrep.models import ChunkRecord, FileRecord, FileRole, IndexState, ResultFilters, SearchResult, TermRecord
from simgrep.store import Store, _sql_literal


def test_memory_store_roundtrip(tmp_path: Path) -> None:
    file_path = tmp_path / "a.py"
    store = Store.memory()
    file_id = store.upsert_file(FileRecord(id=0, path=file_path, size_bytes=10, mtime_ns=100, sha256="h1", role=FileRole.source, language="python"))
    store.insert_chunks([ChunkRecord(label=7, file_id=file_id, text="alpha beta", start_char=0, end_char=10, token_count=2)])
    store.insert_terms([TermRecord(label=7, term="alpha", field="chunk", tf=1, weight=1.0)])
    store.refresh_term_stats()
    counts = store.counts("p")
    assert counts.files_count == 1
    assert counts.chunks_count == 1
    assert store.lexical_candidates(["alpha"], limit=5, filters=ResultFilters())[0]["label"] == 7
    assert store.lookup_chunks([7], ResultFilters())[0]["file_path"] == file_path.resolve()
    store.close()


def test_reserve_labels_persists_sequence() -> None:
    store = Store.memory()
    assert store.reserve_labels(3) == [0, 1, 2]
    assert store.reserve_labels(2) == [3, 4]
    store.close()


def test_persistent_store_schema_mismatch(tmp_path: Path) -> None:
    db_path = tmp_path / "meta.duckdb"
    store = Store.open(db_path)
    store.set_meta("schema_version", "999")
    store.close()
    with pytest.raises(MetadataError, match="Unsupported schema version"):
        Store.open(db_path)


def test_schema_mismatch_includes_rebuild_hint(tmp_path: Path) -> None:
    db_path = tmp_path / "meta_hint.duckdb"
    store = Store.open(db_path)
    store.set_meta("schema_version", "999")
    store.close()
    with pytest.raises(MetadataError) as exc:
        Store.open(db_path)
    assert exc.value.hint is not None
    assert "index --rebuild" in exc.value.hint


def test_store_sets_schema_and_label_meta_on_open() -> None:
    store = Store.memory()
    assert store.get_meta("schema_version") is not None
    assert store.reserve_labels(1) == [0]
    assert store.get_meta("max_label") == "0"
    store.close()


def test_delete_file_returns_labels(tmp_path: Path) -> None:
    store = Store.memory()
    file_id = store.upsert_file(FileRecord(id=3, path=tmp_path / "c.py", size_bytes=1, mtime_ns=1))
    store.insert_chunks([ChunkRecord(label=11, file_id=file_id, text="x", start_char=0, end_char=1, token_count=1)])
    assert store.delete_file(file_id) == [11]
    assert store.counts().chunks_count == 0
    store.close()


def test_read_only_open_without_file_returns_metadata_error_with_hint(tmp_path: Path) -> None:
    db_path = tmp_path / "missing.duckdb"
    with pytest.raises(MetadataError) as exc:
        Store.open(db_path, read_only=True)
    assert exc.value.hint is not None
    assert "index" in exc.value.hint.lower()


def test_read_only_open_missing_meta_schema_returns_controlled_error(tmp_path: Path) -> None:
    db_path = tmp_path / "no_meta.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("CREATE TABLE files (id BIGINT)")
    conn.close()

    with pytest.raises(MetadataError, match="Missing simgrep metadata schema"):
        Store.open(db_path, read_only=True)


def test_old_schema_markers_return_migration_rebuild_hint(tmp_path: Path) -> None:
    db_path = tmp_path / "old_schema.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("CREATE TABLE indexed_files (id BIGINT)")
    conn.close()

    with pytest.raises(MetadataError, match="Old simgrep metadata schema") as exc:
        Store.open(db_path)
    assert exc.value.hint is not None
    assert "rebuild" in exc.value.hint.lower()


def test_invalid_schema_version_value_returns_metadata_error_not_value_error(tmp_path: Path) -> None:
    db_path = tmp_path / "bad_schema_version.duckdb"
    store = Store.open(db_path)
    store.set_meta("schema_version", "not-an-int")
    store.close()

    with pytest.raises(MetadataError):
        Store.open(db_path)


def test_clear_removes_files_chunks_terms_term_stats_and_resets_labels(tmp_path: Path) -> None:
    store = Store.memory()
    file_id = store.upsert_file(FileRecord(id=0, path=tmp_path / "clear.py", size_bytes=1, mtime_ns=1))
    store.insert_chunks([ChunkRecord(label=5, file_id=file_id, text="alpha", start_char=0, end_char=5, token_count=1)])
    store.insert_terms([TermRecord(label=5, term="alpha", field="chunk", tf=1, weight=1.0)])
    store.refresh_term_stats()
    assert store.reserve_labels(1) == [0]

    store.clear()

    assert store.counts().files_count == 0
    assert store.counts().chunks_count == 0
    terms_count = store._conn.execute("SELECT COUNT(*) FROM terms").fetchone()
    term_stats_count = store._conn.execute("SELECT COUNT(*) FROM term_stats").fetchone()
    assert terms_count is not None and term_stats_count is not None
    assert terms_count[0] == 0
    assert term_stats_count[0] == 0
    assert store.reserve_labels(1) == [0]
    store.close()


def test_upsert_file_preserves_existing_id_on_path_update(tmp_path: Path) -> None:
    store = Store.memory()
    path = tmp_path / "same.py"
    first_id = store.upsert_file(FileRecord(id=0, path=path, size_bytes=10, mtime_ns=100))
    second_id = store.upsert_file(FileRecord(id=999, path=path, size_bytes=11, mtime_ns=101))
    assert second_id == first_id
    assert second_id != 999
    store.close()


def test_delete_file_deletes_chunks_and_terms_cascade(tmp_path: Path) -> None:
    store = Store.memory()
    file_id = store.upsert_file(FileRecord(id=0, path=tmp_path / "cascade.py", size_bytes=1, mtime_ns=1))
    store.insert_chunks([ChunkRecord(label=21, file_id=file_id, text="x", start_char=0, end_char=1, token_count=1)])
    store.insert_terms([TermRecord(label=21, term="alpha", field="chunk", tf=1, weight=1.0)])
    labels = store.delete_file(file_id)
    assert labels == [21]
    files_count = store._conn.execute("SELECT COUNT(*) FROM files").fetchone()
    chunks_count = store._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
    terms_count = store._conn.execute("SELECT COUNT(*) FROM terms").fetchone()
    assert files_count is not None and chunks_count is not None and terms_count is not None
    assert files_count[0] == 0
    assert chunks_count[0] == 0
    assert terms_count[0] == 0
    store.close()


def test_lexical_candidates_respect_term_weights_symbol_over_chunk_over_path(tmp_path: Path) -> None:
    store = Store.memory()
    file_id = store.upsert_file(FileRecord(id=0, path=tmp_path / "weights.py", size_bytes=1, mtime_ns=1))
    store.insert_chunks(
        [
            ChunkRecord(label=1, file_id=file_id, text="alpha symbol", start_char=0, end_char=12, token_count=2),
            ChunkRecord(label=2, file_id=file_id, text="alpha chunk", start_char=0, end_char=11, token_count=2),
            ChunkRecord(label=3, file_id=file_id, text="alpha path", start_char=0, end_char=10, token_count=2),
        ]
    )
    store.insert_terms(
        [
            TermRecord(label=1, term="alpha", field="symbol", tf=1, weight=3.0),
            TermRecord(label=2, term="alpha", field="chunk", tf=1, weight=2.0),
            TermRecord(label=3, term="alpha", field="path", tf=1, weight=1.0),
        ]
    )
    rows = store.lexical_candidates(["alpha"], limit=3, filters=ResultFilters())
    assert [int(row["label"]) for row in rows] == [1, 2, 3]
    store.close()


def test_lexical_candidates_limit_applies_after_filters(tmp_path: Path) -> None:
    store = Store.memory()
    file_id = store.upsert_file(FileRecord(id=0, path=tmp_path / "limit.py", size_bytes=1, mtime_ns=1))
    chunks = [ChunkRecord(label=i, file_id=file_id, text=f"alpha {i}", start_char=0, end_char=7, token_count=2) for i in range(10)]
    store.insert_chunks(chunks)
    store.insert_terms([TermRecord(label=i, term="alpha", field="chunk", tf=1, weight=1.0) for i in range(10)])
    rows = store.lexical_candidates(["alpha"], limit=3, filters=ResultFilters(keyword_filter="ALPHA"))
    assert len(rows) <= 3
    store.close()


def test_keyword_filter_is_case_insensitive(tmp_path: Path) -> None:
    store = Store.memory()
    file_id = store.upsert_file(FileRecord(id=0, path=tmp_path / "keyword.py", size_bytes=1, mtime_ns=1))
    store.insert_chunks([ChunkRecord(label=41, file_id=file_id, text="AlphaBeta", start_char=0, end_char=9, token_count=1)])
    store.insert_terms([TermRecord(label=41, term="alpha", field="chunk", tf=1, weight=1.0)])
    rows = store.lexical_candidates(["alpha"], limit=5, filters=ResultFilters(keyword_filter="betA"))
    assert [int(row["label"]) for row in rows] == [41]
    store.close()


def test_filters_match_basename_and_full_path(tmp_path: Path) -> None:
    store = Store.memory()
    nested = tmp_path / "src" / "pkg"
    nested.mkdir(parents=True)
    path = nested / "module.py"
    file_id = store.upsert_file(FileRecord(id=0, path=path, size_bytes=1, mtime_ns=1))
    store.insert_chunks([ChunkRecord(label=51, file_id=file_id, text="alpha", start_char=0, end_char=5, token_count=1)])
    store.insert_terms([TermRecord(label=51, term="alpha", field="chunk", tf=1, weight=1.0)])

    basename_filter = ResultFilters(file_filter=("*.py",))
    assert [int(row["label"]) for row in store.lexical_candidates(["alpha"], limit=5, filters=basename_filter)] == [51]

    full_path_pattern = f"*{nested.as_posix()}*"
    include_filter = ResultFilters(include_globs=(full_path_pattern,))
    assert [int(row["label"]) for row in store.lexical_candidates(["alpha"], limit=5, filters=include_filter)] == [51]

    exclude_by_basename = ResultFilters(exclude_globs=("module.py",))
    assert store.lexical_candidates(["alpha"], limit=5, filters=exclude_by_basename) == []
    store.close()


def test_counts_reflect_index_state_meta() -> None:
    store = Store.memory()
    initial = store.counts("p")
    assert initial.index_state is None
    assert initial.index_exists is False

    store.set_meta("index_state", "indexing")
    indexing = store.counts("p")
    assert indexing.index_state == IndexState.indexing

    store.set_meta("index_state", "ready")
    ready = store.counts("p")
    assert ready.index_state == IndexState.ready
    assert ready.index_exists is False
    store.close()


class StatementSpy:
    """Wraps Store._conn to record executed SQL while delegating transparently."""

    def __init__(self, store: Store) -> None:
        self.sqls: list[str] = []
        original = store._conn

        def execute(sql: str, *args: Any) -> Any:  # type: ignore[no-untyped-def]
            self.sqls.append(sql)
            return original.execute(sql, *args)

        store._conn = _SpyConn(original, execute)  # type: ignore[assignment]


class _SpyConn:
    def __init__(self, original: Any, execute: Any) -> None:
        self._original = original
        self.execute = execute

    def close(self) -> None:
        self._original.close()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "NULL"),
        (True, "TRUE"),
        (False, "FALSE"),
        (0, "0"),
        (1, "1"),
        (-5, "-5"),
        (3.5, "3.5"),
        (0.1, "0.1"),
        (1e308, "1e+308"),
        ("", "''"),
        ("it's", "'it''s'"),
        ("back\\slash", "'back\\slash'"),
        ("line\nbreak", "'line\nbreak'"),
        ("unicode ✓🙂", "'unicode ✓🙂'"),
    ],
)
def test_sql_literal_renders_each_value_partition(value: Any, expected: str) -> None:
    assert _sql_literal(value) == expected


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_sql_literal_casts_non_finite_floats_through_string(value: float) -> None:
    assert _sql_literal(value) == f"'{value}'::DOUBLE"


@pytest.mark.parametrize("value", [b"bytes", [1, 2], {"k": "v"}, object()])
def test_sql_literal_rejects_unsupported_types_with_metadata_error(value: Any) -> None:
    with pytest.raises(MetadataError, match="Unsupported literal type"):
        _sql_literal(value)


@pytest.mark.parametrize("n", [1, 511, 512, 513])
def test_insert_terms_preserves_every_row_across_statement_boundaries(n: int) -> None:
    store = Store.memory()
    try:
        records = [TermRecord(label=i, term=f"t{i}", field="chunk", tf=i, weight=float(i)) for i in range(1, n + 1)]
        store.insert_terms(records)
        row = store._conn.execute("SELECT COUNT(*), MAX(tf) FROM terms").fetchone()
        assert row is not None
        count, max_tf = row
        assert (count, max_tf) == (n, n)
    finally:
        store.close()


def test_insert_terms_splits_huge_rows_into_multiple_statements_within_char_budget() -> None:
    store = Store.memory()
    spy = StatementSpy(store)
    try:
        big = "x" * 1_100_000  # ~1.1MB rendered per row: > half of max_sql_chars, forces one row per statement
        records = [TermRecord(label=i, term=big, field="chunk", tf=i, weight=1.0) for i in range(1, 4)]
        store.insert_terms(records)
        inserts = [sql for sql in spy.sqls if sql.startswith("INSERT INTO terms")]
        assert len(inserts) > 1
        assert all(len(sql) < 2_000_000 for sql in inserts)
        count_row = store._conn.execute("SELECT COUNT(*) FROM terms").fetchone()
        assert count_row is not None
        assert count_row[0] == 3
    finally:
        store.close()


def test_adversarial_chunk_and_term_text_roundtrip_exactly() -> None:
    store = Store.memory()
    try:
        weird = "quote' double'' back\\slash \n newline \t tab ✓🙂 'mixed'\\"
        file_id = store.upsert_file(FileRecord(id=0, path=Path("weird.py"), size_bytes=1, mtime_ns=1))
        store.insert_chunks([ChunkRecord(label=99, file_id=file_id, text=weird, start_char=0, end_char=len(weird), token_count=3)])
        store.insert_terms([TermRecord(label=99, term=weird[:20], field="chunk", tf=1, weight=-0.5)])
        chunk_row = store._conn.execute("SELECT text FROM chunks WHERE label=99").fetchone()
        term_row = store._conn.execute("SELECT term, weight FROM terms WHERE label=99").fetchone()
        assert chunk_row == (weird,)
        assert term_row == (weird[:20], -0.5)
    finally:
        store.close()


def test_insert_terms_deduplicates_by_label_term_field_keeping_last() -> None:
    store = Store.memory()
    try:
        store.insert_terms(
            [
                TermRecord(label=7, term="alpha", field="chunk", tf=1, weight=1.0),
                TermRecord(label=7, term="alpha", field="chunk", tf=9, weight=9.0),
                TermRecord(label=7, term="alpha", field="symbol", tf=2, weight=2.0),
            ]
        )
        rows = store._conn.execute("SELECT term, field, tf, weight FROM terms WHERE label=7 ORDER BY field").fetchall()
        assert rows == [("alpha", "chunk", 9, 9.0), ("alpha", "symbol", 2, 2.0)]
    finally:
        store.close()


def test_insert_terms_removes_all_stored_rows_for_label_before_inserting() -> None:
    """Pins the set-based replace semantics: rows for the label that are absent from the new
    batch are deleted (old INSERT OR REPLACE left them in place). Unreachable in production
    today -- labels are always freshly reserved -- but pinned so future callers cannot
    silently accumulate stale terms."""
    store = Store.memory()
    try:
        store.insert_terms([TermRecord(label=7, term="old", field="chunk", tf=1, weight=1.0)])
        store.insert_terms([TermRecord(label=7, term="new", field="chunk", tf=1, weight=1.0)])
        assert store._conn.execute("SELECT term FROM terms WHERE label=7").fetchall() == [("new",)]
    finally:
        store.close()


@pytest.mark.parametrize(
    ("weight", "matches"),
    [(math.inf, lambda w: w == math.inf), (-math.inf, lambda w: w == -math.inf), (math.nan, math.isnan)],
)
def test_non_finite_weights_roundtrip_through_terms(weight: float, matches: Any) -> None:
    store = Store.memory()
    try:
        store.insert_terms([TermRecord(label=1, term="w", field="chunk", tf=1, weight=weight)])
        weight_row = store._conn.execute("SELECT weight FROM terms WHERE label=1").fetchone()
        assert weight_row is not None
        stored = weight_row[0]
        assert isinstance(stored, float) and matches(stored)
    finally:
        store.close()


@pytest.mark.parametrize("which", ["terms", "chunks"])
def test_empty_batch_public_inserts_are_noops(which: str) -> None:
    store = Store.memory()
    try:
        if which == "terms":
            store.insert_terms([])
        else:
            store.insert_chunks([])
        total_row = store._conn.execute("SELECT (SELECT COUNT(*) FROM terms) + (SELECT COUNT(*) FROM chunks)").fetchone()
        assert total_row is not None
        total = total_row[0]
        assert total == 0
    finally:
        store.close()


def test_open_unreadable_db_path_raises_metadata_error(tmp_path: Path) -> None:
    with pytest.raises(MetadataError, match="Failed to open metadata database"):
        Store.open(tmp_path)


def test_read_only_open_without_schema_version_meta_raises(tmp_path: Path) -> None:
    db_path = tmp_path / "meta.duckdb"
    store = Store.open(db_path)
    store._conn.execute("DELETE FROM meta WHERE key = 'schema_version'")
    store.close()
    with pytest.raises(MetadataError, match="Missing schema version"):
        Store.open(db_path, read_only=True)


def test_lookup_filters_reject_rows_failing_predicates(tmp_path: Path) -> None:
    store = Store.memory()
    file_id = store.upsert_file(FileRecord(id=0, path=tmp_path / "mod.py", size_bytes=1, mtime_ns=1))
    store.insert_chunks([ChunkRecord(label=1, file_id=file_id, text="alpha beta", start_char=0, end_char=10, token_count=2)])
    store.insert_terms([TermRecord(label=1, term="alpha", field="chunk", tf=1, weight=1.0)])

    assert store.lookup_chunks([1], ResultFilters(file_filter=("*.md",))) == []
    assert store.lookup_chunks([1], ResultFilters(include_globs=("*.md",))) == []
    assert store.lookup_chunks([1], ResultFilters(exclude_globs=("mod.py",))) == []
    assert store.lookup_chunks([1], ResultFilters(keyword_filter="gamma")) == []
    store.close()


def test_to_search_results_maps_rows_in_rank_order(tmp_path: Path) -> None:
    store = Store.memory()
    first = store.upsert_file(FileRecord(id=0, path=tmp_path / "a.py", size_bytes=1, mtime_ns=1))
    second = store.upsert_file(FileRecord(id=0, path=tmp_path / "b.py", size_bytes=1, mtime_ns=2))
    store.insert_chunks(
        [
            ChunkRecord(label=9, file_id=second, text="beta", start_char=3, end_char=7, token_count=1, line_start=2, line_end=2),
            ChunkRecord(label=4, file_id=first, text="alpha", start_char=0, end_char=5, token_count=1, line_start=1, line_end=1),
        ]
    )
    results = store.to_search_results([4, 9], ResultFilters())
    assert [r.label for r in results] == [4, 9]
    head = results[0]
    assert isinstance(head, SearchResult)
    assert head.score == 0.0
    assert head.file_path == (tmp_path / "a.py").resolve()
    assert head.chunk_text == "alpha"
    assert (head.start_char, head.end_char) == (0, 5)
    assert (head.line_start, head.line_end) == (1, 1)
    assert head.file_role == FileRole.unknown
    store.close()


def test_empty_input_guards_return_empty_without_mutations(tmp_path: Path) -> None:
    store = Store.memory()
    assert store.reserve_labels(0) == []
    assert store.reserve_labels(-2) == []
    assert store.lexical_candidates([], limit=5, filters=ResultFilters()) == []
    assert store.lexical_candidates(["alpha"], limit=0, filters=ResultFilters()) == []
    before = store.counts("p")
    store._insert_rows("INSERT INTO chunks (label) VALUES", 0, [(1,)])
    assert store.counts("p") == before
    store.close()


def test_injection_payload_in_terms_and_labels_is_parameterized() -> None:
    """Label/term values are bound as parameters, never spliced into SQL."""
    store = Store.memory()
    payload = "x' UNION SELECT 1 --"
    file_id = store.upsert_file(FileRecord(id=0, path=Path("a.py"), size_bytes=10, mtime_ns=100, sha256="h", role=FileRole.source, language="python"))
    store.insert_chunks([ChunkRecord(label=1, file_id=file_id, text="alpha", start_char=0, end_char=5, token_count=1)])
    store.insert_terms([TermRecord(label=1, term="alpha", field="chunk", tf=1, weight=1.0)])
    # Injection payloads in the bound lists must not execute or match anything.
    assert store.lexical_candidates(["alpha", payload], limit=5, filters=ResultFilters())[0]["label"] == 1
    assert store.lookup_chunks([1, -9223372036854775808], ResultFilters())[0]["label"] == 1
    store.insert_terms([TermRecord(label=1, term=payload, field="chunk", tf=1, weight=1.0)])
    assert store.lexical_candidates([payload], limit=5, filters=ResultFilters())[0]["label"] == 1
    store.delete_file(file_id)
    assert store.counts("p").chunks_count == 0
    store.close()


def test_open_corrupt_database_raises_metadata_error_with_recovery_hint(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.duckdb"
    db_path.write_bytes(b"not a duckdb file at all")
    with pytest.raises(MetadataError) as excinfo:
        Store.open(db_path)
    assert db_path.name in str(excinfo.value)
    assert excinfo.value.hint is not None
    assert "reset" in excinfo.value.hint
