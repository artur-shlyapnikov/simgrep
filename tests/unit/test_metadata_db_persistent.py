import pathlib
from typing import Dict, List, Tuple

import duckdb
import pytest

# Ensure consistent exception type
from simgrep.metadata_db import MetadataDBError, connect_persistent_db


@pytest.fixture
def persistent_db_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """provides a path for a persistent DB file within the temp directory."""
    return tmp_path / "persistent_dbs" / "test_simgrep.duckdb"


def get_table_schema(
    conn: duckdb.DuckDBPyConnection, table_name: str
) -> Dict[str, str]:
    """helper to get table schema (column name -> type)."""
    # duckdb's describe returns more info; pragma table_info is sqlite-like.
    # for duckdb, using describe is idiomatic.
    columns_info = conn.execute(f"DESCRIBE {table_name};").fetchall()
    # example row from describe: ('file_id', 'INTEGER', 'NO', None, None, 'YES')
    # (column_name, column_type, null, key, default, extra)
    return {info[0]: info[1] for info in columns_info}


def get_table_constraints(
    conn: duckdb.DuckDBPyConnection, table_name: str
) -> List[Tuple[str, str]]:
    """
    rudimentary way to check for constraints like primary key and unique.
    duckdb's show table_name or pragma show_tables_expanded might be better,
    but describe gives some clues (e.g. 'key' column for primary/unique keys).
    this is a simplified check.
    """
    columns_info = conn.execute(f"DESCRIBE {table_name};").fetchall()
    constraints = []
    for info in columns_info:
        column_name, _type, _null, key_info, _default, _extra = info
        if key_info == "PRI":
            constraints.append((column_name, "PRIMARY KEY"))
        elif (
            key_info == "UNI"
        ):  # note: describe might not always show 'UNI' explicitly for all unique constraints.
            # more robust check might involve querying system catalogs if available/needed.
            constraints.append((column_name, "UNIQUE"))

    # check for unique constraint on indexed_files.file_path
    if table_name == "indexed_files":
        # this is harder to get directly from describe for non-primary key unique constraints.
        # for this test, we'll rely on insertion failures for unique constraint violations.
        pass
    if table_name == "text_chunks":
        # usearch_label bigint unique not null
        pass  # also rely on insertion failures.
    return constraints


class TestPersistentMetadataDB:

    def test_connect_persistent_db_new_creation(
        self, persistent_db_path: pathlib.Path
    ) -> None:
        """test creating a new persistent DB: file and tables are created."""
        assert not persistent_db_path.exists()
        assert not persistent_db_path.parent.exists()

        conn = None
        try:
            conn = connect_persistent_db(persistent_db_path)
            assert isinstance(conn, duckdb.DuckDBPyConnection)
            assert persistent_db_path.exists()
            assert persistent_db_path.is_file()
            assert persistent_db_path.parent.exists()  # check directory creation

            # verify tables exist
            tables = conn.execute("SHOW TABLES;").fetchall()
            table_names = [table[0] for table in tables]
            assert "indexed_files" in table_names
            assert "text_chunks" in table_names

            # verify indexed_files schema
            indexed_files_schema = get_table_schema(conn, "indexed_files")
            assert (
                indexed_files_schema.get("file_id") == "BIGINT"
            )  # pk (bigserial becomes bigint)
            assert indexed_files_schema.get("file_path") == "VARCHAR"  # not null unique
            assert indexed_files_schema.get("content_hash") == "VARCHAR"  # not null
            assert indexed_files_schema.get("file_size_bytes") == "BIGINT"
            assert indexed_files_schema.get("last_modified_os") == "TIMESTAMP"
            assert (
                indexed_files_schema.get("last_indexed_at") == "TIMESTAMP"
            )  # not null default

            # verify text_chunks schema
            text_chunks_schema = get_table_schema(conn, "text_chunks")
            assert (
                text_chunks_schema.get("chunk_id") == "BIGINT"
            )  # pk (bigserial becomes bigint)
            assert text_chunks_schema.get("file_id") == "BIGINT"  # not null fk
            assert (
                text_chunks_schema.get("usearch_label") == "BIGINT"
            )  # not null unique
            assert text_chunks_schema.get("chunk_text_snippet") == "VARCHAR"  # not null
            assert text_chunks_schema.get("start_char_offset") == "INTEGER"  # not null
            assert text_chunks_schema.get("end_char_offset") == "INTEGER"  # not null
            assert text_chunks_schema.get("token_count") == "INTEGER"  # not null
            assert text_chunks_schema.get("embedding_hash") == "VARCHAR"  # nullable

        finally:
            if conn:
                conn.close()

    def test_connect_persistent_db_existing_db(
        self, persistent_db_path: pathlib.Path
    ) -> None:
        """test connecting to an existing DB: tables are still there, no errors."""
        # first, create the db
        conn1 = None
        try:
            conn1 = connect_persistent_db(persistent_db_path)
            # add some dummy data to ensure it persists
            conn1.execute(
                "INSERT INTO indexed_files (file_path, content_hash, file_size_bytes) VALUES (?, ?, ?)",
                ["/test/file.txt", "hash123", 100],
            )
            conn1.commit()  # duckdb auto-commits by default unless in explicit transaction
        finally:
            if conn1:
                conn1.close()

        assert persistent_db_path.exists()

        # connect again
        conn2 = None
        try:
            conn2 = connect_persistent_db(persistent_db_path)
            tables = conn2.execute("SHOW TABLES;").fetchall()
            table_names = [table[0] for table in tables]
            assert "indexed_files" in table_names
            assert "text_chunks" in table_names

            # check if data is still there
            result = conn2.execute(
                "SELECT COUNT(*) FROM indexed_files WHERE file_path = '/test/file.txt';"
            ).fetchone()
            assert result is not None
            count = result[0]
            assert count == 1
        finally:
            if conn2:
                conn2.close()

    def test_connect_persistent_db_directory_creation_failure(
        self, persistent_db_path: pathlib.Path
    ) -> None:
        """test handling of OSError if DB directory creation fails."""
        # to simulate this, we make the parent of the db_path a file, so mkdir fails.
        parent_dir_of_db_parent = persistent_db_path.parent.parent
        parent_dir_of_db_parent.mkdir(parents=True, exist_ok=True)

        path_that_should_be_dir = persistent_db_path.parent  # e.g., .../persistent_dbs/
        path_that_should_be_dir.touch()  # create it as a file

        with pytest.raises(
            MetadataDBError,
            match=f"Could not create directory for database at {str(path_that_should_be_dir)}",
        ):
            connect_persistent_db(persistent_db_path)

    def test_data_persistence_and_fk_constraint(
        self, persistent_db_path: pathlib.Path
    ) -> None:
        """test inserting data, checking FK constraints, and persistence."""
        conn = None
        try:
            conn = connect_persistent_db(persistent_db_path)

            # insert into indexed_files
            file_path1 = "/path/to/file1.txt"
            content_hash1 = "abc"
            conn.execute(
                "INSERT INTO indexed_files "
                "(file_path, content_hash, file_size_bytes, last_modified_os) "
                "VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                [file_path1, content_hash1, 1024],
            )
            file1_id_result = conn.execute(
                "SELECT file_id FROM indexed_files WHERE file_path = ?", [file_path1]
            ).fetchone()
            assert file1_id_result is not None
            file1_id = file1_id_result[0]

            # insert into text_chunks referencing file1_id
            conn.execute(
                "INSERT INTO text_chunks "
                "(file_id, usearch_label, chunk_text_snippet, "
                "start_char_offset, end_char_offset, token_count) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [file1_id, 1001, "snippet1", 0, 10, 5],
            )

            # try inserting a chunk with a non-existent file_id (fk violation)
            with pytest.raises(
                duckdb.ConstraintException,
                match=(
                    r"foreign key constraint|violates foreign key constraint|"
                    r"FOREIGN KEY constraint failed|Violates foreign key constraint"
                ),
            ):
                conn.execute(
                    "INSERT INTO text_chunks (file_id, usearch_label, chunk_text_snippet, "
                    "start_char_offset, end_char_offset, token_count) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    [9999, 1002, "snippet2", 0, 10, 5],
                )  # 9999 is a non-existent file_id

            # try inserting duplicate file_path (unique constraint violation)
            with pytest.raises(
                duckdb.ConstraintException,
                match=(
                    r"UNIQUE constraint failed: indexed_files.file_path|"
                    r"violates unique constraint|duplicate key|Duplicate key"
                ),
            ):
                conn.execute(
                    "INSERT INTO indexed_files (file_path, content_hash, file_size_bytes) "
                    "VALUES (?, ?, ?)",
                    [file_path1, "def", 2048],
                )  # same file_path1

            # try inserting duplicate usearch_label (unique constraint violation)
            with pytest.raises(
                duckdb.ConstraintException,
                match=(
                    r"UNIQUE constraint failed: text_chunks.usearch_label|"
                    r"violates unique constraint|duplicate key|Duplicate key"
                ),
            ):
                insert_sql = (
                    "INSERT INTO text_chunks (file_id, usearch_label, chunk_text_snippet, "
                    "start_char_offset, end_char_offset, token_count) "
                    "VALUES (?, ?, ?, ?, ?, ?)"
                )
                conn.execute(insert_sql, [file1_id, 1001, "snippet3", 10, 20, 6])
        finally:
            if conn:
                conn.close()

        # reconnect and verify data
        conn_reopened = None
        try:
            conn_reopened = connect_persistent_db(persistent_db_path)
            file_count_result = conn_reopened.execute(
                "SELECT COUNT(*) FROM indexed_files"
            ).fetchone()
            assert file_count_result is not None
            file_count = file_count_result[0]

            chunk_count_result = conn_reopened.execute(
                "SELECT COUNT(*) FROM text_chunks"
            ).fetchone()
            assert chunk_count_result is not None
            chunk_count = chunk_count_result[0]

            assert file_count == 1
            assert chunk_count == 1

            retrieved_snippet_result = conn_reopened.execute(
                "SELECT chunk_text_snippet FROM text_chunks WHERE usearch_label = ?",
                [1001],
            ).fetchone()
            assert retrieved_snippet_result is not None
            retrieved_snippet = retrieved_snippet_result[0]
            assert retrieved_snippet == "snippet1"
        finally:
            if conn_reopened:
                conn_reopened.close()
