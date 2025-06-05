import pathlib
import pytest
from rich.console import Console

pytest.importorskip("duckdb")
pytest.importorskip("transformers")
pytest.importorskip("sentence_transformers")
pytest.importorskip("usearch.index")

from .test_indexer_persistent import sample_files_dir, test_console

from simgrep.indexer import Indexer, IndexerConfig
from simgrep.metadata_db import connect_persistent_db
from simgrep.vector_store import load_persistent_index


def _make_config(base_dir: pathlib.Path, project: str) -> IndexerConfig:
    return IndexerConfig(
        project_name=project,
        db_path=base_dir / "metadata.duckdb",
        usearch_index_path=base_dir / "index.usearch",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size_tokens=128,
        chunk_overlap_tokens=20,
        file_scan_patterns=["*.txt", "*.md"],
    )


class TestIndexerParallel:
    def test_parallel_matches_sequential(
        self,
        sample_files_dir: pathlib.Path,
        test_console: Console,
        tmp_path: pathlib.Path,
    ) -> None:
        seq_dir = tmp_path / "seq"
        conc_dir = tmp_path / "conc"
        seq_dir.mkdir()
        conc_dir.mkdir()

        seq_cfg = _make_config(seq_dir, "seq_project")
        conc_cfg = _make_config(conc_dir, "conc_project")

        seq_indexer = Indexer(config=seq_cfg, console=test_console)
        seq_indexer.index_path(target_path=sample_files_dir, wipe_existing=True, max_workers=1)

        conc_indexer = Indexer(config=conc_cfg, console=test_console)
        conc_indexer.index_path(target_path=sample_files_dir, wipe_existing=True, max_workers=4)

        conn_seq = connect_persistent_db(seq_cfg.db_path)
        conn_conc = connect_persistent_db(conc_cfg.db_path)
        try:
            files_seq = set(row[0] for row in conn_seq.execute("SELECT file_path FROM indexed_files;").fetchall())
            files_conc = set(row[0] for row in conn_conc.execute("SELECT file_path FROM indexed_files;").fetchall())
            assert files_seq == files_conc

            query = (
                "SELECT f.file_path, tc.chunk_text_snippet, tc.start_char_offset, tc.end_char_offset, tc.token_count "
                "FROM text_chunks tc JOIN indexed_files f ON tc.file_id = f.file_id"
            )
            chunks_seq = set(conn_seq.execute(query).fetchall())
            chunks_conc = set(conn_conc.execute(query).fetchall())
            assert chunks_seq == chunks_conc
        finally:
            conn_seq.close()
            conn_conc.close()

        idx_seq = load_persistent_index(seq_cfg.usearch_index_path)
        idx_conc = load_persistent_index(conc_cfg.usearch_index_path)
        assert idx_seq is not None and idx_conc is not None
        assert len(idx_seq) == len(idx_conc)
        assert idx_seq.ndim == idx_conc.ndim
