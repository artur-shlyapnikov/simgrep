import pathlib

import pytest
from rich.console import Console

from simgrep.indexer import Indexer, IndexerConfig
from simgrep.processor import ProcessedChunkInfo

pytest.importorskip("transformers")
pytest.importorskip("sentence_transformers")
pytest.importorskip("usearch.index")


@pytest.fixture
def indexer_config(tmp_path: pathlib.Path) -> IndexerConfig:
    return IndexerConfig(
        project_name="unit_test_project",
        db_path=tmp_path / "meta.duckdb",
        usearch_index_path=tmp_path / "index.usearch",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size_tokens=16,
        chunk_overlap_tokens=0,
        file_scan_patterns=["*.txt"],
    )


@pytest.fixture
def indexer_instance(indexer_config: IndexerConfig) -> Indexer:
    return Indexer(config=indexer_config, console=Console(quiet=True))


def test_extract_and_chunk_file(tmp_path: pathlib.Path, indexer_instance: Indexer) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Hello world. This is a test file.")
    chunks = indexer_instance._extract_and_chunk_file(file_path)
    assert len(chunks) > 0
    assert "text" in chunks[0]


def test_generate_embeddings_for_chunks(indexer_instance: Indexer) -> None:
    chunks = [ProcessedChunkInfo(text="hello", start_char_offset=0, end_char_offset=5, token_count=1)]
    embeddings = indexer_instance._generate_embeddings_for_chunks(chunks)
    assert embeddings.shape[0] == len(chunks)
    assert embeddings.shape[1] == indexer_instance.embedding_ndim


def test_store_processed_chunks(tmp_path: pathlib.Path, indexer_instance: Indexer) -> None:
    indexer_instance._prepare_data_stores(wipe_existing=True)
    assert indexer_instance.metadata_store is not None
    file_id = indexer_instance.metadata_store.insert_indexed_file_record(
        file_path=str(tmp_path / "dummy.txt"),
        content_hash="hash",
        file_size_bytes=5,
        last_modified_os_timestamp=0.0,
    )
    assert file_id is not None

    chunks = [ProcessedChunkInfo(text="hello", start_char_offset=0, end_char_offset=5, token_count=1)]
    embeddings = indexer_instance._generate_embeddings_for_chunks(chunks)
    indexer_instance._store_processed_chunks(file_id, chunks, embeddings)

    assert indexer_instance.usearch_index is not None
    assert len(indexer_instance.usearch_index) == len(chunks)
    assert indexer_instance.metadata_store is not None
    count_row = indexer_instance.metadata_store.conn.execute("SELECT COUNT(*) FROM text_chunks;").fetchone()
    assert count_row is not None and count_row[0] == len(chunks)