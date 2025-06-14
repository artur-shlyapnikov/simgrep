import pathlib

import numpy as np
import pytest
from rich.console import Console

from simgrep.core.context import SimgrepContext
from simgrep.core.models import ProcessedChunk
from simgrep.indexer import Indexer, IndexerConfig
from simgrep.services.index_service import IndexService

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
        max_index_workers=1,
    )


@pytest.fixture
def simgrep_context(indexer_config: IndexerConfig) -> SimgrepContext:
    return SimgrepContext.from_defaults(
        model_name=indexer_config.embedding_model_name,
        chunk_size=indexer_config.chunk_size_tokens,
        chunk_overlap=indexer_config.chunk_overlap_tokens,
    )


@pytest.fixture
def indexer_instance(indexer_config: IndexerConfig, simgrep_context: SimgrepContext) -> Indexer:
    return Indexer(config=indexer_config, context=simgrep_context, console=Console(quiet=True))


def test_indexer_prepare_stores(tmp_path: pathlib.Path, indexer_instance: Indexer) -> None:
    """Test that the data stores are prepared correctly."""
    indexer_instance._prepare_data_stores(wipe_existing=True)
    assert indexer_instance.metadata_store is not None
    assert indexer_instance.usearch_index is not None
    assert indexer_instance.config.db_path.exists()
    # Index file is only created on save, so we don't check for its existence here.
    indexer_instance.metadata_store.close()


def test_index_service_process_file(simgrep_context: SimgrepContext, tmp_path: pathlib.Path) -> None:
    """Test the process_file method of IndexService directly."""
    # This is more of an integration test for the service, but useful here.
    service = IndexService(
        extractor=simgrep_context.extractor,
        chunker=simgrep_context.chunker,
        embedder=simgrep_context.embedder,
        store=None,  # type: ignore
        index=None,  # type: ignore
    )
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Hello world.")
    chunks, embeddings = service.process_file(file_path)
    assert len(chunks) == 1
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(chunks)
    assert embeddings.shape[1] == service.embedder.ndim
