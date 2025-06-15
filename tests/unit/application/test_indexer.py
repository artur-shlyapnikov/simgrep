import pathlib
from typing import Callable
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from simgrep.core.abstractions import Embedder, TextExtractor, TokenChunker, VectorIndex
from simgrep.core.context import SimgrepContext
from simgrep.indexer import Indexer, IndexerConfig


@pytest.fixture
def fake_context(
    fake_text_extractor: TextExtractor,
    fake_token_chunker: TokenChunker,
    fake_embedder: Embedder,
    fake_vector_index_factory: Callable[[int], VectorIndex],
) -> SimgrepContext:
    return SimgrepContext(
        extractor=fake_text_extractor,
        chunker=fake_token_chunker,
        embedder=fake_embedder,
        index_factory=lambda ndim: fake_vector_index_factory(ndim),
    )


@pytest.fixture
def indexer_config(tmp_path: pathlib.Path) -> IndexerConfig:
    return IndexerConfig(
        project_name="unit_test_project",
        db_path=tmp_path / "meta.duckdb",
        usearch_index_path=tmp_path / "index.usearch",
        embedding_model_name="fake-model",
        chunk_size_tokens=16,
        chunk_overlap_tokens=0,
        file_scan_patterns=["*.txt"],
        max_index_workers=1,
    )


def test_indexer_run_index(indexer_config: IndexerConfig, fake_context: SimgrepContext, tmp_path: pathlib.Path) -> None:
    indexer = Indexer(config=indexer_config, context=fake_context, console=Console(quiet=True))

    with patch("simgrep.indexer.IndexService") as mock_index_service:
        mock_service_instance = mock_index_service.return_value
        mock_service_instance.run_index.return_value = (1, 10, 0)  # files, chunks, errors
        mock_service_instance.final_max_label = 9

        with patch.object(indexer, "_prepare_data_stores") as mock_prepare:
            indexer.metadata_store = MagicMock()
            indexer.usearch_index = MagicMock()

            target_path = tmp_path / "data"
            target_path.mkdir()
            (target_path / "file.txt").touch()

            indexer.run_index([target_path], wipe_existing=True)

            mock_prepare.assert_called_once_with(True)
            mock_index_service.assert_called_once()
            mock_service_instance.run_index.assert_called_once()
            indexer.metadata_store.set_max_usearch_label.assert_called_once_with(9)
            indexer.usearch_index.save.assert_called_once_with(indexer_config.usearch_index_path)