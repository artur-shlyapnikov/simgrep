import pathlib

import pytest
from rich.console import Console

from simgrep.core.context import SimgrepContext
from simgrep.indexer import Indexer, IndexerConfig

pytest.importorskip("transformers")
pytest.importorskip("sentence_transformers")
pytest.importorskip("usearch.index")


def test_index_with_workers(tmp_path: pathlib.Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "a.txt").write_text("hello world")
    (data_dir / "b.txt").write_text("another file")

    cfg = IndexerConfig(
        project_name="workers_proj",
        db_path=tmp_path / "meta.duckdb",
        usearch_index_path=tmp_path / "index.usearch",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size_tokens=16,
        chunk_overlap_tokens=0,
        file_scan_patterns=["*.txt"],
        max_index_workers=2,
    )
    context = SimgrepContext.from_defaults(
        model_name=cfg.embedding_model_name,
        chunk_size=cfg.chunk_size_tokens,
        chunk_overlap=cfg.chunk_overlap_tokens,
    )
    indexer = Indexer(cfg, context, Console(quiet=True))
    indexer.run_index([data_dir], wipe_existing=True)

    assert cfg.db_path.exists()
    assert cfg.usearch_index_path.exists()