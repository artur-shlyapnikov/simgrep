import duckdb
import numpy as np
import pytest
from rich.console import Console

pytest.importorskip("sentence_transformers")

import simgrep.searcher as searcher
from simgrep.models import OutputMode, SimgrepConfig


class TestPersistentSearcherCaching:
    def test_model_loaded_once_for_repeated_searches(self, tmp_path, monkeypatch):
        call_counter = {"count": 0}

        class DummyModel:
            def __init__(self, name: str) -> None:
                call_counter["count"] += 1
                self.name = name

            def encode(self, texts, show_progress_bar=False):  # noqa: D401
                return np.zeros((len(texts), 3), dtype=np.float32)

        monkeypatch.setattr(searcher, "SentenceTransformer", DummyModel)

        def fake_search_inmemory_index(index, query_embedding, k):  # noqa: D401
            return []

        monkeypatch.setattr(searcher, "search_inmemory_index", fake_search_inmemory_index)

        cfg = SimgrepConfig(default_project_data_dir=tmp_path)
        console = Console(quiet=True)
        db_conn = duckdb.connect(":memory:")

        searcher._embedding_model_cache.clear()
        searcher.perform_persistent_search(
            query_text="first",
            console=console,
            db_conn=db_conn,
            vector_index=None,  # type: ignore[arg-type]
            global_config=cfg,
            output_mode=OutputMode.show,
        )
        assert call_counter["count"] == 1

        searcher.perform_persistent_search(
            query_text="second",
            console=console,
            db_conn=db_conn,
            vector_index=None,  # type: ignore[arg-type]
            global_config=cfg,
            output_mode=OutputMode.show,
        )
        assert call_counter["count"] == 1
