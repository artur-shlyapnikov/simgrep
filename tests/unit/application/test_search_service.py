import numpy as np
import pytest

from simgrep.core.abstractions import Embedder, Repository
from simgrep.core.models import SearchResult
from simgrep.services.search_service import SearchService


def test_search_service_with_results(fake_repository: Repository, fake_embedder: Embedder, fake_vector_index_factory) -> None:
    vector_index = fake_vector_index_factory(ndim=fake_embedder.ndim)
    vector_index.add(np.array([1], dtype=np.int64), np.random.rand(1, fake_embedder.ndim))

    service = SearchService(store=fake_repository, embedder=fake_embedder, index=vector_index)
    results = service.search(query="test", k=1, min_score=0.1, file_filter=None, keyword_filter=None)

    assert len(results) == 1
    assert results[0].label == 1
    assert results[0].score == 0.99
    assert "fake text" in results[0].chunk_text
    assert "file.txt" in str(results[0].file_path)


def test_search_service_no_results(fake_repository: Repository, fake_embedder: Embedder, fake_vector_index_factory) -> None:
    vector_index = fake_vector_index_factory(ndim=fake_embedder.ndim)
    # No data added to vector_index

    service = SearchService(store=fake_repository, embedder=fake_embedder, index=vector_index)
    results = service.search(query="test", k=1, min_score=0.1, file_filter=None, keyword_filter=None)

    assert len(results) == 0


def test_search_service_filters_low_score(fake_repository: Repository, fake_embedder: Embedder, fake_vector_index_factory, monkeypatch) -> None:
    vector_index = fake_vector_index_factory(ndim=fake_embedder.ndim)
    vector_index.add(np.array([1], dtype=np.int64), np.random.rand(1, fake_embedder.ndim))

    # Make search return a low score
    monkeypatch.setattr(vector_index, "search", lambda vec, k: [SearchResult(label=1, score=0.05)])

    service = SearchService(store=fake_repository, embedder=fake_embedder, index=vector_index)
    results = service.search(query="test", k=1, min_score=0.9, file_filter=None, keyword_filter=None)

    assert len(results) == 0
