"""External coverage for CrossEncoderReranker (real ms-marco-MiniLM-L-6-v2 download + inference)."""

from __future__ import annotations

import numpy as np
import pytest

from simgrep.adapters.reranker import CrossEncoderReranker
from simgrep.errors import RerankError

pytestmark = pytest.mark.external

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_lazy_no_model_load_until_score(monkeypatch: pytest.MonkeyPatch) -> None:
    """Construction must not import/instantiate CrossEncoder; first score() does."""
    import sentence_transformers

    calls: list[str] = []

    class _SpyCrossEncoder:
        def __init__(self, model_name: str) -> None:
            calls.append(model_name)

        def predict(self, pairs: list[tuple[str, str]], batch_size: int = 32) -> np.ndarray:
            del batch_size
            return np.asarray([float(len(doc)) for _, doc in pairs], dtype=np.float32)

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", _SpyCrossEncoder)
    reranker = CrossEncoderReranker(MODEL)
    assert calls == []  # lazy: construction performs no model I/O
    scores = reranker.score("q", ["abcd", "de"])
    assert calls == [MODEL]  # loaded exactly once, on first score()
    assert scores.tolist() == [4.0, 2.0]


def test_empty_documents_score_without_loading_model(monkeypatch: pytest.MonkeyPatch) -> None:
    import sentence_transformers

    class _Boom:
        def __init__(self, *_: object) -> None:
            raise AssertionError("model must not load for empty input")

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", _Boom)
    scores = CrossEncoderReranker(MODEL).score("q", [])
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (0,)


def test_model_load_failure_wrapped_in_rerank_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import sentence_transformers

    def _explode(*_: object) -> None:
        raise RuntimeError("network down")

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", _explode)
    reranker = CrossEncoderReranker("does/not/exist")
    with pytest.raises(RerankError) as excinfo:
        reranker.score("q", ["doc"])
    assert "does/not/exist" in str(excinfo.value)
    assert excinfo.value.hint is not None


def test_batch_length_preserved_and_float32() -> None:
    reranker = CrossEncoderReranker(MODEL)
    docs = [
        "",
        "short",
        "a much longer document about password reset workflows and account recovery",
        "x" * 200,
    ]
    scores = reranker.score("password reset", docs)
    assert isinstance(scores, np.ndarray)
    assert scores.dtype == np.float32
    assert scores.shape == (len(docs),)


def test_relevant_document_outranks_irrelevant() -> None:
    reranker = CrossEncoderReranker(MODEL)
    relevant = "To reset your password, enter the email address linked to your account."
    irrelevant = "Giraffes are the tallest living terrestrial animals on earth."
    scores = reranker.score("how do I reset my password?", [relevant, irrelevant])
    assert float(scores[0]) > float(scores[1])
