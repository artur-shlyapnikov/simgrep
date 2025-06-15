from __future__ import annotations

from typing import Callable

from simgrep.core.abstractions import (
    Embedder,
    TextExtractor,
    TokenChunker,
    VectorIndex,
)


class SimgrepContext:
    """A dependency injection container for simgrep services and adapters."""

    def __init__(
        self,
        extractor: TextExtractor,
        chunker: TokenChunker,
        embedder: Embedder,
        index_factory: Callable[[int], VectorIndex],
    ):
        self.extractor = extractor
        self.chunker = chunker
        self.embedder = embedder
        self.index_factory = index_factory

    @classmethod
    def from_defaults(cls, model_name: str, chunk_size: int, chunk_overlap: int) -> SimgrepContext:
        """
        Creates a context with default adapter implementations.
        This is the primary factory for building the context.
        """
        from simgrep.adapters.hf_chunker import HFChunker
        from simgrep.adapters.sentence_embedder import SentenceEmbedder
        from simgrep.adapters.unstructured_extractor import UnstructuredExtractor
        from simgrep.adapters.usearch_index import USearchIndex

        embedder = SentenceEmbedder(model_name=model_name)

        return cls(
            extractor=UnstructuredExtractor(),
            chunker=HFChunker(
                model_name=model_name,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
            ),
            embedder=embedder,
            index_factory=lambda ndim: USearchIndex(ndim=ndim),
        )

    @classmethod
    def default(
        cls,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 128,
        chunk_overlap: int = 20,
    ) -> "SimgrepContext":
        return cls.from_defaults(model_name, chunk_size, chunk_overlap)
