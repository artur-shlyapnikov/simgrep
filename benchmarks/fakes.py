"""Deterministic in-process test doubles for stable CI benchmarks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class Chunk:
    """Simple chunk representation for fake chunker."""

    id: int
    file_id: int
    text: str
    start: int
    end: int
    tokens: int


class DeterministicEmbedder:
    """
    Deterministic fake embedder using stable hash-based vectors.

    Vectors are deterministic across processes and Python versions by using SHA-256.
    """

    def __init__(self, ndim: int = 64):
        self._ndim = ndim

    @property
    def ndim(self) -> int:
        return self._ndim

    def encode(
        self,
        texts: List[str],
        *,
        is_query: bool = False,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Encode texts into deterministic float32 vectors.

        Each vector is normalized for stable inner-product behavior in USearch.
        """
        vectors = []
        for text in texts:
            # Create deterministic hash
            combined = f"embed:{text}"
            hash_bytes = hashlib.sha256(combined.encode()).digest()

            # Convert hash to float32 vector of requested dimension
            # Use multiple hashes to fill dimensions > 32
            values = []
            for block_start in range(0, self._ndim, 32):
                block_hash = hashlib.sha256(f"{hash_bytes}:{block_start}".encode()).digest()
                # Convert 32 bytes to 32 floats (each byte becomes 0-255, then normalize to 0-1)
                block_values = [b / 255.0 for b in block_hash[: min(32, self._ndim - block_start)]]
                values.extend(block_values)

            vec = np.array(values[: self._ndim], dtype=np.float32)

            # Normalize for cosine similarity (inner product with normalized = cosine)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            vectors.append(vec)

        return np.array(vectors, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a query using the same deterministic method."""
        return self.encode([query], is_query=True)[0]


class FixedTokenChunker:
    """
    Simple fixed-size chunker without HF dependency.

    Chunks text into pieces of approximately target tokens with overlap.
    """

    def __init__(self, chunk_size_tokens: int = 128, overlap_tokens: int = 20):
        self._chunk_size = chunk_size_tokens
        self._overlap = overlap_tokens

    def chunk(self, text: str) -> List[Chunk]:
        """Split text into chunks of roughly target token size."""
        # Simple whitespace token estimation (words ~= tokens for English)
        words = text.split()
        tokens_per_word = 1.0  # Conservative estimate

        chunks: List[Chunk] = []
        chunk_id = 0

        if not words:
            return []

        # Calculate approximate words per chunk
        words_per_chunk = int(self._chunk_size / tokens_per_word)
        overlap_words = int(self._overlap / tokens_per_word)

        start_word = 0
        while start_word < len(words):
            end_word = min(start_word + words_per_chunk, len(words))
            chunk_words = words[start_word:end_word]
            chunk_text = " ".join(chunk_words)

            # Find character offsets in original text
            # Approximate: find position in original
            char_start = 0
            if start_word > 0:
                # Find position of start_word in original
                search_start = 0
                for _ in range(start_word):
                    next_space = text.find(" ", search_start)
                    if next_space == -1:
                        break
                    search_start = next_space + 1
                char_start = search_start

            char_end = text.find(chunk_words[-1], char_start) + len(chunk_words[-1]) if chunk_words else char_start

            chunks.append(
                Chunk(
                    id=chunk_id,
                    file_id=0,
                    text=chunk_text,
                    start=char_start,
                    end=char_end,
                    tokens=len(chunk_words),
                )
            )

            chunk_id += 1
            start_word = end_word - overlap_words
            if start_word >= len(words) - overlap_words:
                break

        return chunks


class PlainTextExtractor:
    """Simple text extractor that reads plain text files."""

    def extract(self, path: Path) -> str:
        """Extract text content from a file."""
        return path.read_text(encoding="utf-8", errors="replace")


class FakeChunker:
    """Fake chunker that returns empty chunks for testing."""

    def chunk(self, text: str) -> List[Chunk]:
        """Return a single fake chunk for testing."""
        if not text.strip():
            return []
        return [Chunk(id=0, file_id=0, text=text[:500], start=0, end=min(500, len(text)), tokens=len(text.split()[:100]))]
