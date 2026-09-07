from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Optional, cast

from simgrep.errors import SimgrepError
from simgrep.models import Chunk
from simgrep.text import compute_line_starts, expand_offsets_to_line_bounds

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@lru_cache(maxsize=None)
def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    from transformers import AutoTokenizer

    try:
        # Offline-first: a cached model must not cost a network round-trip.
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)  # type: ignore[no-untyped-call]
        return cast("PreTrainedTokenizerBase", tokenizer)
    except OSError:
        pass
    except Exception as exc:
        raise SimgrepError(f"Failed to load tokenizer for model '{model_name}'.") from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)  # type: ignore[no-untyped-call]
        return cast("PreTrainedTokenizerBase", tokenizer)
    except OSError as exc:
        raise SimgrepError(f"Failed to load tokenizer for model '{model_name}'.") from exc


def _resolve_model_token_limit(model_name: str, tokenizer: PreTrainedTokenizerBase) -> Optional[int]:
    if model_name.strip().lower().startswith("lightonai/lateon-code"):
        return 2048
    max_len = getattr(tokenizer, "model_max_length", None)
    if isinstance(max_len, int) and 0 < max_len < 1_000_000:
        return max_len
    return None


def _clamp_chunking(*, requested_chunk_size: int, requested_overlap: int, token_limit: Optional[int]) -> tuple[int, int]:
    chunk_size = requested_chunk_size
    if token_limit is not None and chunk_size >= token_limit:
        chunk_size = max(1, token_limit - 1)
    overlap = requested_overlap
    if overlap >= chunk_size:
        overlap = max(0, chunk_size - 1)
    return chunk_size, overlap


class HFChunker:
    """Token-boundary chunker that defers tokenizer loading until first use.

    Search-only invocations never tokenize, so paying the transformers import
    and tokenizer load at construction time would tax every query.
    """

    def __init__(self, model_name: str, chunk_size: int, overlap: int):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if overlap < 0:
            raise ValueError("overlap must be non-negative.")
        self.model_name = model_name
        self._requested_chunk_size = chunk_size
        self._requested_overlap = overlap
        self._tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._model_token_limit: Optional[int] = None
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._tokenizer = load_tokenizer(self.model_name)
        self._model_token_limit = _resolve_model_token_limit(self.model_name, self._tokenizer)
        self._chunk_size, self._overlap = _clamp_chunking(
            requested_chunk_size=self._requested_chunk_size,
            requested_overlap=self._requested_overlap,
            token_limit=self._model_token_limit,
        )
        self._loaded = True

    @property
    def model_token_limit(self) -> Optional[int]:
        self._load()
        return self._model_token_limit

    def chunk(self, text: str) -> list[Chunk]:
        if not text.strip():
            return []
        self._load()
        assert self._tokenizer is not None
        encoding = self._tokenizer(text, return_offsets_mapping=True, add_special_tokens=False, truncation=False)
        token_ids = encoding.input_ids
        offsets = encoding.offset_mapping
        if not token_ids:
            return []
        line_starts: list[int] | None = None  # computed lazily on first expansion
        step = max(1, self._chunk_size - self._overlap)
        chunks: list[Chunk] = []
        token_idx = 0
        while token_idx < len(token_ids):
            token_slice_end = token_idx + self._chunk_size
            chunk_offsets = offsets[token_idx:token_slice_end]
            if not chunk_offsets:
                break
            raw_start = int(chunk_offsets[0][0])
            raw_end = int(chunk_offsets[-1][1])
            if line_starts is None:
                line_starts = compute_line_starts(text)
            if raw_start >= raw_end:
                token_idx += step
                continue
            _, _, start, end = expand_offsets_to_line_bounds(text, raw_start, raw_end, max_extra_chars=1000, line_starts=line_starts)
            chunk_text = text[start:end]
            cleaned = chunk_text
            for ch in ("\u200b", "\u200c", "\u200d", "\ufeff"):
                cleaned = cleaned.replace(ch, "")
            if cleaned.strip():
                chunks.append(Chunk(id=-1, file_id=-1, text=chunk_text, start=start, end=end, tokens=len(chunk_offsets)))
            token_idx += step
        return chunks
