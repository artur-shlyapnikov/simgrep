from functools import lru_cache
from typing import List, cast

from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError
from rich.console import Console
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from simgrep.core.abstractions import ChunkSeq, TokenChunker
from simgrep.core.errors import SimgrepError
from simgrep.core.models import Chunk


def _is_model_cached(model_name: str) -> bool:
    try:
        snapshot_download(model_name, local_files_only=True)
        return True
    except (FileNotFoundError, LocalEntryNotFoundError):
        return False


@lru_cache(maxsize=None)
def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    def _load() -> PreTrainedTokenizerBase:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return cast(PreTrainedTokenizerBase, tokenizer)
        except OSError as e:
            raise SimgrepError(
                f"Failed to load tokenizer for model '{model_name}'. "
                f"Ensure the model name is correct and an internet connection "
                f"is available for the first download. Original error: {e}"
            ) from e

    if not _is_model_cached(model_name):
        console = Console()
        with console.status(
            f"[bold yellow]First-time setup: downloading tokenizer for model '{model_name}'...[/bold yellow]",
            spinner="dots",
        ) as status:
            model = _load()
            status.update(f"[bold green]Tokenizer for '{model_name}' downloaded.[/bold green]")
            return model
    else:
        return _load()


class HFChunker(TokenChunker):
    def __init__(self, model_name: str, chunk_size: int, overlap: int):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if overlap < 0:
            raise ValueError("overlap must be a non-negative integer.")
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size.")

        self.model_name = model_name
        self._tokenizer = load_tokenizer(model_name)
        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk(self, text: str) -> ChunkSeq:
        if not text.strip():
            return []

        encoding = self._tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=False,
        )

        all_token_ids = encoding.input_ids
        all_offsets = encoding.offset_mapping

        if not all_token_ids:
            return []

        chunks: List[Chunk] = []
        step = self._chunk_size - self._overlap

        current_token_idx = 0
        while current_token_idx < len(all_token_ids):
            token_slice_end = current_token_idx + self._chunk_size

            chunk_token_ids_batch = all_token_ids[current_token_idx:token_slice_end]
            chunk_offsets_batch = all_offsets[current_token_idx:token_slice_end]

            if not chunk_token_ids_batch:
                break

            start_char = chunk_offsets_batch[0][0]
            end_char = chunk_offsets_batch[-1][1]

            chunk_text = self._tokenizer.decode(chunk_token_ids_batch, skip_special_tokens=True)
            num_tokens_in_this_chunk = len(chunk_token_ids_batch)

            chunks.append(
                Chunk(
                    id=-1,
                    file_id=-1,
                    text=chunk_text,
                    start=start_char,
                    end=end_char,
                    tokens=num_tokens_in_this_chunk,
                )
            )
            current_token_idx += step

        return chunks
