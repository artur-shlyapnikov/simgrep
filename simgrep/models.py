from pathlib import Path

from pydantic import BaseModel


class ChunkData(BaseModel):
    text: str  # The actual text content of the chunk
    source_file_path: Path  # Absolute path to the original file
    source_file_id: (
        int  # Ephemeral ID for the source file within the current processing batch
    )
    # (e.g., index from enumerate(files_to_process))
    usearch_label: int  # Unique label assigned to this chunk for USearch.
    # This will be the global index of the chunk's embedding.
    start_char_offset: (
        int  # Character offset of the chunk's start in the original file content
    )
    end_char_offset: (
        int  # Character offset of the chunk's end in the original file content
    )
    token_count: int  # Number of tokens in this chunk (as per the specified tokenizer)
