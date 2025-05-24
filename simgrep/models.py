from pathlib import Path

from pydantic import BaseModel, Field  # Ensure Field is imported


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


class SimgrepConfig(BaseModel):
    """
    Global configuration for simgrep.
    For Deliverable 3.1, this primarily establishes the directory for the default project's data
    and centralizes other global default settings.
    """
    # Core for Deliverable 3.1: Directory for the default project's database and vector index.
    # The path `~/.config/simgrep/default_project` will store data for the default project.
    # The parent `~/.config/simgrep/` will later hold the global config.toml and global metadata DB.
    default_project_data_dir: Path = Field(
        default_factory=lambda: Path("~/.config/simgrep/default_project").expanduser()
    )

    # Centralizing other global defaults from main.py constants / architecture doc:
    # These will be used by persistent indexing logic in later deliverables (e.g., 3.3, 3.4).
    # Ephemeral search in main.py might still use its local constants for now.
    default_embedding_model_name: str = "all-MiniLM-L6-v2"
    default_chunk_size_tokens: int = 128
    default_chunk_overlap_tokens: int = 20

    # llm_api_key: Optional[str] = None # Deferred to a later phase (e.g., RAG implementation)
    # projects: List[ProjectConfig] = Field(default_factory=list) # Deferred to Deliverable 4.4

    class Config:
        validate_assignment = True # Good practice for Pydantic models