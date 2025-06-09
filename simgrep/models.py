from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, Field


class OutputMode(str, Enum):
    show = "show"
    paths = "paths"
    # future: json, rag, copy-chunks, copy-files, count


class ChunkData(BaseModel):
    text: str  # the actual text content of the chunk
    source_file_path: Path  # absolute path to the original file
    source_file_id: (
        int  # ephemeral id for the source file within the current processing batch
    )
    # (e.g., index from enumerate(files_to_process))
    usearch_label: int  # unique label assigned to this chunk for usearch.
    # this will be the global index of the chunk's embedding.
    start_char_offset: (
        int  # character offset of the chunk's start in the original file content
    )
    end_char_offset: (
        int  # character offset of the chunk's end in the original file content
    )
    token_count: int  # number of tokens in this chunk (as per the specified tokenizer)


@dataclass
class SearchResult:
    label: int
    score: float


class ProjectConfig(BaseModel):
    name: str
    indexed_paths: List[Path] = Field(default_factory=list)
    embedding_model: str
    db_path: Path
    usearch_index_path: Path


class SimgrepConfig(BaseModel):
    """
    Global configuration for simgrep.
    for deliverable 3.1, this primarily establishes the directory for the default project's data
    and centralizes other global default settings.
    """

    # core for deliverable 3.1: directory for the default project's database and vector index.
    # the path `~/.config/simgrep/default_project` will store data for the default project.
    # the parent `~/.config/simgrep/` will later hold the global config.toml and global metadata db.
    default_project_data_dir: Path = Field(
        default_factory=lambda: Path("~/.config/simgrep/default_project").expanduser()
    )

    config_file: Path = Field(
        default_factory=lambda: Path("~/.config/simgrep/config.toml").expanduser()
    )

    db_directory: Path = Field(
        default_factory=lambda: Path("~/.config/simgrep").expanduser()
    )

    projects: Dict[str, ProjectConfig] = Field(default_factory=dict)

    # centralizing other global defaults from main.py constants / architecture doc:
    # these will be used by persistent indexing logic in later deliverables (e.g., 3.3, 3.4).
    # ephemeral search in main.py might still use its local constants for now.
    default_embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    default_chunk_size_tokens: int = 128
    default_chunk_overlap_tokens: int = 20

    # llm_api_key: Optional[str] = None # deferred to a later phase (e.g., rag implementation)
    # projects: List[ProjectConfig] = Field(default_factory=list) # deferred to deliverable 4.4

    model_config = {"validate_assignment": True}  # good practice for pydantic models