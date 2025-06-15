from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


@dataclass
class Chunk:
    """A generic representation of a piece of text from a file, for use in services."""

    id: int
    file_id: int
    text: str
    start: int
    end: int
    tokens: int


class OutputMode(str, Enum):
    """Defines the supported output formats for search results."""

    show = "show"
    paths = "paths"
    json = "json"
    count_results = "count"


class ChunkData(BaseModel):
    """Represents a chunk of text and its source metadata, used in ephemeral search."""

    text: str
    source_file_path: Path
    source_file_id: int
    usearch_label: int
    start_char_offset: int
    end_char_offset: int
    token_count: int


@dataclass
class SearchResult:
    """Represents a single vector search match from the index."""

    label: int
    score: float
    file_path: Optional[Path] = None
    chunk_text: Optional[str] = None
    start_char_offset: Optional[int] = None
    end_char_offset: Optional[int] = None


class ProjectConfig(BaseModel):
    """Configuration for a single simgrep project."""

    name: str
    indexed_paths: List[Path] = Field(default_factory=list)
    embedding_model: str
    db_path: Path
    usearch_index_path: Path


class SimgrepConfig(BaseModel):
    """Global configuration for the simgrep application."""

    default_project_data_dir: Path = Field(default_factory=lambda: Path("~/.config/simgrep/default_project").expanduser())
    config_file: Path = Field(default_factory=lambda: Path("~/.config/simgrep/config.toml").expanduser())
    db_directory: Path = Field(default_factory=lambda: Path("~/.config/simgrep").expanduser())
    default_embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    default_chunk_size_tokens: int = 128
    default_chunk_overlap_tokens: int = 20
    model_config = {"validate_assignment": True}