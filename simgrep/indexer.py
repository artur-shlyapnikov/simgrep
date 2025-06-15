import os
import pathlib
from typing import List, Optional

import duckdb
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .core.abstractions import VectorIndex
from .core.context import SimgrepContext
from .core.errors import IndexerError, MetadataDBError, VectorStoreError
from .repository import MetadataStore
from .services.index_service import IndexService


class IndexerConfig(BaseModel):
    project_name: str
    db_path: pathlib.Path
    usearch_index_path: pathlib.Path
    embedding_model_name: str
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    file_scan_patterns: List[str] = Field(default_factory=lambda: ["*.txt"])
    max_index_workers: int = os.cpu_count() or 4


class Indexer:
    def __init__(self, config: IndexerConfig, context: SimgrepContext, console: Console):
        self.config = config
        self.console = console
        self.context = context
        self.db_conn: Optional[duckdb.DuckDBPyConnection] = None
        self.metadata_store: Optional[MetadataStore] = None
        self.usearch_index: Optional[VectorIndex] = None
        self.index_service: Optional[IndexService] = None

        try:
            self.console.print(f"Using services for model: '{self.config.embedding_model_name}'...")
            self.embedding_ndim: int = self.context.embedder.ndim
            self.console.print(f"Embedding dimension set to: {self.embedding_ndim}")
        except Exception as e:
            raise IndexerError(f"Failed to initialize services from context: {e}") from e

    def _prepare_data_stores(self, wipe_existing: bool) -> None:
        self.console.print("Preparing data stores (database and vector index)...")
        try:
            self.metadata_store = MetadataStore(persistent=True, db_path=self.config.db_path)
            self.db_conn = self.metadata_store.conn
            self.console.print(f"Connected to database: {self.config.db_path}")

            if wipe_existing and self.metadata_store:
                self.console.print(f"Wiping existing data from database: {self.config.db_path}...")
                self.metadata_store.clear_persistent_project_data()
                self.console.print("Database wiped.")
        except MetadataDBError as e:
            raise IndexerError(f"Database preparation failed: {e}") from e

        try:
            # If index is not initialized, or if we need to wipe, create a new one.
            if self.usearch_index is None or wipe_existing:
                self.usearch_index = self.context.index_factory(self.embedding_ndim)

            if wipe_existing:
                self.console.print(f"Wiping existing vector index: {self.config.usearch_index_path}...")
                self.config.usearch_index_path.unlink(missing_ok=True)
            else:
                # If index is empty and file exists, load it.
                if self.config.usearch_index_path.exists() and self.usearch_index and len(self.usearch_index) == 0:
                    self.console.print(f"Loading vector index from: {self.config.usearch_index_path}...")
                    self.usearch_index.load(self.config.usearch_index_path)
                    self.console.print(f"Loaded existing vector index with {len(self.usearch_index)} items.")
        except (VectorStoreError, FileNotFoundError) as e:
            raise IndexerError(f"Vector store preparation failed: {e}") from e

    def run_index(self, target_paths: List[pathlib.Path], wipe_existing: bool) -> None:
        try:
            self._prepare_data_stores(wipe_existing)
            if self.metadata_store is None or self.usearch_index is None:
                raise IndexerError("Data stores were not properly initialized.")

            self.index_service = IndexService(
                extractor=self.context.extractor,
                chunker=self.context.chunker,
                embedder=self.context.embedder,
                store=self.metadata_store,
                index=self.usearch_index,
            )

            progress_columns = [
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TextColumn("/"),
                TimeRemainingColumn(),
            ]
            with Progress(*progress_columns, console=self.console, transient=False) as progress:
                total_files_processed, total_chunks_indexed, total_errors = self.index_service.run_index(
                    target_paths=target_paths,
                    file_scan_patterns=self.config.file_scan_patterns,
                    wipe_existing=wipe_existing,
                    max_workers=self.config.max_index_workers,
                    progress=progress,
                    console=self.console,
                )

            if self.index_service is not None and self.metadata_store is not None:
                final_label = self.index_service.final_max_label
                if final_label >= 0:
                    self.metadata_store.set_max_usearch_label(final_label)

            if self.usearch_index is not None:
                self.console.print(f"Saving vector index with {len(self.usearch_index)} items...")
                self.usearch_index.save(self.config.usearch_index_path)
                self.console.print("Vector index saved.")

            self.console.print(f"\n[bold green]Indexing complete for project '{self.config.project_name}'.[/bold green]")
            self.console.print(f"  Summary: {total_files_processed} files processed, {total_chunks_indexed} chunks indexed, {total_errors} errors encountered.")

        except IndexerError as e:
            self.console.print(f"[bold red]Indexer Error: {e}[/bold red]")
            raise
        finally:
            if self.metadata_store:
                self.metadata_store.close()
