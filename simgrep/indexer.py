import datetime
import hashlib
import os
import pathlib # Use pathlib consistently
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import numpy as np
import usearch.index
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
from transformers import PreTrainedTokenizerBase

from .exceptions import MetadataDBError, VectorStoreError # Assuming these are in .exceptions
from .metadata_db import (
    batch_insert_text_chunks,
    clear_persistent_project_data,
    connect_persistent_db,
    insert_indexed_file_record,
)
from .processor import (
    ProcessedChunkInfo,
    chunk_text_by_tokens,
    extract_text_from_file,
    generate_embeddings,
    load_tokenizer,
)
from .vector_store import load_persistent_index, save_persistent_index


class IndexerError(Exception):
    """Custom exception for errors during the indexing process."""

    pass


class IndexerConfig(BaseModel):
    project_name: str
    db_path: pathlib.Path
    usearch_index_path: pathlib.Path
    embedding_model_name: str
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    file_scan_patterns: List[str] = Field(default_factory=lambda: ["*.txt"])


class Indexer:
    def __init__(self, config: IndexerConfig, console: Console):
        self.config = config
        self.console = console
        self.db_conn: Optional[duckdb.DuckDBPyConnection] = None
        self.usearch_index: Optional[usearch.index.Index] = None
        self._current_usearch_label: int = 0 # Global counter for unique USearch labels

        try:
            self.console.print(f"Loading tokenizer for model: '{self.config.embedding_model_name}'...")
            self.tokenizer: PreTrainedTokenizerBase = load_tokenizer(
                self.config.embedding_model_name
            )
            self.console.print("Tokenizer loaded.")
        except RuntimeError as e:
            raise IndexerError(f"Failed to load tokenizer: {e}") from e

        try:
            self.console.print("Determining embedding dimension...")
            # Determine embedding dimension
            dummy_emb = generate_embeddings(["simgrep_test_string"], self.config.embedding_model_name)
            if dummy_emb.ndim != 2 or dummy_emb.shape[0] == 0 or dummy_emb.shape[1] == 0:
                raise IndexerError(
                    f"Could not determine embedding dimension using model {self.config.embedding_model_name}. "
                    f"Dummy embedding shape: {dummy_emb.shape}"
                )
            self.embedding_ndim: int = dummy_emb.shape[1]
            self.console.print(f"Embedding dimension set to: {self.embedding_ndim}")
        except RuntimeError as e:
            raise IndexerError(
                f"Failed to load embedding model or generate dummy embedding for ndim: {e}"
            ) from e


    def _calculate_file_content_hash(self, file_path: pathlib.Path) -> str:
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                # Read and update hash string value in blocks of 4K
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except FileNotFoundError:
            self.console.print(f"[bold red]Error: File not found during hash calculation: {file_path}[/bold red]")
            raise IndexerError(f"File not found for hashing: {file_path}")
        except IOError as e:
            self.console.print(f"[bold red]Error: I/O error hashing file {file_path}: {e}[/bold red]")
            raise IndexerError(f"I/O error hashing file {file_path}: {e}")


    def _prepare_data_stores(self, wipe_existing: bool) -> None:
        self.console.print("Preparing data stores (database and vector index)...")
        # Database
        try:
            self.db_conn = connect_persistent_db(self.config.db_path)
            self.console.print(f"Connected to database: {self.config.db_path}")
            if wipe_existing:
                self.console.print(f"Wiping existing data from database: {self.config.db_path}...")
                clear_persistent_project_data(self.db_conn)
                self._current_usearch_label = 0 # Reset label counter when wiping
                self.console.print("Database wiped.")
        except MetadataDBError as e:
            raise IndexerError(f"Database preparation failed: {e}") from e

        # Vector Store
        try:
            if wipe_existing:
                self.console.print(f"Wiping existing vector index: {self.config.usearch_index_path}...")
                self.config.usearch_index_path.unlink(missing_ok=True)
                self.usearch_index = usearch.index.Index(
                    ndim=self.embedding_ndim, metric="cos", dtype="f32" # TODO: Make metric/dtype configurable
                )
                self.console.print("New empty vector index created.")
            else: # For future incremental logic
                self.console.print(f"Loading vector index from: {self.config.usearch_index_path}...")
                self.usearch_index = load_persistent_index(self.config.usearch_index_path)
                if self.usearch_index is None:
                    self.console.print("No existing vector index found. Creating new one.")
                    self.usearch_index = usearch.index.Index(
                        ndim=self.embedding_ndim, metric="cos", dtype="f32"
                    )
                    # For incremental, if loading, would need to determine max label.
                    # For wipe_existing=True, _current_usearch_label is already 0.
                else:
                    self.console.print(f"Loaded existing vector index with {len(self.usearch_index)} items.")
                    # Future: determine self._current_usearch_label from max label in index or DB
                    # For now, this path is not fully utilized by Deliverable 3.3's wipe_existing=True.
                    if len(self.usearch_index) > 0 and self.usearch_index.max_key() is not None:
                         self._current_usearch_label = self.usearch_index.max_key() + 1
                    else:
                         self._current_usearch_label = 0


        except VectorStoreError as e:
            raise IndexerError(f"Vector store preparation failed: {e}") from e
        except Exception as e: # Catch other unexpected errors like usearch.index.Index init
            raise IndexerError(f"Unexpected error during vector store preparation: {e}") from e

        if self.usearch_index is None: # Should not happen if logic is correct
            raise IndexerError("USearch index was not initialized after _prepare_data_stores.")


    def _process_and_index_file(
        self, file_path: pathlib.Path, progress: Progress, task_id
    ) -> Tuple[int, int]:
        num_chunks_this_file = 0
        errors_this_file = 0
        file_display_name = file_path.name

        progress.update(task_id, description=f"Processing: {file_display_name}...")
        
        if not self.db_conn: # Should be set by _prepare_data_stores
            self.console.print(f"[bold red]Error: DB connection not available for file {file_path}. Skipping.[/bold red]")
            errors_this_file +=1
            progress.update(task_id, advance=1, description=f"Skipped (DB error): {file_display_name}")
            return num_chunks_this_file, errors_this_file

        try:
            # File Metadata
            content_hash = self._calculate_file_content_hash(file_path)
            file_stat = file_path.stat()
            file_size = file_stat.st_size
            last_modified_ts = file_stat.st_mtime

            file_id = insert_indexed_file_record(
                self.db_conn,
                file_path=str(file_path.resolve()),
                content_hash=content_hash,
                file_size_bytes=file_size,
                last_modified_os_timestamp=last_modified_ts,
            )

            if file_id is None:
                self.console.print(f"[bold red]Error: Failed to get file_id for {file_path}. Skipping.[/bold red]")
                errors_this_file += 1
                progress.update(task_id, advance=1, description=f"Skipped (DB insert): {file_display_name}")
                return num_chunks_this_file, errors_this_file

            # Text Extraction & Chunking
            text_content = extract_text_from_file(file_path)
            if not text_content.strip():
                self.console.print(f"[yellow]Info: File {file_path} is empty or whitespace only. Skipping chunking.[/yellow]")
                progress.update(task_id, advance=1, description=f"Processed (empty): {file_display_name}")
                return num_chunks_this_file, errors_this_file

            processed_chunks: List[ProcessedChunkInfo] = chunk_text_by_tokens(
                text_content,
                self.tokenizer,
                self.config.chunk_size_tokens,
                self.config.chunk_overlap_tokens,
            )

            if not processed_chunks:
                self.console.print(f"[yellow]Info: No chunks generated for {file_path}.[/yellow]")
                progress.update(task_id, advance=1, description=f"Processed (no chunks): {file_display_name}")
                return num_chunks_this_file, errors_this_file

            # Embedding & Storing Chunks
            chunk_texts = [chunk['text'] for chunk in processed_chunks]
            embeddings_np = generate_embeddings(chunk_texts, self.config.embedding_model_name)

            chunk_db_records: List[Dict[str, Any]] = []
            usearch_labels_for_batch: List[int] = []

            for i, chunk_info in enumerate(processed_chunks):
                current_label = self._current_usearch_label
                usearch_labels_for_batch.append(current_label)
                
                chunk_db_records.append({
                    "file_id": file_id,
                    "usearch_label": current_label,
                    "chunk_text_snippet": chunk_info['text'][:255], # Truncate for snippet
                    "start_char_offset": chunk_info['start_char_offset'],
                    "end_char_offset": chunk_info['end_char_offset'],
                    "token_count": chunk_info['token_count'],
                    "embedding_hash": None, # Placeholder for V1
                })
                self._current_usearch_label += 1
            
            batch_insert_text_chunks(self.db_conn, chunk_db_records)
            
            if self.usearch_index is None: # Should be initialized
                 raise IndexerError("USearch index is None during file processing.")
            self.usearch_index.add(
                keys=np.array(usearch_labels_for_batch, dtype=np.int64),
                vectors=embeddings_np
            )
            num_chunks_this_file = len(processed_chunks)
            progress.update(task_id, advance=1, description=f"Processed: {file_display_name} ({num_chunks_this_file} chunks)")

        except FileNotFoundError as e: # From _calculate_file_content_hash or stat()
            self.console.print(f"[bold red]Error: File not found processing {file_path}: {e}[/bold red]")
            errors_this_file += 1
            progress.update(task_id, advance=1, description=f"Skipped (Not Found): {file_display_name}")
        except RuntimeError as e: # From processor functions
            self.console.print(f"[bold red]Error: Runtime error processing {file_path}: {e}[/bold red]")
            errors_this_file += 1
            progress.update(task_id, advance=1, description=f"Skipped (Runtime Err): {file_display_name}")
        except (MetadataDBError, VectorStoreError) as e:
            self.console.print(f"[bold red]Error: Data store error for {file_path}: {e}[/bold red]")
            errors_this_file += 1
            progress.update(task_id, advance=1, description=f"Skipped (Store Err): {file_display_name}")
        except IOError as e: # General IO
            self.console.print(f"[bold red]Error: I/O error processing {file_path}: {e}[/bold red]")
            errors_this_file += 1
            progress.update(task_id, advance=1, description=f"Skipped (I/O Err): {file_display_name}")
        except Exception as e:
            self.console.print(f"[bold red]Error: Unexpected error processing {file_path}: {e}[/bold red]")
            errors_this_file += 1
            progress.update(task_id, advance=1, description=f"Skipped (Unexpected): {file_display_name}")
            # Optionally re-raise for critical unexpected errors or log traceback
            # import traceback; self.console.print(traceback.format_exc())

        return num_chunks_this_file, errors_this_file


    def index_path(self, target_path: pathlib.Path, wipe_existing: bool) -> None:
        total_files_processed = 0
        total_chunks_indexed = 0
        total_errors = 0

        try:
            self._prepare_data_stores(wipe_existing)
            if not self.db_conn or not self.usearch_index: # Guard
                raise IndexerError("Data stores were not properly initialized.")

            # File Discovery
            files_to_process: List[pathlib.Path] = []
            if target_path.is_file():
                files_to_process.append(target_path)
            elif target_path.is_dir():
                self.console.print(f"Scanning directory '{target_path}' for files matching patterns: {self.config.file_scan_patterns}...")
                found_files_set = set()
                for pattern in self.config.file_scan_patterns:
                    for file_p in target_path.rglob(pattern):
                        if file_p.is_file(): # Ensure it's a file
                           found_files_set.add(file_p.resolve()) # Resolve for uniqueness
                files_to_process = sorted(list(found_files_set))
            
            if not files_to_process:
                self.console.print(f"[yellow]No files found to index at '{target_path}' with current patterns.[/yellow]")
                return

            self.console.print(f"Found {len(files_to_process)} file(s) to process.")

            # Rich Progress Bar Setup
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
                file_processing_task = progress.add_task(
                    "[cyan]Indexing files...", total=len(files_to_process)
                )
                for file_p in files_to_process:
                    num_c, num_e = self._process_and_index_file(
                        file_p, progress, file_processing_task
                    )
                    total_chunks_indexed += num_c
                    total_errors += num_e
                    if num_e == 0 and num_c >= 0 : # Successfully processed or skipped empty/no chunks
                        total_files_processed +=1
                    # If num_e > 0, it's an error, file not fully processed.
                    # progress.update advances automatically in _process_and_index_file

            # Finalization
            if self.usearch_index is not None and len(self.usearch_index) > 0:
                self.console.print(f"Saving vector index with {len(self.usearch_index)} items...")
                save_persistent_index(self.usearch_index, self.config.usearch_index_path)
                self.console.print("Vector index saved.")
            elif self.usearch_index is not None and len(self.usearch_index) == 0:
                 self.console.print("Vector index is empty. Not saving.")
                 # Optionally delete an old index file if it exists and current one is empty after wipe
                 self.config.usearch_index_path.unlink(missing_ok=True)


            self.console.print(
                f"\n[bold green]Indexing complete for project '{self.config.project_name}'.[/bold green]"
            )
            self.console.print(
                f"  Summary: {total_files_processed} files processed, "
                f"{total_chunks_indexed} chunks indexed, {total_errors} errors encountered."
            )

        except IndexerError as e: # Catch errors from _prepare_data_stores or other Indexer logic
            self.console.print(f"[bold red]Indexer Error: {e}[/bold red]")
            raise # Re-raise for main.py to catch
        except Exception as e:
            self.console.print(f"[bold red]Unexpected critical error during indexing: {e}[/bold red]")
            # import traceback; self.console.print(traceback.format_exc())
            raise IndexerError(f"Unexpected critical error: {e}") from e # Wrap in IndexerError
        finally:
            if self.db_conn is not None:
                try:
                    # DuckDB auto-commits by default unless explicit transaction started.
                    # self.db_conn.commit() # Generally not needed for DuckDB auto-commit mode
                    self.db_conn.close()
                    self.console.print("Database connection closed.")
                except Exception as e:
                    self.console.print(f"[bold red]Error closing database connection: {e}[/bold red]")
```