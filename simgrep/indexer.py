import pathlib  # use pathlib consistently
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
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from transformers import PreTrainedTokenizerBase

from .exceptions import (
    MetadataDBError,
    VectorStoreError,
)

# assuming these are in .exceptions
from .metadata_db import (
    batch_insert_text_chunks,
    clear_persistent_project_data,
    connect_persistent_db,
    delete_file_records,
    get_all_indexed_file_records,
    insert_indexed_file_record,
)
from .processor import (
    ProcessedChunkInfo,
    calculate_file_hash,
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
        self._current_usearch_label: int = 0  # global counter for unique usearch labels

        try:
            self.console.print(f"Loading tokenizer for model: '{self.config.embedding_model_name}'...")
            self.tokenizer: PreTrainedTokenizerBase = load_tokenizer(self.config.embedding_model_name)
            self.console.print("Tokenizer loaded.")
        except RuntimeError as e:
            raise IndexerError(f"Failed to load tokenizer: {e}") from e

        try:
            self.console.print(f"Loading embedding model: '{self.config.embedding_model_name}'...")
            # Import SentenceTransformer here or at the top of the file
            from sentence_transformers import SentenceTransformer

            self.embedding_model: SentenceTransformer = SentenceTransformer(self.config.embedding_model_name)
            self.console.print("Embedding model loaded.")

            self.console.print("Determining embedding dimension...")
            dummy_emb = generate_embeddings(["simgrep_test_string"], model=self.embedding_model)
            if dummy_emb.ndim != 2 or dummy_emb.shape[0] == 0 or dummy_emb.shape[1] == 0:
                raise IndexerError(
                    f"Could not determine embedding dimension using model {self.config.embedding_model_name}. "
                    f"Dummy embedding shape: {dummy_emb.shape}"
                )
            self.embedding_ndim: int = dummy_emb.shape[1]
            self.console.print(f"Embedding dimension set to: {self.embedding_ndim}")
        except RuntimeError as e:
            raise IndexerError(f"Failed to load embedding model or generate dummy embedding for ndim: {e}") from e
        except Exception as e_model_load:  # Catch other potential errors from SentenceTransformer
            raise IndexerError(
                f"Unexpected error loading embedding model '{self.config.embedding_model_name}': {e_model_load}"
            ) from e_model_load

    def _prepare_data_stores(self, wipe_existing: bool) -> None:
        self.console.print("Preparing data stores (database and vector index)...")
        # database
        try:
            self.db_conn = connect_persistent_db(self.config.db_path)
            self.console.print(f"Connected to database: {self.config.db_path}")

            if wipe_existing:
                self.console.print(f"Wiping existing data from database: {self.config.db_path}...")
                clear_persistent_project_data(self.db_conn)
                self._current_usearch_label = 0  # reset label counter when wiping
                self.console.print("Database wiped.")
        except MetadataDBError as e:
            self.db_conn = None  # ensure it's none if setup failed
            raise IndexerError(f"Database preparation failed: {e}") from e
        except Exception as e_db_unexpected:  # catch-all for unexpected db errors
            self.db_conn = None  # ensure it's none if setup failed unexpectedly
            raise IndexerError(f"Unexpected error during database preparation: {e_db_unexpected}") from e_db_unexpected

        # vector store
        try:
            if wipe_existing:
                self.console.print(f"Wiping existing vector index: {self.config.usearch_index_path}...")
                self.config.usearch_index_path.unlink(missing_ok=True)
                self.usearch_index = usearch.index.Index(
                    ndim=self.embedding_ndim,
                    metric="cos",
                    dtype="f32",  # todo: make metric/dtype configurable
                )
                self.console.print("New empty vector index created.")
            else:  # for future incremental logic
                self.console.print(f"Loading vector index from: {self.config.usearch_index_path}...")
                self.usearch_index = load_persistent_index(self.config.usearch_index_path)
                if self.usearch_index is None:
                    self.console.print("No existing vector index found. Creating new one.")
                    self.usearch_index = usearch.index.Index(ndim=self.embedding_ndim, metric="cos", dtype="f32")
                else:
                    self.console.print(f"Loaded existing vector index with {len(self.usearch_index)} items.")
                    if len(self.usearch_index) > 0:
                        # The USearch index is iterable and yields keys
                        try:
                            max_existing_label = max(self.usearch_index.keys)
                            self._current_usearch_label = max_existing_label + 1
                        except ValueError:  # Handles case where index is empty despite len > 0
                            # (should not happen) or contains non-numeric keys (not expected)
                            self.console.print(
                                "[yellow]Warning: Could not determine max key from existing index, "
                                "starting labels from 0.[/yellow]"
                            )
                            self._current_usearch_label = 0
                    else:
                        self._current_usearch_label = 0
        except VectorStoreError as e:
            self.usearch_index = None  # ensure it's none if setup failed
            raise IndexerError(f"Vector store preparation failed: {e}") from e
        except Exception as e_vs_unexpected:  # catch-all for unexpected vs errors
            self.usearch_index = None  # ensure it's none if setup failed
            raise IndexerError(f"Unexpected error during vector store preparation: {e_vs_unexpected}") from e_vs_unexpected

        # final checks
        if self.usearch_index is None:
            raise IndexerError("USearch index is None at the end of _prepare_data_stores.")
        if self.db_conn is None:
            raise IndexerError("DB connection is None at the end of _prepare_data_stores.")

    def _process_and_index_file(self, file_path: pathlib.Path, progress: Progress, task_id: TaskID) -> Tuple[int, int]:
        num_chunks_this_file = 0
        errors_this_file = 0
        file_display_name = file_path.name

        progress.update(task_id, description=f"Processing: {file_display_name}...")

        if not self.db_conn:  # should be set by _prepare_data_stores
            self.console.print(f"[bold red]Error: DB connection not available for file {file_path}. Skipping.[/bold red]")
            errors_this_file += 1
            progress.update(
                task_id,
                advance=1,
                description=f"Skipped (DB error): {file_display_name}",
            )
            return num_chunks_this_file, errors_this_file

        try:
            # file metadata
            content_hash = calculate_file_hash(file_path)
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
                progress.update(
                    task_id,
                    advance=1,
                    description=f"Skipped (DB insert): {file_display_name}",
                )
                return num_chunks_this_file, errors_this_file

            # text extraction & chunking
            text_content = extract_text_from_file(file_path)
            if not text_content.strip():
                self.console.print(f"[yellow]Info: File {file_path} is empty or whitespace only. Skipping chunking.[/yellow]")
                progress.update(
                    task_id,
                    advance=1,
                    description=f"Processed (empty): {file_display_name}",
                )
                return num_chunks_this_file, errors_this_file

            processed_chunks: List[ProcessedChunkInfo] = chunk_text_by_tokens(
                text_content,
                self.tokenizer,
                self.config.chunk_size_tokens,
                self.config.chunk_overlap_tokens,
            )

            if not processed_chunks:
                self.console.print(f"[yellow]Info: No chunks generated for {file_path}.[/yellow]")
                progress.update(
                    task_id,
                    advance=1,
                    description=f"Processed (no chunks): {file_display_name}",
                )
                return num_chunks_this_file, errors_this_file

            # embedding & storing chunks
            chunk_texts = [chunk["text"] for chunk in processed_chunks]
            embeddings_np = generate_embeddings(chunk_texts, model=self.embedding_model)

            chunk_db_records: List[Dict[str, Any]] = []
            usearch_labels_for_batch: List[int] = []

            for i, chunk_info in enumerate(processed_chunks):
                current_label = self._current_usearch_label
                usearch_labels_for_batch.append(current_label)

                chunk_db_records.append(
                    {
                        "file_id": file_id,
                        "usearch_label": current_label,
                        "chunk_text_snippet": chunk_info["text"][:255],  # truncate for snippet
                        "start_char_offset": chunk_info["start_char_offset"],
                        "end_char_offset": chunk_info["end_char_offset"],
                        "token_count": chunk_info["token_count"],
                        "embedding_hash": None,  # placeholder for v1
                    }
                )
                self._current_usearch_label += 1

            batch_insert_text_chunks(self.db_conn, chunk_db_records)

            if self.usearch_index is None:  # should be initialized
                raise IndexerError("USearch index is None during file processing.")
            self.usearch_index.add(
                keys=np.array(usearch_labels_for_batch, dtype=np.int64),
                vectors=embeddings_np,
            )
            num_chunks_this_file = len(processed_chunks)
            progress.update(
                task_id,
                advance=1,
                description=f"Processed: {file_display_name} ({num_chunks_this_file} chunks)",
            )

        except FileNotFoundError as e:  # from calculate_file_hash or stat()
            self.console.print(f"[bold red]Error: File not found processing {file_path}: {e}[/bold red]")
            errors_this_file += 1
            progress.update(
                task_id,
                advance=1,
                description=f"Skipped (Not Found): {file_display_name}",
            )
        except RuntimeError as e:  # from processor functions
            self.console.print(f"[bold red]Error: Runtime error processing {file_path}: {e}[/bold red]")
            errors_this_file += 1
            progress.update(
                task_id,
                advance=1,
                description=f"Skipped (Runtime Err): {file_display_name}",
            )
        except (MetadataDBError, VectorStoreError) as e:
            self.console.print(f"[bold red]Error: Data store error for {file_path}: {e}[/bold red]")
            errors_this_file += 1
            progress.update(
                task_id,
                advance=1,
                description=f"Skipped (Store Err): {file_display_name}",
            )
        except IOError as e:  # general io
            self.console.print(f"[bold red]Error: I/O error processing {file_path}: {e}[/bold red]")
            errors_this_file += 1
            progress.update(
                task_id,
                advance=1,
                description=f"Skipped (I/O Err): {file_display_name}",
            )
        except Exception as e:
            self.console.print(f"[bold red]Error: Unexpected error processing {file_path}: {e}[/bold red]")
            errors_this_file += 1
            progress.update(
                task_id,
                advance=1,
                description=f"Skipped (Unexpected): {file_display_name}",
            )
            # optionally re-raise for critical unexpected errors or log traceback
            # import traceback; self.console.print(traceback.format_exc())

        return num_chunks_this_file, errors_this_file

    def index_path(self, target_path: pathlib.Path, wipe_existing: bool) -> None:
        total_files_processed = 0
        total_chunks_indexed = 0
        total_errors = 0

        try:
            self._prepare_data_stores(wipe_existing)
            if self.db_conn is None or self.usearch_index is None:  # guard
                raise IndexerError("Data stores were not properly initialized (db_conn or usearch_index is None).")

            # file discovery
            files_to_process: List[pathlib.Path] = []
            if target_path.is_file():
                files_to_process.append(target_path)
            elif target_path.is_dir():
                self.console.print(
                    f"Scanning directory '{target_path}' for files matching patterns: {self.config.file_scan_patterns}..."
                )
                found_files_set = set()
                for pattern in self.config.file_scan_patterns:
                    for file_p in target_path.rglob(pattern):
                        if file_p.is_file():
                            found_files_set.add(file_p.resolve())
                files_to_process = sorted(list(found_files_set))

            if not files_to_process:
                self.console.print(f"[yellow]No files found to index at '{target_path}' with current patterns.[/yellow]")
                # removed early return here to allow summary to print
            else:
                self.console.print(f"Found {len(files_to_process)} file(s) to process.")

            existing_records: Dict[pathlib.Path, Tuple[int, str]] = {}
            if not wipe_existing:
                try:
                    assert self.db_conn is not None
                    for fid, path_str, chash in get_all_indexed_file_records(self.db_conn):
                        existing_records[pathlib.Path(path_str).resolve()] = (fid, chash)
                except MetadataDBError as e:
                    raise IndexerError(f"Failed to fetch existing file records: {e}") from e

                existing_paths = set(existing_records.keys())
                current_paths = set(p.resolve() for p in files_to_process)
                deleted_paths = existing_paths - current_paths
                for del_p in deleted_paths:
                    fid, _ = existing_records[del_p]
                    removed_labels = delete_file_records(self.db_conn, fid)
                    if self.usearch_index is not None and removed_labels:
                        self.usearch_index.remove(keys=np.array(removed_labels, dtype=np.int64))

            # rich progress bar setup
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
                file_processing_task = progress.add_task("[cyan]Indexing files...", total=len(files_to_process))
                for file_p in files_to_process:
                    resolved_fp = file_p.resolve()
                    if not wipe_existing and resolved_fp in existing_records:
                        try:
                            current_hash = calculate_file_hash(resolved_fp)
                        except FileNotFoundError:
                            progress.update(
                                file_processing_task,
                                advance=1,
                                description=f"Skipped (missing): {file_p.name}",
                            )
                            total_errors += 1
                            continue

                        stored_id, stored_hash = existing_records[resolved_fp]
                        if current_hash == stored_hash:
                            progress.update(
                                file_processing_task,
                                advance=1,
                                description=f"Skipped (unchanged): {file_p.name}",
                            )
                            total_files_processed += 1
                            continue
                        removed_labels = delete_file_records(self.db_conn, stored_id)
                        if self.usearch_index is not None and removed_labels:
                            self.usearch_index.remove(keys=np.array(removed_labels, dtype=np.int64))

                    num_c, num_e = self._process_and_index_file(resolved_fp, progress, file_processing_task)
                    total_chunks_indexed += num_c
                    total_errors += num_e
                    if num_e == 0 and num_c >= 0:  # successfully processed or skipped empty/no chunks
                        total_files_processed += 1
                    # if num_e > 0, it's an error, file not fully processed.
                    # progress.update advances automatically in _process_and_index_file

            # finalization
            if self.usearch_index is not None and len(self.usearch_index) > 0:
                self.console.print(f"Saving vector index with {len(self.usearch_index)} items...")
                save_persistent_index(self.usearch_index, self.config.usearch_index_path)
                self.console.print("Vector index saved.")
            elif self.usearch_index is not None and len(self.usearch_index) == 0:
                self.console.print("Vector index is empty. Not saving.")
                # optionally delete an old index file if it exists and current one is empty after wipe
                self.config.usearch_index_path.unlink(missing_ok=True)

            self.console.print(f"\n[bold green]Indexing complete for project '{self.config.project_name}'.[/bold green]")
            self.console.print(
                f"  Summary: {total_files_processed} files processed, "
                f"{total_chunks_indexed} chunks indexed, {total_errors} errors encountered."
            )

        except IndexerError as e:  # catch errors from _prepare_data_stores or other indexer logic
            self.console.print(f"[bold red]Indexer Error: {e}[/bold red]")
            raise  # re-raise for main.py to catch
        except Exception as e:
            self.console.print(f"[bold red]Unexpected critical error during indexing: {e}[/bold red]")
            # import traceback; self.console.print(traceback.format_exc())
            raise IndexerError(f"Unexpected critical error: {e}") from e  # wrap in indexererror
        finally:
            if self.db_conn is not None:
                try:
                    # duckdb auto-commits by default unless explicit transaction started.
                    # self.db_conn.commit() # generally not needed for duckdb auto-commit mode
                    self.db_conn.close()
                except Exception as e:
                    self.console.print(f"[bold red]Error closing database connection: {e}[/bold red]")
