import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rich.console import Console
from rich.progress import Progress, TaskID

from simgrep.core.abstractions import (
    Embedder,
    Repository,
    TextExtractor,
    TokenChunker,
    VectorIndex,
)
from simgrep.core.models import Chunk
from simgrep.utils import calculate_file_hash, gather_files_to_process


class IndexService:
    """Encapsulates the logic for processing a single file into chunks and embeddings."""

    def __init__(
        self,
        extractor: TextExtractor,
        chunker: TokenChunker,
        embedder: Embedder,
        store: Repository,
        index: VectorIndex,
    ):
        self.extractor = extractor
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.index = index
        self._current_usearch_label: int = 0

        max_label_from_db = self.store.get_max_usearch_label()
        if max_label_from_db is not None:
            self._current_usearch_label = max_label_from_db + 1
        elif self.index is not None and len(self.index) > 0:
            # Fallback to old logic if value not in DB (for backward compatibility)
            try:
                keys = self.index.keys
                if keys.size > 0:
                    # Get the highest existing label to continue from there.
                    # USearch keys are not guaranteed to be sorted, so we find the max.
                    max_existing_label = np.max(keys)
                    self._current_usearch_label = int(max_existing_label) + 1
            except (
                ValueError,
                IndexError,
            ):  # ValueError if keys is empty, though we check len > 0
                self._current_usearch_label = 0

    @property
    def final_max_label(self) -> int:
        """Returns the highest usearch label that was actually used."""
        return self._current_usearch_label - 1

    def process_file(self, file_path: Path) -> Tuple[List[Chunk], np.ndarray]:
        """
        Processes a single file by extracting text, chunking it, and generating embeddings.

        Returns:
            A tuple containing a list of Chunk objects and a NumPy array of embeddings.
            Returns an empty list and an empty array if the file is empty or yields no text.
        """
        text_content = self.extractor.extract(file_path)
        if not text_content.strip():
            return [], np.empty((0, self.embedder.ndim), dtype=np.float32)

        chunks = list(self.chunker.chunk(text_content))
        if not chunks:
            return [], np.empty((0, self.embedder.ndim), dtype=np.float32)

        embeddings = self.embedder.encode([c.text for c in chunks], is_query=False)
        return chunks, embeddings

    def store_file_chunks(
        self,
        file_id: int,
        processed_chunks: List[Chunk],
        embeddings_np: np.ndarray,
    ) -> None:
        """
        Stores processed chunks and their embeddings into the database and vector index.
        Uses and updates an internal counter for usearch labels.

        Args:
            file_id: The database ID of the file.
            processed_chunks: A list of chunks from the file.
            embeddings_np: A numpy array of embeddings for the chunks.
        """
        chunk_db_records: List[Dict[str, Any]] = []
        usearch_labels_for_batch: List[int] = []

        for chunk_info in processed_chunks:
            current_label = self._current_usearch_label
            usearch_labels_for_batch.append(current_label)
            chunk_db_records.append(
                {
                    "file_id": file_id,
                    "usearch_label": current_label,
                    "chunk_text": chunk_info.text,
                    "start_char_offset": chunk_info.start,
                    "end_char_offset": chunk_info.end,
                    "token_count": chunk_info.tokens,
                    "embedding_hash": None,
                }
            )
            self._current_usearch_label += 1

        self.store.batch_insert_text_chunks(chunk_db_records)
        self.index.add(
            keys=np.array(usearch_labels_for_batch, dtype=np.int64),
            vecs=embeddings_np,
        )

    def run_index(
        self,
        target_paths: List[Path],
        file_scan_patterns: List[str],
        wipe_existing: bool,
        max_workers: int,
        progress: Optional[Progress] = None,
        console: Optional[Console] = None,
    ) -> Tuple[int, int, int]:
        total_files_processed = 0
        total_chunks_indexed = 0
        total_errors = 0

        _console = console or Console(quiet=True)

        all_found_files_set: set[pathlib.Path] = set()
        _console.print(f"Scanning {len(target_paths)} path(s) for files matching patterns: {file_scan_patterns}...")

        for target_path in target_paths:
            if not target_path.exists():
                _console.print(f"[yellow]Warning: Path '{target_path}' does not exist. Skipping.[/yellow]")
                continue
            # gather_files_to_process handles both files and directories, respects .gitignore,
            # and resolves paths to be absolute.
            found_files = gather_files_to_process(target_path, file_scan_patterns)
            all_found_files_set.update(found_files)

        files_to_process = sorted(list(all_found_files_set))

        existing_records = {}
        if not wipe_existing:
            existing_records = {
                Path(p).resolve(): (fid, chash) for fid, p, chash in self.store.get_all_indexed_file_records()
            }
            current_paths = {p.resolve() for p in files_to_process}
            deleted_paths = set(existing_records.keys()) - current_paths
            if deleted_paths:
                plural = "s" if len(deleted_paths) > 1 else ""
                _console.print(f"[dim]Pruning {len(deleted_paths)} deleted file{plural} from index...[/dim]")
                for del_p in deleted_paths:
                    fid, _ = existing_records[del_p]
                    removed_labels = self.store.delete_file_records(fid)
                    if self.index is not None and removed_labels:
                        self.index.remove(keys=np.array(removed_labels, dtype=np.int64))

        if not files_to_process:
            _console.print(
                "[yellow]No files found to index in any of the provided paths with current patterns.[/yellow]"
            )
            return 0, 0, 0
        else:
            _console.print(f"Found {len(files_to_process)} total file(s) to process.")

        file_processing_task: Optional[TaskID] = None
        if progress:
            file_processing_task = progress.add_task("[cyan]Indexing files...", total=len(files_to_process))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {}
            for file_p in files_to_process:
                resolved_fp = file_p.resolve()
                if not wipe_existing and resolved_fp in existing_records:
                    try:
                        current_hash = calculate_file_hash(resolved_fp)
                        stored_id, stored_hash = existing_records[resolved_fp]
                        if current_hash == stored_hash:
                            _console.print(f"[yellow]Skipped (unchanged): {file_p}[/yellow]")
                            if progress and file_processing_task is not None:
                                progress.update(
                                    file_processing_task,
                                    advance=1,
                                    description=f"Skipped (unchanged): {file_p.name}",
                                )
                            total_files_processed += 1
                            continue
                        removed_labels = self.store.delete_file_records(stored_id)
                        if self.index is not None and removed_labels:
                            self.index.remove(keys=np.array(removed_labels, dtype=np.int64))
                    except (IOError, FileNotFoundError) as e:
                        _console.print(f"[bold red]Error checking file {resolved_fp}: {e}[/bold red]")
                        if progress and file_processing_task is not None:
                            progress.update(
                                file_processing_task,
                                advance=1,
                                description=f"Skipped (error): {file_p.name}",
                            )
                        total_errors += 1
                        continue

                fut = pool.submit(self.process_file, resolved_fp)
                future_map[fut] = resolved_fp

            for fut in as_completed(future_map):
                file_path = future_map[fut]
                file_display = file_path.name
                try:
                    chunks, embeddings_np = fut.result()
                    stat = file_path.stat()
                    file_hash = calculate_file_hash(file_path)
                    file_id = self.store.insert_indexed_file_record(
                        file_path=str(file_path),
                        content_hash=file_hash,
                        file_size_bytes=stat.st_size,
                        last_modified_os_timestamp=stat.st_mtime,
                    )
                    if file_id is not None and chunks:
                        self.store_file_chunks(
                            file_id=file_id,
                            processed_chunks=chunks,
                            embeddings_np=embeddings_np,
                        )
                        total_chunks_indexed += len(chunks)
                    total_files_processed += 1
                    if progress and file_processing_task is not None:
                        progress.update(
                            file_processing_task,
                            advance=1,
                            description=f"Processed: {file_display} ({len(chunks)} chunks)",
                        )
                except Exception as e:
                    _console.print(f"[bold red]Error processing {file_path}: {e}[/bold red]")
                    total_errors += 1
                    if progress and file_processing_task is not None:
                        progress.update(
                            file_processing_task,
                            advance=1,
                            description=f"Skipped (error): {file_display}",
                        )

        return total_files_processed, total_chunks_indexed, total_errors
