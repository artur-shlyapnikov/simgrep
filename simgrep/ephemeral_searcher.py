from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from .config import DEFAULT_K_RESULTS, SimgrepConfig
from .formatter import format_count, format_json, format_paths, format_show_basic
from .metadata_store import MetadataStore
from .models import ChunkData, OutputMode, SearchResult
from .utils import gather_files_to_process
from .vector_store import create_inmemory_index, search_inmemory_index

chunk_text_by_tokens: Optional[Any] = None
extract_text_from_file: Optional[Any] = None
generate_embeddings: Optional[Any] = None
load_embedding_model: Optional[Any] = None
load_tokenizer: Optional[Any] = None
ProcessedChunkInfo: Optional[Any] = None


class EphemeralSearcher:
    """Utility to perform one-off searches on arbitrary files or directories."""

    def __init__(self, console: Optional[Console] = None, config: Optional[SimgrepConfig] = None, *, quiet: bool = False) -> None:
        self.console = console or Console()
        self.config = config or SimgrepConfig()
        self.quiet = quiet

    def _log(self, message: str) -> None:
        if not self.quiet:
            self.console.print(message)

    def search(
        self,
        query_text: str,
        path_to_search: Path,
        *,
        patterns: Optional[List[str]] = None,
        output_mode: OutputMode = OutputMode.show,
        top: int = DEFAULT_K_RESULTS,
        relative_paths: bool = False,
        min_score: float = 0.1,
        file_filter: Optional[List[str]] = None,
        keyword_filter: Optional[str] = None,
    ) -> None:
        """Run an ephemeral search and print results to the console."""
        global chunk_text_by_tokens, extract_text_from_file, generate_embeddings, load_embedding_model, load_tokenizer, ProcessedChunkInfo
        if any(
            f is None
            for f in (
                chunk_text_by_tokens,
                extract_text_from_file,
                generate_embeddings,
                load_embedding_model,
                load_tokenizer,
            )
        ):
            from .processor import (
                ProcessedChunkInfo as _ProcessedChunkInfo,
            )
            from .processor import (
                chunk_text_by_tokens as _chunk_text_by_tokens,
            )
            from .processor import (
                extract_text_from_file as _extract_text_from_file,
            )
            from .processor import (
                generate_embeddings as _generate_embeddings,
            )
            from .processor import (
                load_embedding_model as _load_embedding_model,
            )
            from .processor import (
                load_tokenizer as _load_tokenizer,
            )

            if chunk_text_by_tokens is None:
                chunk_text_by_tokens = _chunk_text_by_tokens
            if extract_text_from_file is None:
                extract_text_from_file = _extract_text_from_file
            if generate_embeddings is None:
                generate_embeddings = _generate_embeddings
            if load_embedding_model is None:
                load_embedding_model = _load_embedding_model
            if load_tokenizer is None:
                load_tokenizer = _load_tokenizer
            if ProcessedChunkInfo is None:
                ProcessedChunkInfo = _ProcessedChunkInfo


        is_machine = output_mode in (OutputMode.json, OutputMode.paths)
        cfg = self.config

        if not is_machine:
            self._log(f"Performing ephemeral search for: '[bold blue]{query_text}[/bold blue]' in path: '[green]{path_to_search}[/green]'")

        store: Optional[MetadataStore] = None
        try:
            if not is_machine:
                self._log("\n[bold]Setup: Initializing In-Memory Database[/bold]")
            store = MetadataStore()
            if not is_machine:
                self._log("  In-memory database and tables created.")

            if not is_machine:
                self._log("\n[bold]Setup: Loading Tokenizer[/bold]")
                self._log(f"  Loading tokenizer for model: '{cfg.default_embedding_model_name}'...")
            tokenizer = load_tokenizer(cfg.default_embedding_model_name)
            if not is_machine:
                self._log(f"    Tokenizer loaded successfully: {tokenizer.__class__.__name__}")

            search_patterns = list(patterns) if patterns else ["*.txt"]
            files_to_process = gather_files_to_process(path_to_search, search_patterns)

            if not is_machine:
                if path_to_search.is_file():
                    self._log(f"Processing single file: [green]{path_to_search}[/green]")
                else:
                    self._log(f"Scanning directory: [green]{path_to_search}[/green] for files matching: {search_patterns}...")
                    if not files_to_process:
                        self._log(f"[yellow]No files found in directory {path_to_search} with patterns {search_patterns}[/yellow]")
                    else:
                        self._log(f"Found {len(files_to_process)} file(s) to process.")

            if not files_to_process:
                if not is_machine:
                    self._log("No files selected for processing. Exiting.")
                raise typer.Exit()

            all_chunks: List[ChunkData] = []
            label_counter = 0

            if not is_machine:
                self._log("\n[bold]Step 1 & 2: Processing files, extracting and chunking text (token-based)[/bold]")
            progress_columns = [
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            ]
            with Progress(*progress_columns, console=self.console, transient=False, disable=is_machine or self.quiet) as progress:
                task = progress.add_task("Processing files...", total=len(files_to_process))
                for file_idx, file_path in enumerate(files_to_process):
                    progress.update(task, description=f"Processing: {file_path.name}")
                    try:
                        text = extract_text_from_file(file_path)
                        if not text.strip():
                            if not is_machine:
                                self._log(f"    [yellow]Skipped: File '{file_path}' is empty or contains only whitespace.[/yellow]")
                            progress.advance(task)
                            continue

                        chunk_infos: List[ProcessedChunkInfo] = chunk_text_by_tokens(
                            full_text=text,
                            tokenizer=tokenizer,
                            chunk_size_tokens=cfg.default_chunk_size_tokens,
                            overlap_tokens=cfg.default_chunk_overlap_tokens,
                        )
                        for c in chunk_infos:
                            all_chunks.append(
                                ChunkData(
                                    text=c["text"],
                                    source_file_path=file_path,
                                    source_file_id=file_idx,
                                    usearch_label=label_counter,
                                    start_char_offset=c["start_char_offset"],
                                    end_char_offset=c["end_char_offset"],
                                    token_count=c["token_count"],
                                )
                            )
                            label_counter += 1
                        if not is_machine:
                            self._log(f"    Extracted {len(chunk_infos)} token-based chunk(s).")
                    finally:
                        progress.advance(task)

            if not all_chunks:
                if not is_machine:
                    self._log("\n[yellow]No text chunks extracted from any files. Cannot perform search.[/yellow]")
                raise typer.Exit()

            if not is_machine:
                self._log("\n[bold]Setup: Populating In-Memory Database[/bold]")
            files_meta = {(c.source_file_id, c.source_file_path) for c in all_chunks}
            store.batch_insert_files(list(files_meta))
            store.batch_insert_chunks(all_chunks)
            if not is_machine:
                self._log(f"  Inserted {len(all_chunks)} chunk(s) into DB.")

            if not is_machine:
                self._log(f"\n[bold]Step 3: Generating Embeddings for {len(all_chunks)} total chunk(s)[/bold]")
            model = load_embedding_model(cfg.default_embedding_model_name)
            query_embedding = generate_embeddings(
                texts=[query_text],
                model_name=cfg.default_embedding_model_name,
                model=model,
                is_query=True,
            )
            chunk_embeddings = generate_embeddings(
                texts=[c.text for c in all_chunks],
                model_name=cfg.default_embedding_model_name,
                model=model,
                is_query=False,
            )

            if chunk_embeddings.size == 0:
                if not is_machine:
                    self._log("  No chunk embeddings available. Skipping vector search.")
                search_matches: List[SearchResult] = []
            else:
                labels_np = np.array([c.usearch_label for c in all_chunks], dtype=np.int64)
                idx = create_inmemory_index(embeddings=chunk_embeddings, labels_for_usearch=labels_np)
                search_matches = search_inmemory_index(index=idx, query_embedding=query_embedding, k=top)

            if not is_machine:
                self._log("\n[bold]Step 5: Displaying Results[/bold]")

            if relative_paths and output_mode != OutputMode.paths and not is_machine:
                self._log(
                    (
                        "[yellow]Warning: --relative-paths is only effective with --output paths. "
                        "Paths will be displayed according to the selected output mode.[/yellow]"
                    )
                )

            if not search_matches:
                if output_mode == OutputMode.paths:
                    self._log(format_paths(file_paths=[], use_relative=False, base_path=None, console=self.console))
                elif output_mode == OutputMode.json:
                    self._log("[]")
                elif output_mode == OutputMode.count_results:
                    self._log(format_count([]))
                else:
                    if not is_machine:
                        self._log("  No relevant chunks found for your query in the processed file(s).")
                return

            results: List[Dict[str, Any]] = []
            for match in search_matches:
                if match.score < min_score:
                    continue
                details = store.retrieve_chunk_for_display(match.label)
                if details:
                    txt, pth, start_off, end_off = details
                    results.append(
                        {
                            "file_path": pth,
                            "chunk_text": txt,
                            "score": match.score,
                            "start_char_offset": start_off,
                            "end_char_offset": end_off,
                            "usearch_label": match.label,
                        }
                    )

            results.sort(key=lambda r: r["score"], reverse=True)

            if not results:
                if output_mode == OutputMode.paths:
                    print(format_paths(file_paths=[], use_relative=False, base_path=None, console=self.console))
                elif output_mode == OutputMode.json:
                    print("[]")
                elif output_mode == OutputMode.count_results:
                    print(format_count([]))
                else:
                    if not is_machine and not self.quiet:
                        self.console.print("  No relevant chunks found for your query in the processed file(s) (after filtering).")
                return

            if output_mode == OutputMode.paths:
                output_paths = [r["file_path"] for r in results]
                base: Optional[Path] = None
                if relative_paths:
                    base = path_to_search if path_to_search.is_dir() else path_to_search.parent
                out_str = format_paths(
                    file_paths=output_paths,
                    use_relative=relative_paths,
                    base_path=base,
                    console=self.console,
                )
                print(out_str)
            elif output_mode == OutputMode.show:
                header = f"Search Results (Top {len(results)}):"
                if self.quiet:
                    print(header)
                    for r in results:
                        print("---")
                        print(format_show_basic(file_path=r["file_path"], chunk_text=r["chunk_text"], score=r["score"]))
                else:
                    self.console.print(f"\n[bold cyan]{header}[/bold cyan]")
                    for r in results:
                        out = format_show_basic(file_path=r["file_path"], chunk_text=r["chunk_text"], score=r["score"])
                        self.console.print("---")
                        self.console.print(out)
            elif output_mode == OutputMode.json:
                print(format_json(results))
            elif output_mode == OutputMode.count_results:
                if self.quiet:
                    print(format_count(results))
                else:
                    self.console.print(format_count(results))
        finally:
            if store:
                if not is_machine:
                    self._log("\n[bold]Cleanup: Closing In-Memory Database[/bold]")
                store.close()
                if not is_machine:
                    self._log("  Database connection closed.")
