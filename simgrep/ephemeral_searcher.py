from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from .config import DEFAULT_K_RESULTS
from .core.context import SimgrepContext
from .core.models import Chunk, ChunkData, OutputMode, SearchResult
from .repository import MetadataStore
from .services.search_service import SearchService
from .ui.formatters import format_count, format_json, format_paths, format_show_basic
from .utils import gather_files_to_process


class EphemeralSearcher:
    """Utility to perform one-off searches on arbitrary files or directories."""

    def __init__(self, context: SimgrepContext, console: Optional[Console] = None) -> None:
        self.context = context
        self.console = console or Console()

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
        is_machine = output_mode in (OutputMode.json, OutputMode.paths)

        if not is_machine:
            self.console.print(f"Performing ephemeral search for: '[bold blue]{query_text}[/bold blue]' in path: '[green]{path_to_search}[/green]'")

        store: Optional[MetadataStore] = None
        try:
            if not is_machine:
                self.console.print("\n[bold]Setup: Initializing In-Memory Database[/bold]")
            store = MetadataStore()
            if not is_machine:
                self.console.print("  In-memory database and tables created.")

            if not is_machine:
                self.console.print("\n[bold]Setup: Preparing services from context[/bold]")
            extractor = self.context.extractor
            chunker = self.context.chunker
            embedder = self.context.embedder
            if not is_machine:
                self.console.print("  Services ready.")

            search_patterns = list(patterns) if patterns else ["*.txt"]
            files_to_process = gather_files_to_process(path_to_search, search_patterns)

            if not is_machine:
                if path_to_search.is_file():
                    self.console.print(f"Processing single file: [green]{path_to_search}[/green]")
                else:
                    self.console.print(f"Scanning directory: [green]{path_to_search}[/green] for files matching: {search_patterns}...")
                    if not files_to_process:
                        self.console.print(f"[yellow]No files found in directory {path_to_search} with patterns {search_patterns}[/yellow]")
                    else:
                        self.console.print(f"Found {len(files_to_process)} file(s) to process.")

            if not files_to_process:
                if not is_machine:
                    self.console.print("No files selected for processing. Exiting.")
                raise typer.Exit()

            all_chunks: List[ChunkData] = []
            label_counter = 0

            if not is_machine:
                self.console.print("\n[bold]Processing files, extracting and chunking text[/bold]")
            progress_columns = [
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            ]
            with Progress(*progress_columns, console=self.console, transient=False, disable=is_machine) as progress:
                task = progress.add_task("Processing files...", total=len(files_to_process))
                for file_idx, file_path in enumerate(files_to_process):
                    progress.update(task, description=f"Processing: {file_path.name}")
                    try:
                        text = extractor.extract(file_path)
                        if not text.strip():
                            if not is_machine:
                                self.console.print(f"    [yellow]Skipped: File '{file_path}' is empty or contains only whitespace.[/yellow]")
                            progress.advance(task)
                            continue

                        chunk_infos: List[Chunk] = list(chunker.chunk(text))
                        for c in chunk_infos:
                            all_chunks.append(
                                ChunkData(
                                    text=c.text,
                                    source_file_path=file_path,
                                    source_file_id=file_idx,
                                    usearch_label=label_counter,
                                    start_char_offset=c.start,
                                    end_char_offset=c.end,
                                    token_count=c.tokens,
                                )
                            )
                            label_counter += 1
                        if not is_machine:
                            self.console.print(f"    Extracted {len(chunk_infos)} token-based chunk(s).")
                    finally:
                        progress.advance(task)

            if not all_chunks:
                if not is_machine:
                    self.console.print("\n[yellow]No text chunks extracted from any files. Cannot perform search.[/yellow]")
                raise typer.Exit()

            if not is_machine:
                self.console.print("\n[bold]Setup: Populating In-Memory Database[/bold]")
            files_meta = {(c.source_file_id, c.source_file_path) for c in all_chunks}
            store.batch_insert_files(list(files_meta))
            store.batch_insert_chunks(all_chunks)
            if not is_machine:
                self.console.print(f"  Inserted {len(all_chunks)} chunk(s) into DB.")

            if not is_machine:
                self.console.print(f"\n[bold]Generating embeddings for {len(all_chunks)} total chunk(s)[/bold]")
            chunk_embeddings = embedder.encode(texts=[c.text for c in all_chunks], is_query=False)

            results: List[SearchResult] = []
            if chunk_embeddings.size == 0:
                if not is_machine:
                    self.console.print("  No chunk embeddings available. Skipping vector search.")
            else:
                labels_np = np.array([c.usearch_label for c in all_chunks], dtype=np.int64)
                idx = self.context.index_factory(embedder.ndim)
                idx.add(labels_np, chunk_embeddings)

                search_service = SearchService(store=store, embedder=embedder, index=idx)
                results = search_service.search(
                    query=query_text,
                    k=top,
                    min_score=min_score,
                    file_filter=file_filter,
                    keyword_filter=keyword_filter,
                )

            if not is_machine:
                self.console.print("\n[bold]Displaying results[/bold]")

            if relative_paths and output_mode != OutputMode.paths and not is_machine:
                self.console.print(
                    (
                        "[yellow]Warning: --relative-paths is only effective with --output paths. "
                        "Paths will be displayed according to the selected output mode.[/yellow]"
                    )
                )

            if not results:
                if output_mode == OutputMode.paths:
                    self.console.print(format_paths(file_paths=[], use_relative=False, base_path=None, console=self.console))
                elif output_mode == OutputMode.json:
                    self.console.print("[]")
                elif output_mode == OutputMode.count_results:
                    self.console.print(format_count([]))
                else:
                    if not is_machine:
                        self.console.print("  No relevant chunks found for your query in the processed file(s) (after filtering).")
                return

            if output_mode == OutputMode.paths:
                output_paths = [r.file_path for r in results if r.file_path]
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
                self.console.print(f"\n[bold cyan]Search Results (Top {len(results)}):[/bold cyan]")
                for r in results:
                    if r.file_path and r.chunk_text:
                        out = format_show_basic(file_path=r.file_path, chunk_text=r.chunk_text, score=r.score)
                        self.console.print("---")
                        self.console.print(out)
            elif output_mode == OutputMode.json:
                print(format_json(results))
            elif output_mode == OutputMode.count_results:
                self.console.print(format_count(results))
        finally:
            if store:
                if not is_machine:
                    self.console.print("\n[bold]Cleanup: Closing In-Memory Database[/bold]")
                store.close()
                if not is_machine:
                    self.console.print("  Database connection closed.")
