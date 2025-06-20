from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console

from .config import DEFAULT_K_RESULTS, SimgrepConfig, SimgrepConfigError, load_global_config
from .core.context import SimgrepContext
from .core.models import OutputMode
from .indexer import Indexer, IndexerConfig
from .repository import MetadataStore
from .services.search_service import SearchService
from .ui.formatters import format_count, format_json, format_paths, format_show_basic
from .utils import get_ephemeral_cache_paths


class EphemeralSearcher:
    """Utility to perform ephemeral searches with disk caching."""

    def __init__(self, console: Optional[Console] = None) -> None:
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

        if not path_to_search.exists():
            self.console.print(f"[bold red]Error: Path '{path_to_search}' does not exist.[/bold red]")
            raise SystemExit(1)

        try:
            cfg = load_global_config()
        except SimgrepConfigError:
            cfg = SimgrepConfig()

        db_path, index_path = get_ephemeral_cache_paths(path_to_search, cfg, patterns)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        indexer_cfg = IndexerConfig(
            project_name="quicksearch",
            db_path=db_path,
            usearch_index_path=index_path,
            embedding_model_name=cfg.default_embedding_model_name,
            chunk_size_tokens=cfg.default_chunk_size_tokens,
            chunk_overlap_tokens=cfg.default_chunk_overlap_tokens,
            file_scan_patterns=list(patterns) if patterns else ["*.txt"],
            max_index_workers=os.cpu_count() or 4,
        )

        context = SimgrepContext.from_defaults(
            model_name=indexer_cfg.embedding_model_name,
            chunk_size=indexer_cfg.chunk_size_tokens,
            chunk_overlap=indexer_cfg.chunk_overlap_tokens,
        )

        indexer_console = self.console
        null_file = None
        if is_machine:
            null_file = open(os.devnull, "w")
            indexer_console = Console(file=null_file, force_terminal=False, color_system=None)

        indexer = Indexer(config=indexer_cfg, context=context, console=indexer_console)
        indexer.run_index(target_paths=[path_to_search], wipe_existing=False)
        if null_file:
            null_file.close()

        store = MetadataStore(persistent=True, db_path=db_path)
        vector_index = context.index_factory(context.embedder.ndim)
        if index_path.exists():
            vector_index.load(index_path)

        search_service = SearchService(store=store, embedder=context.embedder, index=vector_index)
        results = search_service.search(
            query=query_text,
            k=top,
            min_score=min_score,
            file_filter=file_filter,
            keyword_filter=keyword_filter,
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
                    self.console.print(
                        "  No relevant chunks found for your query in the processed file(s) (after filtering)."
                    )
            store.close()
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
                    self.console.print("---")
                    self.console.print(
                        format_show_basic(file_path=r.file_path, chunk_text=r.chunk_text, score=r.score)
                    )
        elif output_mode == OutputMode.json:
            print(format_json(results))
        elif output_mode == OutputMode.count_results:
            self.console.print(format_count(results))

        store.close()
