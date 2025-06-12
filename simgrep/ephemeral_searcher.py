from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional

from rich.console import Console

from .config import DEFAULT_K_RESULTS, SimgrepConfig
from .indexer import Indexer, IndexerConfig
from .metadata_store import MetadataStore
from .models import OutputMode
from .searcher import perform_persistent_search
from .utils import get_ephemeral_cache_paths
from .vector_store import load_persistent_index


class EphemeralSearcher:
    """Perform ad-hoc searches with a cached on-disk index."""

    def __init__(self, console: Optional[Console] = None, config: Optional[SimgrepConfig] = None) -> None:
        self.console = console or Console()
        self.config = config or SimgrepConfig()

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
        cfg = self.config
        search_patterns = list(patterns) if patterns else ["*.txt"]
        is_machine = output_mode in (OutputMode.json, OutputMode.paths)

        db_path, index_path = get_ephemeral_cache_paths(path_to_search, cfg)
        store: Optional[MetadataStore] = None
        index = None

        if db_path.exists() and index_path.exists():
            store = MetadataStore(persistent=True, db_path=db_path)
            index = load_persistent_index(index_path)

        if store is None or index is None:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            idx_cfg = IndexerConfig(
                project_name="ephemeral",
                db_path=db_path,
                usearch_index_path=index_path,
                embedding_model_name=cfg.default_embedding_model_name,
                chunk_size_tokens=cfg.default_chunk_size_tokens,
                chunk_overlap_tokens=cfg.default_chunk_overlap_tokens,
                file_scan_patterns=search_patterns,
            )
            index_console = self.console if not is_machine else Console(file=io.StringIO())
            indexer = Indexer(config=idx_cfg, console=index_console)
            indexer.run_index([path_to_search], wipe_existing=True)
            store = MetadataStore(persistent=True, db_path=db_path)
            index = load_persistent_index(index_path)

        if store is None or index is None:
            self.console.print(f"[bold red]Failed to prepare ephemeral index at {index_path}")
            return

        try:
            perform_persistent_search(
                query_text=query_text,
                console=self.console,
                metadata_store=store,
                vector_index=index,
                global_config=cfg,
                output_mode=output_mode,
                k_results=top,
                display_relative_paths=relative_paths,
                base_path_for_relativity=path_to_search if relative_paths else None,
                min_score=min_score,
                file_filter=file_filter,
                keyword_filter=keyword_filter,
            )
        finally:
            store.close()
