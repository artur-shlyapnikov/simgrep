import pathlib
from typing import Dict, List, Optional

import usearch.index
from rich.console import Console

from .config import DEFAULT_K_RESULTS, SimgrepConfig
from .exceptions import MetadataDBError, VectorStoreError
from .formatter import format_count, format_json, format_paths, format_show_basic
from .metadata_store import MetadataStore
from .models import OutputMode, SearchResult
from .processor import generate_embeddings
from .vector_store import (
    search_inmemory_index,  # this function is generic for any usearch index
)


def perform_persistent_search(
    query_text: str,
    console: Console,
    metadata_store: MetadataStore,
    vector_index: usearch.index.Index,
    global_config: SimgrepConfig,
    output_mode: OutputMode,
    k_results: int = DEFAULT_K_RESULTS,
    display_relative_paths: bool = False,
    base_path_for_relativity: Optional[pathlib.Path] = None,
    min_score: float = 0.1,
    file_filter: Optional[List[str]] = None,
    keyword_filter: Optional[str] = None,
) -> None:
    """
    Orchestrates the search process against a pre-existing, loaded persistent index.
    """
    embedding_model_name = global_config.default_embedding_model_name
    if output_mode != OutputMode.json:
        console.print(f"  Embedding query: '[italic blue]{query_text}[/italic blue]' using model '{embedding_model_name}'...")
    try:
        query_embedding = generate_embeddings(texts=[query_text], model_name=embedding_model_name, is_query=True)
    except RuntimeError as e:
        console.print(f"[bold red]Failed to generate query embedding:[/bold red]\n  {e}")
        raise  # re-raise for main.py to catch and exit

    if output_mode != OutputMode.json:
        console.print(f"  Searching persistent index for top {k_results} similar chunks...")
    try:
        search_matches: List[SearchResult] = search_inmemory_index(index=vector_index, query_embedding=query_embedding, k=k_results)
    except (VectorStoreError, ValueError) as e:
        console.print(f"[bold red]Error during vector search:[/bold red]\n  {e}")
        raise  # re-raise

    if not search_matches:
        if output_mode == OutputMode.paths:
            # format_paths handles "no matching files found."
            console.print(
                format_paths(
                    file_paths=[],
                    use_relative=display_relative_paths,
                    base_path=base_path_for_relativity,
                    console=console,
                )
            )
        elif output_mode == OutputMode.json:
            console.print("[]")
        elif output_mode == OutputMode.count_results:
            console.print(format_count([]))
        else:  # outputmode.show
            console.print("  No relevant chunks found in the persistent index.")
        return

    label_to_score: Dict[int, float] = {m.label: m.score for m in search_matches}
    usearch_labels = list(label_to_score.keys())

    try:
        filtered_db_results = metadata_store.retrieve_filtered_chunk_details(
            usearch_labels=usearch_labels,
            file_filter=file_filter,
            keyword_filter=keyword_filter,
        )
    except MetadataDBError as e:
        console.print(f"[yellow]Warning: Database error retrieving chunk details: {e}[/yellow]")
        if output_mode == OutputMode.paths:
            console.print(format_paths(file_paths=[], use_relative=False, base_path=None, console=console))
        else:
            console.print("  Could not retrieve chunk details from database.")
        return

    # Combine DB results with scores and filter by min_score
    final_results = []
    for record in filtered_db_results:
        score = label_to_score.get(record["usearch_label"])
        if score is not None and score >= min_score:
            record["score"] = score
            final_results.append(record)

    # Sort by score descending
    final_results.sort(key=lambda r: r["score"], reverse=True)

    if not final_results:
        if output_mode == OutputMode.paths:
            console.print(
                format_paths(
                    file_paths=[],
                    use_relative=display_relative_paths,
                    base_path=base_path_for_relativity,
                    console=console,
                )
            )
        elif output_mode == OutputMode.json:
            console.print("[]")
        elif output_mode == OutputMode.count:
            console.print(format_count([]))
        else:  # outputmode.show
            console.print("  No relevant chunks found in the persistent index (after filtering).")
        return

    # process and format results
    if output_mode == OutputMode.show:
        console.print(f"\n[bold cyan]Search Results (Top {len(final_results)} from persistent index):[/bold cyan]")
        for result in final_results:
            output_string = format_show_basic(
                file_path=result["file_path"],
                chunk_text=result["chunk_text"],
                score=result["score"],
            )
            console.print("---")
            console.print(output_string)
    elif output_mode == OutputMode.paths:
        paths_from_matches: List[pathlib.Path] = []
        unique_paths_seen = set()
        for result in final_results:
            file_path_obj = result["file_path"]
            if file_path_obj not in unique_paths_seen:
                paths_from_matches.append(file_path_obj)
                unique_paths_seen.add(file_path_obj)

        output_string = format_paths(
            file_paths=paths_from_matches,
            use_relative=display_relative_paths,
            base_path=base_path_for_relativity,
            console=console,
        )
        console.print(output_string)
    elif output_mode == OutputMode.json:
        # Use a direct print for JSON to avoid Rich's wrapping
        print(format_json(final_results))
    elif output_mode == OutputMode.count_results:
        console.print(format_count(final_results))