import pathlib
from typing import List, Optional, Tuple

import usearch.index
from rich.console import Console

from .config import DEFAULT_K_RESULTS, SimgrepConfig
from .exceptions import MetadataDBError, VectorStoreError
from .formatter import format_paths, format_show_basic
from .metadata_store import PersistentMetadataStore
from .models import OutputMode
from .processor import generate_embeddings
from .vector_store import (
    search_inmemory_index,  # this function is generic for any usearch index
)


def perform_persistent_search(
    query_text: str,
    console: Console,
    metadata_store: PersistentMetadataStore,
    vector_index: usearch.index.Index,
    global_config: SimgrepConfig,
    output_mode: OutputMode,
    k_results: int = DEFAULT_K_RESULTS,
    display_relative_paths: bool = False,
    base_path_for_relativity: Optional[pathlib.Path] = None,
    min_score: float = 0.1,
) -> None:
    """
    Orchestrates the search process against a pre-existing, loaded persistent index.
    """
    embedding_model_name = global_config.default_embedding_model_name
    console.print(f"  Embedding query: '[italic blue]{query_text}[/italic blue]' using model '{embedding_model_name}'...")
    try:
        query_embedding = generate_embeddings(texts=[query_text], model_name=embedding_model_name)
    except RuntimeError as e:
        console.print(f"[bold red]Failed to generate query embedding:[/bold red]\n  {e}")
        raise  # re-raise for main.py to catch and exit

    console.print(f"  Searching persistent index for top {k_results} similar chunks...")
    try:
        search_matches: List[Tuple[int, float]] = search_inmemory_index(
            index=vector_index, query_embedding=query_embedding, k=k_results
        )
    except (VectorStoreError, ValueError) as e:
        console.print(f"[bold red]Error during vector search:[/bold red]\n  {e}")
        raise  # re-raise

    # Filter by min_score
    filtered_matches = [(label, score) for (label, score) in search_matches if score >= min_score]

    if not filtered_matches:
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
        else:  # outputmode.show
            console.print("  No relevant chunks found in the persistent index.")
        return

    # process and format results
    if output_mode == OutputMode.show:
        console.print(f"\n[bold cyan]Search Results (Top {len(filtered_matches)} from persistent index):[/bold cyan]")
        for matched_usearch_label, similarity_score in filtered_matches:
            try:
                retrieved_details = metadata_store.retrieve_chunk_details_persistent(matched_usearch_label)
            except MetadataDBError as e:
                console.print(
                    f"[yellow]Warning: Database error retrieving details for chunk label {matched_usearch_label}: {e}[/yellow]"
                )
                continue  # skip this result

            if retrieved_details:
                text_snippet, file_path_obj, _start, _end = retrieved_details
                output_string = format_show_basic(
                    file_path=file_path_obj,
                    chunk_text=text_snippet,  # using snippet from db for now
                    score=similarity_score,
                )
                console.print("---")
                console.print(output_string)
            else:
                console.print(
                    f"[yellow]Warning: Could not retrieve details for chunk label {matched_usearch_label} from DB.[/yellow]"
                )
    elif output_mode == OutputMode.paths:
        paths_from_matches: List[pathlib.Path] = []
        unique_paths_seen = set()  # to ensure uniqueness before format_paths
        for matched_usearch_label, _similarity_score in filtered_matches:
            try:
                retrieved_details = metadata_store.retrieve_chunk_details_persistent(matched_usearch_label)
            except MetadataDBError as e:
                console.print(
                    f"[yellow]Warning: Database error retrieving path for chunk label {matched_usearch_label}: {e}[/yellow]"
                )
                continue

            if retrieved_details:
                file_path_obj = retrieved_details[1]  # the file_path_obj
                if file_path_obj not in unique_paths_seen:
                    paths_from_matches.append(file_path_obj)
                    unique_paths_seen.add(file_path_obj)
            else:
                console.print(f"[yellow]Warning: Could not retrieve path for chunk label {matched_usearch_label}.[/yellow]")

        output_string = format_paths(
            file_paths=paths_from_matches,  # already unique
            use_relative=display_relative_paths,
            base_path=base_path_for_relativity,
            console=console,
        )
        # format_paths itself handles "no matching files found." if paths_from_matches is empty.
        console.print(output_string)
