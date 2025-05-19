import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any # Added Dict, Any

import numpy as np
import typer
from rich.console import Console
import usearch.index

try:
    from .processor import chunk_text_simple, extract_text_from_file, generate_embeddings
    from .vector_store import create_inmemory_index, search_inmemory_index
    from .formatter import format_show_basic
except ImportError:
    if __name__ == "__main__":
        import os
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from simgrep.processor import chunk_text_simple, extract_text_from_file, generate_embeddings
        from simgrep.vector_store import create_inmemory_index, search_inmemory_index
        from simgrep.formatter import format_show_basic
    else:
        raise

warnings.filterwarnings(
    "ignore",
    message=(
        "libmagic is unavailable but assists in filetype detection. "
        "Please consider installing libmagic for better results."
    ),
)

__version__ = "0.1.0"

app = typer.Typer()
console = Console()


def version_callback(value: bool) -> None:
    if value:
        console.print(f"simgrep version: {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: bool = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """
    simgrep CLI application.
    """
    pass


@app.command()
def search(
    query_text: str = typer.Argument(..., help="The text or concept to search for."),
    path_to_search: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=True, # Changed to True for D2.1
        readable=True,
        resolve_path=True,
        help="The path to the text file or directory to search within.",
    ),
) -> None:
    """
    Searches for a query within a specified text file or files in a directory.
    """
    console.print(f"Searching for: '[bold blue]{query_text}[/bold blue]' in path: '[green]{path_to_search}[/green]'")

    files_to_process: List[Path] = []
    files_skipped: List[Tuple[Path, str]] = []

    if path_to_search.is_file():
        files_to_process.append(path_to_search)
        console.print(f"Processing single file: [green]{path_to_search}[/green]")
    elif path_to_search.is_dir():
        console.print(f"Scanning directory: [green]{path_to_search}[/green] for '.txt' files...")
        discovered_files_generator = path_to_search.rglob("*.txt")
        files_to_process = list(discovered_files_generator)
        if not files_to_process:
            console.print(f"[yellow]No '.txt' files found in directory: {path_to_search}[/yellow]")
            # Allow proceeding, will result in "no chunks found" later
        else:
            console.print(f"Found {len(files_to_process)} '.txt' file(s) to process.")
    # Typer's path validation (exists, file_okay, dir_okay) should handle other cases.

    if not files_to_process:
        console.print("No files selected for processing. Exiting.")
        # Subsequent logic will naturally lead to no chunks being processed or found.
        # If an explicit exit is desired here, uncomment:
        # raise typer.Exit()

    aggregated_chunk_metadata: List[Dict[str, Any]] = []
    all_chunk_texts_for_embedding: List[str] = []

    # Define chunking parameters (hardcoded for now as per D1.2)
    chunk_size_chars: int = 200
    overlap_chars: int = 50

    console.print("\n[bold]Step 1 & 2: Processing files, extracting and chunking text[/bold]")
    if files_to_process: # Only proceed if there are files
        for file_idx, file_path_item in enumerate(files_to_process):
            console.print(f"  ({file_idx + 1}/{len(files_to_process)}) Processing: [dim]{file_path_item}[/dim]")
            try:
                extracted_content = extract_text_from_file(file_path_item)
                if not extracted_content.strip():
                    console.print(f"    [yellow]Skipped: File '{file_path_item}' is empty or contains only whitespace.[/yellow]")
                    files_skipped.append((file_path_item, "Empty or whitespace-only"))
                    continue

                text_chunks_for_file: List[str] = chunk_text_simple(
                    text=extracted_content,
                    chunk_size_chars=chunk_size_chars,
                    overlap_chars=overlap_chars,
                )

                if text_chunks_for_file:
                    for chunk_idx_in_file, chunk_text_item in enumerate(text_chunks_for_file):
                        all_chunk_texts_for_embedding.append(chunk_text_item)
                        aggregated_chunk_metadata.append({
                            'text': chunk_text_item,
                            'source_path': file_path_item,
                            'original_chunk_index_in_file': chunk_idx_in_file
                        })
                    console.print(f"    Extracted {len(text_chunks_for_file)} chunk(s).")
                else:
                    console.print(f"    [yellow]No chunks generated for '{file_path_item}' (text might be too short for chunking parameters).[/yellow]")
                    files_skipped.append((file_path_item, "No chunks generated (too short for parameters)"))

            except FileNotFoundError: # Should be caught by Typer or initial checks, but as a safeguard
                console.print(f"    [bold red]Error: File not found during processing loop: {file_path_item}. Skipping.[/bold red]")
                files_skipped.append((file_path_item, "File not found during processing"))
            except RuntimeError as e: # Catch errors from extract_text_from_file or chunk_text_simple
                console.print(f"    [bold red]Error processing file '{file_path_item}': {e}. Skipping.[/bold red]")
                files_skipped.append((file_path_item, str(e)))
            except ValueError as ve: # Catch errors from chunk_text_simple parameters
                console.print(f"    [bold red]Error chunking file '{file_path_item}': {ve}. Skipping.[/bold red]")
                files_skipped.append((file_path_item, str(ve)))
            except Exception as e: # Catch-all for unexpected errors per file
                console.print(f"    [bold red]Unexpected error processing file '{file_path_item}': {e}. Skipping.[/bold red]")
                files_skipped.append((file_path_item, f"Unexpected: {str(e)}"))

    if files_skipped:
        console.print("\n[bold yellow]Summary of skipped files:[/bold yellow]")
        for f_path, reason in files_skipped:
            console.print(f"  - {f_path}: {reason}")

    if not all_chunk_texts_for_embedding:
        console.print("\n[yellow]No text chunks extracted from any files. Cannot perform search.[/yellow]")
        # The logic below will handle skipping search steps if chunk_embeddings is empty.
    
    # --- Embedding Generation ---
    query_embedding: np.ndarray
    chunk_embeddings: np.ndarray = np.array([])

    if all_chunk_texts_for_embedding:
        console.print(f"\n[bold]Step 3: Generating Embeddings for {len(all_chunk_texts_for_embedding)} total chunk(s)[/bold]")
    else:
        console.print("\n[bold]Step 3: Generating Embeddings[/bold]")

    console.print("  (This may take a moment on first run if the embedding model needs to be downloaded...)")

    try:
        console.print(f"  Embedding query: '[italic blue]{query_text}[/italic blue]'")
        query_embedding = generate_embeddings(texts=[query_text])
        console.print(f"    Query embedding shape: {query_embedding.shape}")

        if all_chunk_texts_for_embedding:
            console.print(f"  Embedding {len(all_chunk_texts_for_embedding)} text chunk(s)...")
            chunk_embeddings = generate_embeddings(texts=all_chunk_texts_for_embedding)
            console.print(f"    Chunk embeddings shape: {chunk_embeddings.shape}")

            if chunk_embeddings.size > 0 and query_embedding.shape[1] != chunk_embeddings.shape[1]:
                console.print(
                    f"[bold red]Error: Query embedding dimension ({query_embedding.shape[1]}) "
                    f"does not match chunk embedding dimension ({chunk_embeddings.shape[1]}). "
                    f"This should not happen with the same model.[/bold red]"
                )
                raise typer.Exit(code=1)
        else:
            console.print("  No text chunks available from any file to embed.")
            # chunk_embeddings remains an empty np.array

    except RuntimeError as e:
        console.print(f"[bold red]Embedding Generation Failed:[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred during embedding: {e}[/bold red]")
        raise typer.Exit(code=1)

    # --- In-Memory Vector Search ---
    console.print("\n[bold]Step 4: Performing In-Memory Vector Search[/bold]")
    DEFAULT_K_RESULTS: int = 5
    search_matches: List[Tuple[int, float]] = []

    if chunk_embeddings.size == 0 or chunk_embeddings.shape[0] == 0:
        console.print("  No chunk embeddings generated. Skipping vector search.")
    else:
        try:
            console.print(f"  Creating in-memory index for {chunk_embeddings.shape[0]} chunk embedding(s)...")
            vector_index: usearch.index.Index = create_inmemory_index(embeddings=chunk_embeddings)
            console.print(
                f"    Index created with {len(vector_index)} item(s). "
                f"Metric: {vector_index.metric}, DType: {str(vector_index.dtype)}"
            )

            console.print(f"  Searching index for top {DEFAULT_K_RESULTS} similar chunk(s)...")
            search_matches = search_inmemory_index(
                index=vector_index,
                query_embedding=query_embedding,
                k=DEFAULT_K_RESULTS,
            )

            if not search_matches:
                console.print("  No matches found in the vector index for the query.")
            # Raw match display removed, will be handled in Step 5

        except ValueError as ve:
            console.print(f"[bold red]Error during vector search operation: {ve}[/bold red]")
            raise typer.Exit(code=1)
        except Exception as e:
            console.print(f"[bold red]An unexpected error occurred during vector search: {e}[/bold red]")
            raise typer.Exit(code=1)

    # --- Displaying Results ---
    console.print("\n[bold]Step 5: Displaying Results[/bold]")

    if not all_chunk_texts_for_embedding: # Or check aggregated_chunk_metadata
        console.print("  No text chunks were processed, so no search results to display.")
    elif not search_matches:
        console.print("  No relevant chunks found for your query in the processed file(s).")
    else:
        console.print(f"\n[bold cyan]Search Results (Top {len(search_matches)}):[/bold cyan]")
        for global_chunk_idx, similarity_score in search_matches:
            if 0 <= global_chunk_idx < len(aggregated_chunk_metadata):
                matched_chunk_info = aggregated_chunk_metadata[global_chunk_idx]
                actual_chunk_text = matched_chunk_info['text']
                source_file_path = matched_chunk_info['source_path']
                # original_chunk_idx_in_file = matched_chunk_info['original_chunk_index_in_file'] # Available if needed

                output_string = format_show_basic(
                    file_path=source_file_path,
                    chunk_text=actual_chunk_text,
                    score=similarity_score
                )
                console.print("---") # Visual separator for multiple results
                console.print(output_string)
            else:
                console.print(
                    f"[bold red]Critical Error: Search result index {global_chunk_idx} "
                    f"is out of bounds for aggregated_chunk_metadata (length {len(aggregated_chunk_metadata)}).[/bold red]"
                )


if __name__ == "__main__":
    app()