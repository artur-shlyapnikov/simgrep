import warnings
from pathlib import Path
from typing import (
    List,
    Tuple,
    Any,
    Dict,
    Optional,
)
from enum import Enum

import numpy as np
import typer
from rich.console import Console
import usearch.index
import duckdb # Added duckdb

# Assuming simgrep is installed or path is correctly set for sibling imports
try:
    from .models import ChunkData
    from .processor import (
        extract_text_from_file,
        generate_embeddings,
        load_tokenizer,
        chunk_text_by_tokens,
        ProcessedChunkInfo,
    )
    from .vector_store import create_inmemory_index, search_inmemory_index
    from .formatter import format_show_basic, format_paths
    from .metadata_db import (
        create_inmemory_db_connection,
        setup_ephemeral_tables,
        batch_insert_files,
        batch_insert_chunks,
        retrieve_chunk_for_display
    )
except ImportError:
    # Fallback for running main.py directly during development
    if __name__ == "__main__":
        import os
        import sys

        sys.path.insert(
            0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        )
        from simgrep.models import ChunkData
        from simgrep.processor import (
            extract_text_from_file,
            generate_embeddings,
            load_tokenizer,
            chunk_text_by_tokens,
            ProcessedChunkInfo,
        )
        from simgrep.vector_store import create_inmemory_index, search_inmemory_index
        from simgrep.formatter import format_show_basic, format_paths
        from simgrep.metadata_db import (
            create_inmemory_db_connection,
            setup_ephemeral_tables,
            batch_insert_files,
            batch_insert_chunks,
            retrieve_chunk_for_display
        )
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


class OutputMode(str, Enum):
    show = "show"
    paths = "paths"
    # json = "json" # Future modes
    # rag = "rag"   # Future modes


EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
CHUNK_SIZE_TOKENS: int = 128
OVERLAP_TOKENS: int = 20
DEFAULT_K_RESULTS: int = 5


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
        dir_okay=True,
        readable=True,
        resolve_path=True, # Ensures path_to_search is absolute and symlinks resolved
        help="The path to the text file or directory to search within.",
    ),
    output: OutputMode = typer.Option(
        OutputMode.show, # Default output mode
        "--output",
        "-o",
        help="Output mode. 'paths' mode lists unique, sorted file paths containing matches.",
        case_sensitive=False,
    ),
    relative_paths: bool = typer.Option(
        False,
        "--relative-paths/--absolute-paths", # Provides --no-relative-paths as well
        help=(
            "Display relative file paths. "
            "Paths are relative to the initial search target directory, or its parent if the target is a file. "
            "Only used with '--output paths'."
        ),
    )
) -> None:
    """
    Searches for a query within a specified text file or files in a directory using token-based chunking
    and an in-memory DuckDB for metadata management.
    """
    console.print(
        f"Searching for: '[bold blue]{query_text}[/bold blue]' in path: '[green]{path_to_search}[/green]'"
    )

    db_conn: Optional[duckdb.DuckDBPyConnection] = None
    try:
        # --- Initialize In-Memory Database ---
        console.print("\n[bold]Setup: Initializing In-Memory Database[/bold]")
        db_conn = create_inmemory_db_connection()
        setup_ephemeral_tables(db_conn)
        console.print("  In-memory database and tables created.")

        # --- Load Tokenizer ---
        console.print(f"\n[bold]Setup: Loading Tokenizer[/bold]")
        console.print(f"  Loading tokenizer for model: '{EMBEDDING_MODEL_NAME}'...")
        try:
            tokenizer = load_tokenizer(EMBEDDING_MODEL_NAME)
            console.print(
                f"    Tokenizer loaded successfully: {tokenizer.__class__.__name__}"
            )
        except RuntimeError as e:
            console.print(
                f"[bold red]Fatal Error: Could not load tokenizer.[/bold red]\n  Details: {e}"
            )
            raise typer.Exit(code=1)

        # --- File Discovery ---
        files_to_process: List[Path] = []
        files_skipped: List[Tuple[Path, str]] = []

        if path_to_search.is_file():
            files_to_process.append(path_to_search)
            console.print(f"Processing single file: [green]{path_to_search}[/green]")
        elif path_to_search.is_dir():
            console.print(
                f"Scanning directory: [green]{path_to_search}[/green] for processable files..."
            )
            discovered_files_generator = path_to_search.rglob("*.txt")
            files_to_process = list(discovered_files_generator)
            if not files_to_process:
                console.print(
                    f"[yellow]No '.txt' files found in directory: {path_to_search}[/yellow]"
                )
            else:
                console.print(f"Found {len(files_to_process)} '.txt' file(s) to process.")

        if not files_to_process:
            console.print("No files selected for processing. Exiting.")
            raise typer.Exit()

        # --- Text Extraction and Token-based Chunking ---
        all_chunkdata_objects: List[ChunkData] = []
        global_usearch_label_counter: int = 0

        console.print(
            "\n[bold]Step 1 & 2: Processing files, extracting and chunking text (token-based)[/bold]"
        )
        for file_idx, file_path_item in enumerate(files_to_process):
            console.print(
                f"  ({file_idx + 1}/{len(files_to_process)}) Processing: [dim]{file_path_item}[/dim]"
            )
            try:
                extracted_content = extract_text_from_file(file_path_item)
                if not extracted_content.strip():
                    console.print(
                        f"    [yellow]Skipped: File '{file_path_item}' is empty or contains only whitespace.[/yellow]"
                    )
                    files_skipped.append((file_path_item, "Empty or whitespace-only"))
                    continue

                intermediate_chunks_info: List[ProcessedChunkInfo] = chunk_text_by_tokens(
                    full_text=extracted_content,
                    tokenizer=tokenizer,
                    chunk_size_tokens=CHUNK_SIZE_TOKENS,
                    overlap_tokens=OVERLAP_TOKENS,
                )

                if intermediate_chunks_info:
                    for partial_chunk in intermediate_chunks_info:
                        chunk_data_item = ChunkData(
                            text=partial_chunk["text"],
                            source_file_path=file_path_item,
                            source_file_id=file_idx,
                            usearch_label=global_usearch_label_counter,
                            start_char_offset=partial_chunk["start_char_offset"],
                            end_char_offset=partial_chunk["end_char_offset"],
                            token_count=partial_chunk["token_count"],
                        )
                        all_chunkdata_objects.append(chunk_data_item)
                        global_usearch_label_counter += 1
                    console.print(
                        f"    Extracted {len(intermediate_chunks_info)} token-based chunk(s)."
                    )
                else:
                    console.print(
                        f"    [yellow]No token-based chunks generated for '{file_path_item}' (text might be too short or empty for current parameters).[/yellow]"
                    )
                    files_skipped.append(
                        (file_path_item, "No token-based chunks generated")
                    )

            except FileNotFoundError:
                console.print(
                    f"    [bold red]Error: File not found during processing loop: {file_path_item}. Skipping.[/bold red]"
                )
                files_skipped.append((file_path_item, "File not found during processing"))
            except RuntimeError as e:
                console.print(
                    f"    [bold red]Error processing or chunking file '{file_path_item}': {e}. Skipping.[/bold red]"
                )
                files_skipped.append((file_path_item, str(e)))
            except ValueError as ve:
                console.print(
                    f"    [bold red]Error with chunking parameters for file '{file_path_item}': {ve}. Skipping.[/bold red]"
                )
                files_skipped.append((file_path_item, str(ve)))
            except Exception as e:
                console.print(
                    f"    [bold red]Unexpected error processing file '{file_path_item}': {e}. Skipping.[/bold red]"
                )
                files_skipped.append((file_path_item, f"Unexpected: {str(e)}"))

        if files_skipped:
            console.print("\n[bold yellow]Summary of skipped files:[/bold yellow]")
            for f_path, reason in files_skipped:
                console.print(f"  - {f_path}: {reason}")

        if not all_chunkdata_objects:
            console.print(
                "\n[yellow]No text chunks extracted from any files. Cannot perform search.[/yellow]"
            )
            raise typer.Exit()

        # --- Populate In-Memory Database ---
        console.print("\n[bold]Setup: Populating In-Memory Database[/bold]")
        unique_files_metadata_dict: Dict[int, Path] = {}
        for cd_item in all_chunkdata_objects:
            if cd_item.source_file_id not in unique_files_metadata_dict:
                unique_files_metadata_dict[cd_item.source_file_id] = cd_item.source_file_path
        
        processed_files_metadata_for_db: List[Tuple[int, Path]] = [
            (fid, fpath) for fid, fpath in unique_files_metadata_dict.items()
        ]

        if processed_files_metadata_for_db:
            batch_insert_files(db_conn, processed_files_metadata_for_db)
            console.print(f"  Inserted metadata for {len(processed_files_metadata_for_db)} file(s) into DB.")
        
        batch_insert_chunks(db_conn, all_chunkdata_objects)
        console.print(f"  Inserted {len(all_chunkdata_objects)} chunk(s) into DB.")


        # --- Embedding Generation ---
        query_embedding: np.ndarray
        chunk_embeddings: np.ndarray

        console.print(
            f"\n[bold]Step 3: Generating Embeddings for {len(all_chunkdata_objects)} total chunk(s)[/bold]"
        )
        console.print(
            "  (This may take a moment on first run if the embedding model needs to be downloaded...)"
        )

        try:
            console.print(
                f"  Embedding query: '[italic blue]{query_text}[/italic blue]' using model '{EMBEDDING_MODEL_NAME}'"
            )
            query_embedding = generate_embeddings(
                texts=[query_text], model_name=EMBEDDING_MODEL_NAME
            )
            console.print(f"    Query embedding shape: {query_embedding.shape}")

            chunk_texts_for_embedding: List[str] = [cd.text for cd in all_chunkdata_objects]
            console.print(
                f"  Embedding {len(chunk_texts_for_embedding)} text chunk(s) using model '{EMBEDDING_MODEL_NAME}'..."
            )
            chunk_embeddings = generate_embeddings(
                texts=chunk_texts_for_embedding, model_name=EMBEDDING_MODEL_NAME
            )
            console.print(f"    Chunk embeddings shape: {chunk_embeddings.shape}")

            if (
                chunk_embeddings.size > 0
                and query_embedding.shape[1] != chunk_embeddings.shape[1]
            ):
                console.print(
                    f"[bold red]Error: Query embedding dimension ({query_embedding.shape[1]}) "
                    f"does not match chunk embedding dimension ({chunk_embeddings.shape[1]}). "
                    f"This should not happen with the same model.[/bold red]"
                )
                raise typer.Exit(code=1)

        except RuntimeError as e:
            console.print(f"[bold red]Embedding Generation Failed:[/bold red] {e}")
            raise typer.Exit(code=1)
        except Exception as e:
            console.print(
                f"[bold red]An unexpected error occurred during embedding: {e}[/bold red]"
            )
            raise typer.Exit(code=1)

        # --- In-Memory Vector Search ---
        console.print("\n[bold]Step 4: Performing In-Memory Vector Search[/bold]")
        search_matches: List[Tuple[int, float]] = []

        if chunk_embeddings.size == 0 or chunk_embeddings.shape[0] == 0:
            console.print("  No chunk embeddings available. Skipping vector search.")
        else:
            try:
                console.print(
                    f"  Creating in-memory index for {chunk_embeddings.shape[0]} chunk embedding(s)..."
                )
                usearch_labels_np = np.array([cd.usearch_label for cd in all_chunkdata_objects], dtype=np.int64)
                vector_index: usearch.index.Index = create_inmemory_index(
                    embeddings=chunk_embeddings, labels_for_usearch=usearch_labels_np
                )
                console.print(
                    f"    Index created with {len(vector_index)} item(s). "
                    f"Metric: {vector_index.metric}, DType: {str(vector_index.dtype)}"
                )

                console.print(
                    f"  Searching index for top {DEFAULT_K_RESULTS} similar chunk(s)..."
                )
                search_matches = search_inmemory_index(
                    index=vector_index,
                    query_embedding=query_embedding,
                    k=DEFAULT_K_RESULTS,
                )

                if not search_matches:
                    console.print("  No matches found in the vector index for the query.")

            except ValueError as ve:
                console.print(
                    f"[bold red]Error during vector search operation: {ve}[/bold red]"
                )
                raise typer.Exit(code=1)
            except Exception as e:
                console.print(
                    f"[bold red]An unexpected error occurred during vector search: {e}[/bold red]"
                )
                raise typer.Exit(code=1)

        console.print("\n[bold]Step 5: Displaying Results[/bold]")

        if relative_paths and output != OutputMode.paths:
            console.print(
                "[yellow]Warning: --relative-paths is only effective with --output paths. "
                "Paths will be displayed according to the selected output mode.[/yellow]"
            )

        if not search_matches:
            if output == OutputMode.paths:
                console.print(format_paths(file_paths=[], use_relative=False, base_path=None)) # Handles "No matching files found."
            else: # OutputMode.show or other future modes that might show "no results"
                console.print(
                    "  No relevant chunks found for your query in the processed file(s)."
                )
        else:
            if output == OutputMode.paths:
                paths_from_matches: List[Path] = []
                for matched_chunk_id, _similarity_score in search_matches:
                    retrieved_details = retrieve_chunk_for_display(db_conn, matched_chunk_id)
                    if retrieved_details:
                        _text_content, retrieved_path, _start_char, _end_char = retrieved_details
                        paths_from_matches.append(retrieved_path)
                    else:
                        console.print(f"[yellow]Warning: Could not retrieve path for chunk_id {matched_chunk_id}.[/yellow]")
                
                current_base_path_for_relativity: Optional[Path] = None
                actual_use_relative = relative_paths

                if actual_use_relative:
                    if path_to_search.is_dir():
                        current_base_path_for_relativity = path_to_search
                    else: # path_to_search is a file
                        current_base_path_for_relativity = path_to_search.parent
                
                output_string = format_paths(
                    file_paths=paths_from_matches,
                    use_relative=actual_use_relative,
                    base_path=current_base_path_for_relativity
                )
                if output_string:
                    console.print(output_string)

            elif output == OutputMode.show:
                console.print(
                    f"\n[bold cyan]Search Results (Top {len(search_matches)}):[/bold cyan]"
                )
                for matched_chunk_id, similarity_score in search_matches:
                    retrieved_details = retrieve_chunk_for_display(db_conn, matched_chunk_id)
                    
                    if retrieved_details:
                        retrieved_text, retrieved_path, _start_offset, _end_offset = retrieved_details
                        output_string = format_show_basic(
                            file_path=retrieved_path,
                            chunk_text=retrieved_text,
                            score=similarity_score,
                        )
                        console.print("---")  # Visual separator
                        console.print(output_string)
                    else:
                        console.print(
                            f"[bold yellow]Warning: Could not retrieve details for chunk_id {matched_chunk_id} from DB.[/bold yellow]"
                        )

    finally:
        if db_conn:
            console.print("\n[bold]Cleanup: Closing In-Memory Database[/bold]")
            db_conn.close()
            console.print("  Database connection closed.")


if __name__ == "__main__":
    app()