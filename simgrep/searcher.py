import pathlib
from typing import Dict, List, Optional, Tuple

import usearch.index
from rich.console import Console

from .config import DEFAULT_K_RESULTS, SimgrepConfig
from .exceptions import MetadataDBError, VectorStoreError
from .formatter import format_paths, format_show_basic
from .metadata_store import MetadataStore
from .models import ChunkData, OutputMode, SearchResult
from .processor import (
    ProcessedChunkInfo,
    chunk_text_by_tokens,
    extract_text_from_file,
    generate_embeddings,
    load_embedding_model,
    load_tokenizer,
)
from .utils import gather_files_to_process
from .vector_store import (
    create_inmemory_index,
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
        search_matches: List[SearchResult] = search_inmemory_index(
            index=vector_index, query_embedding=query_embedding, k=k_results
        )
    except (VectorStoreError, ValueError) as e:
        console.print(f"[bold red]Error during vector search:[/bold red]\n  {e}")
        raise  # re-raise

    # Filter by min_score
    filtered_matches = [m for m in search_matches if m.score >= min_score]

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
        for result in filtered_matches:
            matched_usearch_label = result.label
            similarity_score = result.score
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
        for result in filtered_matches:
            matched_usearch_label = result.label
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


def perform_ephemeral_search(
    query_text: str,
    path_to_search: pathlib.Path,
    console: Console,
    global_config: SimgrepConfig,
    *,
    patterns: Optional[List[str]] = None,
    output_mode: OutputMode = OutputMode.show,
    k_results: int = DEFAULT_K_RESULTS,
    relative_paths: bool = False,
) -> None:
    """Execute an ephemeral search against files on disk."""
    console.print(
        f"Performing ephemeral search for: '[bold blue]{query_text}[/bold blue]' in path: '[green]{path_to_search}[/green]'"
    )

    store: Optional[MetadataStore] = None
    try:
        console.print("\n[bold]Setup: Initializing In-Memory Database[/bold]")
        store = MetadataStore()
        console.print("  In-memory database and tables created.")

        console.print("\n[bold]Setup: Loading Tokenizer[/bold]")
        console.print(
            f"  Loading tokenizer for model: '{global_config.default_embedding_model_name}'..."
        )
        try:
            tokenizer = load_tokenizer(global_config.default_embedding_model_name)
            console.print(f"    Tokenizer loaded successfully: {tokenizer.__class__.__name__}")
        except RuntimeError as e:
            console.print(f"[bold red]Fatal Error: Could not load tokenizer.[/bold red]\n  Details: {e}")
            raise typer.Exit(code=1)

        files_to_process: List[pathlib.Path] = []
        files_skipped: List[Tuple[pathlib.Path, str]] = []

        search_patterns = list(patterns) if patterns else ["*.txt"]
        files_to_process = gather_files_to_process(path_to_search, search_patterns)

        if path_to_search.is_file():
            console.print(f"Processing single file: [green]{path_to_search}[/green]")
        else:
            console.print(
                f"Scanning directory: [green]{path_to_search}[/green] for files matching: {search_patterns}..."
            )
            if not files_to_process:
                console.print(
                    f"[yellow]No files found in directory {path_to_search} with patterns {search_patterns}[/yellow]"
                )
            else:
                console.print(f"Found {len(files_to_process)} file(s) to process.")

        if not files_to_process:
            console.print("No files selected for processing. Exiting.")
            raise typer.Exit()

        all_chunkdata_objects: List[ChunkData] = []
        global_usearch_label_counter = 0

        console.print("\n[bold]Step 1 & 2: Processing files, extracting and chunking text (token-based)[/bold]")
        progress_columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ]
        with Progress(*progress_columns, console=console, transient=False) as progress:
            processing_task = progress.add_task("Processing files...", total=len(files_to_process))
            for file_idx, file_path_item in enumerate(files_to_process):
                progress.update(processing_task, description=f"Processing: {file_path_item.name}")
                try:
                    extracted_content = extract_text_from_file(file_path_item)
                    if not extracted_content.strip():
                        console.print(
                            f"    [yellow]Skipped: File '{file_path_item}' is empty or contains only whitespace.[/yellow]"
                        )
                        files_skipped.append((file_path_item, "Empty or whitespace-only"))
                        progress.advance(processing_task)
                        continue

                    intermediate_chunks_info: List[ProcessedChunkInfo] = chunk_text_by_tokens(
                        full_text=extracted_content,
                        tokenizer=tokenizer,
                        chunk_size_tokens=global_config.default_chunk_size_tokens,
                        overlap_tokens=global_config.default_chunk_overlap_tokens,
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
                        console.print(f"    Extracted {len(intermediate_chunks_info)} token-based chunk(s).")
                    else:
                        console.print(
                            f"    [yellow]No token-based chunks generated for '{file_path_item}' (text might be too short or empty for current parameters).[/yellow]"
                        )
                        files_skipped.append((file_path_item, "No token-based chunks generated"))

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
                except Exception as e:  # pragma: no cover - unexpected errors
                    console.print(
                        f"    [bold red]Unexpected error processing file '{file_path_item}': {e}. Skipping.[/bold red]"
                    )
                    files_skipped.append((file_path_item, f"Unexpected: {str(e)}"))
                finally:
                    progress.advance(processing_task)

        if files_skipped:
            console.print("\n[bold yellow]Summary of skipped files:[/bold yellow]")
            for f_path, reason in files_skipped:
                console.print(f"  - {f_path}: {reason}")

        if not all_chunkdata_objects:
            console.print("\n[yellow]No text chunks extracted from any files. Cannot perform search.[/yellow]")
            raise typer.Exit()

        console.print("\n[bold]Setup: Populating In-Memory Database[/bold]")
        unique_files_metadata_dict: Dict[int, pathlib.Path] = {}
        for cd_item in all_chunkdata_objects:
            if cd_item.source_file_id not in unique_files_metadata_dict:
                unique_files_metadata_dict[cd_item.source_file_id] = cd_item.source_file_path

        processed_files_metadata_for_db: List[Tuple[int, pathlib.Path]] = [
            (fid, fpath) for fid, fpath in unique_files_metadata_dict.items()
        ]

        if processed_files_metadata_for_db:
            assert store is not None
            store.batch_insert_files(processed_files_metadata_for_db)
            console.print(f"  Inserted metadata for {len(processed_files_metadata_for_db)} file(s) into DB.")

        assert store is not None
        store.batch_insert_chunks(all_chunkdata_objects)
        console.print(f"  Inserted {len(all_chunkdata_objects)} chunk(s) into DB.")

        console.print("\n[bold]Step 3: Generating Embeddings[/bold]")
        console.print(
            "  (This may take a moment on first run if the embedding model needs to be downloaded...)"
        )
        try:
            embedding_model_instance = load_embedding_model(global_config.default_embedding_model_name)
            console.print("    Embedding model loaded.")

            console.print(f"  Embedding query: '[italic blue]{query_text}[/italic blue]'...")
            query_embedding = generate_embeddings(texts=[query_text], model=embedding_model_instance)
            console.print(f"    Query embedding shape: {query_embedding.shape}")

            chunk_texts_for_embedding: List[str] = [cd.text for cd in all_chunkdata_objects]
            console.print(f"  Embedding {len(chunk_texts_for_embedding)} text chunk(s)...")
            chunk_embeddings = generate_embeddings(texts=chunk_texts_for_embedding, model=embedding_model_instance)
            console.print(f"    Chunk embeddings shape: {chunk_embeddings.shape}")

            if chunk_embeddings.size > 0 and query_embedding.shape[1] != chunk_embeddings.shape[1]:
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
            console.print(f"[bold red]An unexpected error occurred during embedding: {e}[/bold red]")
            raise typer.Exit(code=1)

        console.print("\n[bold]Step 4: Performing In-Memory Vector Search[/bold]")
        search_matches: List[SearchResult] = []

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
                    f"    Index created with {len(vector_index)} item(s). Metric: {vector_index.metric}, DType: {str(vector_index.dtype)}"
                )

                console.print(f"  Searching index for top {k_results} similar chunk(s)...")
                search_matches = search_inmemory_index(
                    index=vector_index,
                    query_embedding=query_embedding,
                    k=k_results,
                )

                if not search_matches:
                    console.print("  No matches found in the vector index for the query.")

            except ValueError as ve:
                console.print(f"[bold red]Error during vector search operation: {ve}[/bold red]")
                raise typer.Exit(code=1)
            except Exception as e:
                console.print(f"[bold red]An unexpected error occurred during vector search: {e}[/bold red]")
                raise typer.Exit(code=1)

        console.print("\n[bold]Step 5: Displaying Results[/bold]")

        if relative_paths and output_mode != OutputMode.paths:
            console.print(
                "[yellow]Warning: --relative-paths is only effective with --output paths. Paths will be displayed according to the selected output mode.[/yellow]"
            )

        if not search_matches:
            if output_mode == OutputMode.paths:
                console.print(
                    format_paths(
                        file_paths=[],
                        use_relative=False,
                        base_path=None,
                        console=console,
                    )
                )
            else:
                console.print("  No relevant chunks found for your query in the processed file(s).")
        else:
            if output_mode == OutputMode.paths:
                paths_from_matches: List[pathlib.Path] = []
                for result in search_matches:
                    matched_chunk_id = result.label
                    assert store is not None
                    retrieved_details = store.retrieve_chunk_for_display(matched_chunk_id)

                    if retrieved_details:
                        _text_content, retrieved_path, _start_char, _end_char = retrieved_details
                        paths_from_matches.append(retrieved_path)
                    else:
                        console.print(
                            f"[yellow]Warning: Could not retrieve path for chunk_id {matched_chunk_id}.[/yellow]"
                        )

                current_base_path_for_relativity: Optional[pathlib.Path] = None
                actual_use_relative = relative_paths

                if actual_use_relative:
                    if path_to_search.is_dir():
                        current_base_path_for_relativity = path_to_search
                    else:
                        current_base_path_for_relativity = path_to_search.parent

                output_string = format_paths(
                    file_paths=paths_from_matches,
                    use_relative=actual_use_relative,
                    base_path=current_base_path_for_relativity,
                    console=console,
                )
                if output_string:
                    console.print(output_string)

            elif output_mode == OutputMode.show:
                console.print(f"\n[bold cyan]Search Results (Top {len(search_matches)}):[/bold cyan]")
                for result in search_matches:
                    matched_chunk_id = result.label
                    similarity_score = result.score
                    assert store is not None
                    retrieved_details = store.retrieve_chunk_for_display(matched_chunk_id)

                    if retrieved_details:
                        retrieved_text, retrieved_path, _start_offset, _end_offset = retrieved_details
                        output_string = format_show_basic(
                            file_path=retrieved_path,
                            chunk_text=retrieved_text,
                            score=similarity_score,
                        )
                        console.print("---")
                        console.print(output_string)
                    else:
                        console.print(
                            f"[bold yellow]Warning: Could not retrieve details for chunk_id {matched_chunk_id} from DB.[/bold yellow]"
                        )

    finally:
        if store:
            console.print("\n[bold]Cleanup: Closing In-Memory Database[/bold]")
            store.close()
            console.print("  Database connection closed.")


class SearchEngine:
    """Encapsulates search operations."""

    def __init__(self, console: Console) -> None:
        self.console = console

    def search_persistent(
        self,
        query_text: str,
        *,
        metadata_store: MetadataStore,
        vector_index: usearch.index.Index,
        global_config: SimgrepConfig,
        output_mode: OutputMode,
        k_results: int = DEFAULT_K_RESULTS,
        display_relative_paths: bool = False,
        base_path_for_relativity: Optional[pathlib.Path] = None,
        min_score: float = 0.1,
    ) -> None:
        perform_persistent_search(
            query_text=query_text,
            console=self.console,
            metadata_store=metadata_store,
            vector_index=vector_index,
            global_config=global_config,
            output_mode=output_mode,
            k_results=k_results,
            display_relative_paths=display_relative_paths,
            base_path_for_relativity=base_path_for_relativity,
            min_score=min_score,
        )

    def search_ephemeral(
        self,
        query_text: str,
        path_to_search: pathlib.Path,
        *,
        global_config: SimgrepConfig,
        patterns: Optional[List[str]] = None,
        output_mode: OutputMode = OutputMode.show,
        k_results: int = DEFAULT_K_RESULTS,
        relative_paths: bool = False,
    ) -> None:
        perform_ephemeral_search(
            query_text=query_text,
            path_to_search=path_to_search,
            console=self.console,
            global_config=global_config,
            patterns=patterns,
            output_mode=output_mode,
            k_results=k_results,
            relative_paths=relative_paths,
        )
