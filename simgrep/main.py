import sys  # For printing to stderr in case of config error
import warnings
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np
import typer
import usearch.index
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)

# Assuming simgrep is installed or path is correctly set for sibling imports
try:
    from .config import (
        DEFAULT_K_RESULTS,
        SimgrepConfigError,
        add_project_to_config,
        load_or_create_global_config,
    )
    from .formatter import format_paths, format_show_basic
    from .indexer import Indexer, IndexerConfig, IndexerError
    from .metadata_db import (
        MetadataDBError,
        connect_global_db,
        get_all_projects,
        get_index_counts,
        get_project_by_name,
        insert_project,
    )
    from .metadata_store import MetadataStore
    from .models import ChunkData, OutputMode, SearchResult, SimgrepConfig  # OutputMode moved here
    from .processor import (
        ProcessedChunkInfo,
        chunk_text_by_tokens,
        extract_text_from_file,
        generate_embeddings,
        load_tokenizer,
    )
    from .searcher import SearchEngine
    from .utils import gather_files_to_process
    from .vector_store import (
        VectorStoreError,  # Added VectorStoreError
        create_inmemory_index,
        load_persistent_index,
        search_inmemory_index,
    )
except ImportError:
    # Fallback for running main.py directly during development
    if __name__ == "__main__":
        import os
        import sys

        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from simgrep.config import (
            DEFAULT_K_RESULTS,
            SimgrepConfigError,
            add_project_to_config,
            load_or_create_global_config,
        )
        from simgrep.formatter import format_paths, format_show_basic
        from simgrep.indexer import Indexer, IndexerConfig, IndexerError
        from simgrep.metadata_db import (
            MetadataDBError,
            connect_global_db,
            get_all_projects,
            get_index_counts,
            get_project_by_name,
            insert_project,
        )
        from simgrep.metadata_store import MetadataStore
        from simgrep.models import ChunkData, OutputMode, SearchResult, SimgrepConfig
        from simgrep.processor import (
            ProcessedChunkInfo,
            chunk_text_by_tokens,
            extract_text_from_file,
            generate_embeddings,
            load_tokenizer,
        )
        from simgrep.searcher import SearchEngine
        from simgrep.utils import gather_files_to_process
        from simgrep.vector_store import (  # Added
            VectorStoreError,
            create_inmemory_index,
            load_persistent_index,
            search_inmemory_index,
        )
    else:
        raise

# --- Constants for Ephemeral Search (Deliverable 1 & 2 focus) ---
# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Now from SimgrepConfig
# CHUNK_SIZE_TOKENS = 128 # Now from SimgrepConfig
# OVERLAP_TOKENS = 20     # Now from SimgrepConfig
# DEFAULT_K_RESULTS = 5  # Moved to config.py

# Suppress specific warnings from sentence_transformers if needed, or handle them appropriately
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="sentence_transformers.SentenceTransformer",
)


# OutputMode moved to models.py

# Initialize Typer app and Rich console
app = typer.Typer(
    name="simgrep",
    help="A command-line tool for semantic search in local files.",
    add_completion=False,
    no_args_is_help=True,
)
project_app = typer.Typer(help="Manage simgrep projects.")
app.add_typer(project_app, name="project")
console = Console()


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(None, "--version", "-v", help="Show simgrep version and exit.", is_eager=True),
) -> None:
    """
    simgrep CLI application.
    Initializes global configuration, ensuring necessary directories are created.
    """
    if version:  # If --version is passed, version_callback handles exit.
        return

    try:
        # Load configuration. This step ensures the default_project_data_dir is created.
        # The '_config' variable itself is not used further by main_callback or
        # the ephemeral search command in *this deliverable*.
        # It will be crucial for persistent indexing commands later.
        _config = load_or_create_global_config()
    except SimgrepConfigError:
        # Error already printed by load_or_create_global_config
        # console.print(f"[bold red]Fatal Configuration Error:[/bold red]\n{e}") # Redundant if config prints
        raise typer.Exit(code=1)
    except Exception as e:  # Catch any other unexpected errors during config load
        console.print(f"[bold red]An unexpected error occurred during Simgrep initialization:[/bold red]\n{e}")
        raise typer.Exit(code=1)


@app.command()
def search(
    query_text: str = typer.Argument(..., help="The text or concept to search for."),
    path_to_search: Optional[Path] = typer.Argument(
        None,  # Default to None, making it optional
        exists=True,  # This will only be checked if path_to_search is not None
        file_okay=True,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="The path to a text file or directory for ephemeral search. If omitted, searches the default persistent index.",
    ),
    patterns: Optional[List[str]] = typer.Option(
        None,
        "--pattern",
        "-p",
        help="Glob pattern(s) for files when searching directories. Can be used multiple times. Defaults to '*.txt'.",
    ),
    output: OutputMode = typer.Option(
        OutputMode.show,  # Default output mode
        "--output",
        "-o",
        help="Output mode. 'paths' mode lists unique, sorted file paths containing matches.",
        case_sensitive=False,
    ),
    top: int = typer.Option(
        DEFAULT_K_RESULTS,
        "--top",
        "--k",
        help="Number of top results to return from the vector search.",
    ),
    relative_paths: bool = typer.Option(
        False,
        "--relative-paths/--absolute-paths",  # Provides --no-relative-paths as well
        help=(
            "Display relative file paths. "
            "Paths are relative to the initial search target directory, or its parent if the target is a file. "
            "Only used with '--output paths'."
        ),
    ),
    project: str = typer.Option(
        "default",
        "--project",
        help="Project name to use for persistent search.",
    ),
) -> None:
    """Searches for a query within files.

    If ``path_to_search`` is a directory, files are discovered using the given
    ``patterns`` (default ``['*.txt']``). If ``path_to_search`` is omitted the
    persistent index is searched instead.
    """
    global_simgrep_config: SimgrepConfig = load_or_create_global_config()  # Load config early
    search_engine = SearchEngine(console)
    project_cfg = global_simgrep_config.projects.get(project)
    if project_cfg is None:
        console.print(f"[bold red]Error: Project '{project}' not found.[/bold red]")
        raise typer.Exit(code=1)

    if path_to_search is None:
        # --- Persistent Search ---
        console.print(
            f"Searching for: '[bold blue]{query_text}[/bold blue]' in project '[magenta]{project}[/magenta]'"
        )
        project_db_file = project_cfg.db_path
        project_usearch_file = project_cfg.usearch_index_path

        if not project_db_file.exists() or not project_usearch_file.exists():
            console.print(
                f"[bold yellow]Warning: Persistent index for project '{project}' not found at '{project_cfg.db_path.parent}'.[/bold yellow]\n"
                f"Please run 'simgrep index <path> --project {project}' first to create an index."
            )
            if output == OutputMode.paths:
                console.print("No matching files found.")
                raise typer.Exit()
            raise typer.Exit(code=1)

        persistent_store: Optional[MetadataStore] = None
        persistent_vector_index: Optional[usearch.index.Index] = None
        try:
            console.print("  Loading persistent database...")
            persistent_store = MetadataStore(persistent=True, db_path=project_db_file)
            console.print("  Loading persistent vector index...")
            persistent_vector_index = load_persistent_index(project_usearch_file)

            if persistent_vector_index is None:
                console.print(
                    f"[bold red]Error: Vector index file loaded as None from {project_usearch_file}. "
                    "Index might be corrupted or empty after a failed write.[/bold red]"
                )
                raise typer.Exit(code=1)
            if len(persistent_vector_index) == 0:
                console.print("[yellow]Warning: The persistent vector index is empty. No search can be performed.[/yellow]")
                # Output "no results" based on mode
                if output == OutputMode.paths:
                    console.print(
                        format_paths(
                            file_paths=[],
                            use_relative=False,
                            base_path=None,
                            console=console,
                        )
                    )
                else:  # show
                    console.print("  No relevant chunks found in the persistent index.")
                raise typer.Exit()

            search_engine.search_persistent(
                query_text=query_text,
                metadata_store=persistent_store,
                vector_index=persistent_vector_index,
                global_config=global_simgrep_config,
                output_mode=output,
                k_results=top,
                display_relative_paths=relative_paths,
                base_path_for_relativity=Path.cwd() if relative_paths else None,
            )
        except (MetadataDBError, VectorStoreError, RuntimeError, ValueError) as e:
            console.print(f"[bold red]Error during persistent search: {e}[/bold red]")
            raise typer.Exit(code=1)
        finally:
            if persistent_store:
                persistent_store.close()
                console.print("  Persistent database connection closed.")
        return  # End of persistent search path

    # --- Ephemeral Search (path_to_search is provided) ---
    search_engine.search_ephemeral(
        query_text=query_text,
        path_to_search=path_to_search,
        global_config=global_simgrep_config,
        patterns=patterns,
        output_mode=output,
        k_results=top,
        relative_paths=relative_paths,
    )



@app.command()
def index(
    path_to_index: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        resolve_path=True,  # Ensures path is absolute and symlinks resolved
        help="The path to the file or directory to index.",
    ),
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        help="Wipe existing data and rebuild the project's index from scratch.",
    ),
    project: str = typer.Option(
        "default",
        "--project",
        help="Project name to index.",
    ),
) -> None:
    """
    Creates or updates a persistent index for the specified path within a project.
    If --rebuild is provided, existing data for that project will be deleted before indexing.
    """
    console.print(f"Starting indexing for path: [green]{path_to_index}[/green] into project '[magenta]{project}[/magenta]'")
    if rebuild:
        console.print(f"[bold yellow]Warning: This will wipe and rebuild the '{project}' project index.[/bold yellow]")
        if not typer.confirm(
            f"Are you sure you want to wipe and rebuild the '{project}' project index?",
            default=False,
        ):
            console.print("Aborted indexing.")
            raise typer.Abort()

    try:
        global_simgrep_config: SimgrepConfig = load_or_create_global_config()
        project_cfg = global_simgrep_config.projects.get(project)
        if project_cfg is None:
            console.print(f"[bold red]Error: Project '{project}' not found.[/bold red]")
            raise typer.Exit(code=1)

        project_db_file = project_cfg.db_path
        project_usearch_file = project_cfg.usearch_index_path

        # Prepare configuration for the Indexer
        indexer_config = IndexerConfig(
            project_name=project,
            db_path=project_db_file,
            usearch_index_path=project_usearch_file,
            embedding_model_name=global_simgrep_config.default_embedding_model_name,
            chunk_size_tokens=global_simgrep_config.default_chunk_size_tokens,
            chunk_overlap_tokens=global_simgrep_config.default_chunk_overlap_tokens,
            file_scan_patterns=["*.txt"],  # Initially hardcode to .txt, make configurable later
        )

        indexer_instance = Indexer(config=indexer_config, console=console)
        indexer_instance.index_path(target_path=path_to_index, wipe_existing=rebuild)

        console.print(f"[bold green]Successfully indexed '{path_to_index}' into project '{project}'.[/bold green]")

    except SimgrepConfigError as e:
        console.print(f"[bold red]Configuration Error:[/bold red]\n  {e}")
        raise typer.Exit(code=1)
    except IndexerError as e:  # Custom error from Indexer
        console.print(f"[bold red]Indexing Error:[/bold red]\n  {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred during indexing:[/bold red]\n  {e}")
        # For detailed debugging, consider logging the full traceback
        # import traceback
        # console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(code=1)


@app.command()
def status() -> None:
    try:
        cfg: SimgrepConfig = load_or_create_global_config()
    except SimgrepConfigError:
        raise typer.Exit(code=1)

    db_file = cfg.default_project_data_dir / "metadata.duckdb"
    if not db_file.exists():
        console.print(
            f"[bold yellow]Warning: Default persistent index not found at '{cfg.default_project_data_dir}'.[/bold yellow]\n"
            "Please run 'simgrep index <path>' first to create an index."
        )
        raise typer.Exit(code=1)

    store: Optional[MetadataStore] = None
    try:
        store = MetadataStore(persistent=True, db_path=db_file)
        files_count, chunks_count = get_index_counts(store.conn)
        console.print(f"Default Project: {files_count} files indexed, {chunks_count} chunks.")
    except MetadataDBError as e:
        console.print(f"[bold red]Error retrieving status: {e}[/bold red]")
        raise typer.Exit(code=1)
    finally:
        if store:
            store.close()


@project_app.command("create")
def project_create(name: str) -> None:
    """Create a new named project."""
    try:
        cfg = load_or_create_global_config()
    except SimgrepConfigError:
        raise typer.Exit(code=1)

    global_db_path = cfg.db_directory / "global_metadata.duckdb"
    conn = connect_global_db(global_db_path)
    try:
        if get_project_by_name(conn, name) is not None:
            console.print(f"[bold red]Project '{name}' already exists.[/bold red]")
            raise typer.Exit(code=1)

        proj_cfg = add_project_to_config(cfg, name)
        insert_project(
            conn,
            project_name=name,
            db_path=str(proj_cfg.db_path),
            usearch_index_path=str(proj_cfg.usearch_index_path),
            embedding_model_name=proj_cfg.embedding_model,
        )
        console.print(f"[green]Project '{name}' created.[/green]")
    except (MetadataDBError, SimgrepConfigError) as e:
        console.print(f"[bold red]Error creating project: {e}[/bold red]")
        raise typer.Exit(code=1)
    finally:
        conn.close()


@project_app.command("list")
def project_list() -> None:
    """List all configured projects."""
    try:
        cfg = load_or_create_global_config()
    except SimgrepConfigError:
        raise typer.Exit(code=1)

    global_db_path = cfg.db_directory / "global_metadata.duckdb"
    conn = connect_global_db(global_db_path)
    try:
        projects = get_all_projects(conn)
    except MetadataDBError as e:
        console.print(f"[bold red]Error listing projects: {e}[/bold red]")
        raise typer.Exit(code=1)
    finally:
        conn.close()

    for proj in projects:
        label = " (default)" if proj == "default" else ""
        console.print(f"{proj}{label}")


if __name__ == "__main__":
    app()
import sys  # For printing to stderr in case of config error
import warnings
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np
import typer
import usearch.index
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)

# Assuming simgrep is installed or path is correctly set for sibling imports
try:
    from .config import (
        DEFAULT_K_RESULTS,
        SimgrepConfigError,
        add_project_to_config,
        load_or_create_global_config,
    )
    from .formatter import format_paths, format_show_basic
    from .indexer import Indexer, IndexerConfig, IndexerError
    from .metadata_db import (
        MetadataDBError,
        connect_global_db,
        get_all_projects,
        get_index_counts,
        get_project_by_name,
        insert_project,
    )
    from .metadata_store import MetadataStore
    from .models import ChunkData, OutputMode, SearchResult, SimgrepConfig  # OutputMode moved here
    from .processor import (
        ProcessedChunkInfo,
        chunk_text_by_tokens,
        extract_text_from_file,
        generate_embeddings,
        load_tokenizer,
    )
    from .searcher import SearchEngine
    from .utils import gather_files_to_process
    from .vector_store import (
        VectorStoreError,  # Added VectorStoreError
        create_inmemory_index,
        load_persistent_index,
        search_inmemory_index,
    )
except ImportError:
    # Fallback for running main.py directly during development
    if __name__ == "__main__":
        import os
        import sys

        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from simgrep.config import (
            DEFAULT_K_RESULTS,
            SimgrepConfigError,
            add_project_to_config,
            load_or_create_global_config,
        )
        from simgrep.formatter import format_paths, format_show_basic
        from simgrep.indexer import Indexer, IndexerConfig, IndexerError
        from simgrep.metadata_db import (
            MetadataDBError,
            connect_global_db,
            get_all_projects,
            get_index_counts,
            get_project_by_name,
            insert_project,
        )
        from simgrep.metadata_store import MetadataStore
        from simgrep.models import ChunkData, OutputMode, SearchResult, SimgrepConfig
        from simgrep.processor import (
            ProcessedChunkInfo,
            chunk_text_by_tokens,
            extract_text_from_file,
            generate_embeddings,
            load_tokenizer,
        )
        from simgrep.searcher import SearchEngine
        from simgrep.utils import gather_files_to_process
        from simgrep.vector_store import (  # Added
            VectorStoreError,
            create_inmemory_index,
            load_persistent_index,
            search_inmemory_index,
        )
    else:
        raise

# --- Constants for Ephemeral Search (Deliverable 1 & 2 focus) ---
# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Now from SimgrepConfig
# CHUNK_SIZE_TOKENS = 128 # Now from SimgrepConfig
# OVERLAP_TOKENS = 20     # Now from SimgrepConfig
# DEFAULT_K_RESULTS = 5  # Moved to config.py

# Suppress specific warnings from sentence_transformers if needed, or handle them appropriately
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="sentence_transformers.SentenceTransformer",
)


# OutputMode moved to models.py

# Initialize Typer app and Rich console
app = typer.Typer(
    name="simgrep",
    help="A command-line tool for semantic search in local files.",
    add_completion=False,
    no_args_is_help=True,
)
project_app = typer.Typer(help="Manage simgrep projects.")
app.add_typer(project_app, name="project")
console = Console()


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(None, "--version", "-v", help="Show simgrep version and exit.", is_eager=True),
) -> None:
    """
    simgrep CLI application.
    Initializes global configuration, ensuring necessary directories are created.
    """
    if version:  # If --version is passed, version_callback handles exit.
        return

    try:
        # Load configuration. This step ensures the default_project_data_dir is created.
        # The '_config' variable itself is not used further by main_callback or
        # the ephemeral search command in *this deliverable*.
        # It will be crucial for persistent indexing commands later.
        _config = load_or_create_global_config()
    except SimgrepConfigError:
        # Error already printed by load_or_create_global_config
        # console.print(f"[bold red]Fatal Configuration Error:[/bold red]\n{e}") # Redundant if config prints
        raise typer.Exit(code=1)
    except Exception as e:  # Catch any other unexpected errors during config load
        console.print(f"[bold red]An unexpected error occurred during Simgrep initialization:[/bold red]\n{e}")
        raise typer.Exit(code=1)


@app.command()
def search(
    query_text: str = typer.Argument(..., help="The text or concept to search for."),
    path_to_search: Optional[Path] = typer.Argument(
        None,  # Default to None, making it optional
        exists=True,  # This will only be checked if path_to_search is not None
        file_okay=True,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="The path to a text file or directory for ephemeral search. If omitted, searches the default persistent index.",
    ),
    patterns: Optional[List[str]] = typer.Option(
        None,
        "--pattern",
        "-p",
        help="Glob pattern(s) for files when searching directories. Can be used multiple times. Defaults to '*.txt'.",
    ),
    output: OutputMode = typer.Option(
        OutputMode.show,  # Default output mode
        "--output",
        "-o",
        help="Output mode. 'paths' mode lists unique, sorted file paths containing matches.",
        case_sensitive=False,
    ),
    top: int = typer.Option(
        DEFAULT_K_RESULTS,
        "--top",
        "--k",
        help="Number of top results to return from the vector search.",
    ),
    relative_paths: bool = typer.Option(
        False,
        "--relative-paths/--absolute-paths",  # Provides --no-relative-paths as well
        help=(
            "Display relative file paths. "
            "Paths are relative to the initial search target directory, or its parent if the target is a file. "
            "Only used with '--output paths'."
        ),
    ),
    project: str = typer.Option(
        "default",
        "--project",
        help="Project name to use for persistent search.",
    ),
) -> None:
    """Searches for a query within files.

    If ``path_to_search`` is a directory, files are discovered using the given
    ``patterns`` (default ``['*.txt']``). If ``path_to_search`` is omitted the
    persistent index is searched instead.
    """
    global_simgrep_config: SimgrepConfig = load_or_create_global_config()  # Load config early
    search_engine = SearchEngine(console)
    project_cfg = global_simgrep_config.projects.get(project)
    if project_cfg is None:
        console.print(f"[bold red]Error: Project '{project}' not found.[/bold red]")
        raise typer.Exit(code=1)

    if path_to_search is None:
        # --- Persistent Search ---
        console.print(
            f"Searching for: '[bold blue]{query_text}[/bold blue]' in project '[magenta]{project}[/magenta]'"
        )
        project_db_file = project_cfg.db_path
        project_usearch_file = project_cfg.usearch_index_path

        if not project_db_file.exists() or not project_usearch_file.exists():
            console.print(
                f"[bold yellow]Warning: Persistent index for project '{project}' not found at '{project_cfg.db_path.parent}'.[/bold yellow]\n"
                f"Please run 'simgrep index <path> --project {project}' first to create an index."
            )
            if output == OutputMode.paths:
                console.print("No matching files found.")
                raise typer.Exit()
            raise typer.Exit(code=1)

        persistent_store: Optional[MetadataStore] = None
        persistent_vector_index: Optional[usearch.index.Index] = None
        try:
            console.print("  Loading persistent database...")
            persistent_store = MetadataStore(persistent=True, db_path=project_db_file)
            console.print("  Loading persistent vector index...")
            persistent_vector_index = load_persistent_index(project_usearch_file)

            if persistent_vector_index is None:
                console.print(
                    f"[bold red]Error: Vector index file loaded as None from {project_usearch_file}. "
                    "Index might be corrupted or empty after a failed write.[/bold red]"
                )
                raise typer.Exit(code=1)
            if len(persistent_vector_index) == 0:
                console.print("[yellow]Warning: The persistent vector index is empty. No search can be performed.[/yellow]")
                # Output "no results" based on mode
                if output == OutputMode.paths:
                    console.print(
                        format_paths(
                            file_paths=[],
                            use_relative=False,
                            base_path=None,
                            console=console,
                        )
                    )
                else:  # show
                    console.print("  No relevant chunks found in the persistent index.")
                raise typer.Exit()

            search_engine.search_persistent(
                query_text=query_text,
                metadata_store=persistent_store,
                vector_index=persistent_vector_index,
                global_config=global_simgrep_config,
                output_mode=output,
                k_results=top,
                display_relative_paths=relative_paths,
                base_path_for_relativity=Path.cwd() if relative_paths else None,
            )
        except (MetadataDBError, VectorStoreError, RuntimeError, ValueError) as e:
            console.print(f"[bold red]Error during persistent search: {e}[/bold red]")
            raise typer.Exit(code=1)
        finally:
            if persistent_store:
                persistent_store.close()
                console.print("  Persistent database connection closed.")
        return  # End of persistent search path

    # --- Ephemeral Search (path_to_search is provided) ---
    console.print(f"Performing ephemeral search for: '[bold blue]{query_text}[/bold blue]' in path: '[green]{path_to_search}[/green]'")
    store: Optional[MetadataStore] = None
    try:
        # --- Initialize In-Memory Database ---
        console.print("\n[bold]Setup: Initializing In-Memory Database[/bold]")
        store = MetadataStore()
        console.print("  In-memory database and tables created.")

        # --- Load Tokenizer ---
        console.print("\n[bold]Setup: Loading Tokenizer[/bold]")
        console.print(f"  Loading tokenizer for model: '{global_simgrep_config.default_embedding_model_name}'...")
        try:
            tokenizer = load_tokenizer(global_simgrep_config.default_embedding_model_name)
            console.print(f"    Tokenizer loaded successfully: {tokenizer.__class__.__name__}")
        except RuntimeError as e:
            console.print(f"[bold red]Fatal Error: Could not load tokenizer.[/bold red]\n  Details: {e}")
            raise typer.Exit(code=1)

        # --- File Discovery ---
        files_to_process: List[Path] = []
        files_skipped: List[Tuple[Path, str]] = []

        search_patterns = list(patterns) if patterns else ["*.txt"]
        files_to_process = gather_files_to_process(path_to_search, search_patterns)

        if path_to_search.is_file():
            console.print(f"Processing single file: [green]{path_to_search}[/green]")
        else:
            console.print(f"Scanning directory: [green]{path_to_search}[/green] for files matching: {search_patterns}...")
            if not files_to_process:
                console.print(f"[yellow]No files found in directory {path_to_search} with patterns {search_patterns}[/yellow]")
            else:
                console.print(f"Found {len(files_to_process)} file(s) to process.")

        if not files_to_process:
            console.print("No files selected for processing. Exiting.")
            raise typer.Exit()

        # --- Text Extraction and Token-based Chunking ---
        all_chunkdata_objects: List[ChunkData] = []
        global_usearch_label_counter: int = 0

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
                        console.print(f"    [yellow]Skipped: File '{file_path_item}' is empty or contains only whitespace.[/yellow]")
                        files_skipped.append((file_path_item, "Empty or whitespace-only"))
                        progress.advance(processing_task)
                        continue

                    intermediate_chunks_info: List[ProcessedChunkInfo] = chunk_text_by_tokens(
                        full_text=extracted_content,
                        tokenizer=tokenizer,
                        chunk_size_tokens=global_simgrep_config.default_chunk_size_tokens,
                        overlap_tokens=global_simgrep_config.default_chunk_overlap_tokens,
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
                            f"    [yellow]No token-based chunks generated for '{file_path_item}' "
                            f"(text might be too short or empty for current parameters).[/yellow]"
                        )
                        files_skipped.append((file_path_item, "No token-based chunks generated"))

                except FileNotFoundError:
                    console.print(f"    [bold red]Error: File not found during processing loop: {file_path_item}. Skipping.[/bold red]")
                    files_skipped.append((file_path_item, "File not found during processing"))
                except RuntimeError as e:
                    console.print(f"    [bold red]Error processing or chunking file '{file_path_item}': {e}. Skipping.[/bold red]")
                    files_skipped.append((file_path_item, str(e)))
                except ValueError as ve:
                    console.print(f"    [bold red]Error with chunking parameters for file '{file_path_item}': {ve}. Skipping.[/bold red]")
                    files_skipped.append((file_path_item, str(ve)))
                except Exception as e:
                    console.print(f"    [bold red]Unexpected error processing file '{file_path_item}': {e}. Skipping.[/bold red]")
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

        # --- Populate In-Memory Database ---
        console.print("\n[bold]Setup: Populating In-Memory Database[/bold]")
        unique_files_metadata_dict: Dict[int, Path] = {}
        for cd_item in all_chunkdata_objects:
            if cd_item.source_file_id not in unique_files_metadata_dict:
                unique_files_metadata_dict[cd_item.source_file_id] = cd_item.source_file_path

        processed_files_metadata_for_db: List[Tuple[int, Path]] = [(fid, fpath) for fid, fpath in unique_files_metadata_dict.items()]

        if processed_files_metadata_for_db:
            assert store is not None
            store.batch_insert_files(processed_files_metadata_for_db)
            console.print(f"  Inserted metadata for {len(processed_files_metadata_for_db)} file(s) into DB.")

        assert store is not None
        store.batch_insert_chunks(all_chunkdata_objects)
        console.print(f"  Inserted {len(all_chunkdata_objects)} chunk(s) into DB.")

        # --- Embedding Generation ---
        query_embedding: np.ndarray
        chunk_embeddings: np.ndarray

        console.print(f"\n[bold]Step 3: Generating Embeddings for {len(all_chunkdata_objects)} total chunk(s)[/bold]")
        console.print("  (This may take a moment on first run if the embedding model needs to be downloaded...)")

        try:
            # Load embedding model once for ephemeral search
            from sentence_transformers import SentenceTransformer

            from .processor import load_embedding_model

            embedding_model_instance: Optional[SentenceTransformer] = None
            try:
                console.print(f"  Loading embedding model '{global_simgrep_config.default_embedding_model_name}'...")
                embedding_model_instance = load_embedding_model(global_simgrep_config.default_embedding_model_name)
                console.print("    Embedding model loaded.")
            except Exception as e_model_load:
                console.print(f"[bold red]Fatal Error: Could not load embedding model.[/bold red]\n  Details: {e_model_load}")
                raise typer.Exit(code=1)

            console.print(f"  Embedding query: '[italic blue]{query_text}[/italic blue]'...")
            query_embedding = generate_embeddings(
                texts=[query_text],
                model=embedding_model_instance,  # Pass pre-loaded model
            )
            console.print(f"    Query embedding shape: {query_embedding.shape}")

            chunk_texts_for_embedding: List[str] = [cd.text for cd in all_chunkdata_objects]
            console.print(f"  Embedding {len(chunk_texts_for_embedding)} text chunk(s)...")
            chunk_embeddings = generate_embeddings(
                texts=chunk_texts_for_embedding,
                model=embedding_model_instance,  # Pass pre-loaded model
            )
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

        # --- In-Memory Vector Search ---
        console.print("\n[bold]Step 4: Performing In-Memory Vector Search[/bold]")
        search_matches: List[SearchResult] = []

        if chunk_embeddings.size == 0 or chunk_embeddings.shape[0] == 0:
            console.print("  No chunk embeddings available. Skipping vector search.")
        else:
            try:
                console.print(f"  Creating in-memory index for {chunk_embeddings.shape[0]} chunk embedding(s)...")
                usearch_labels_np = np.array([cd.usearch_label for cd in all_chunkdata_objects], dtype=np.int64)
                vector_index: usearch.index.Index = create_inmemory_index(embeddings=chunk_embeddings, labels_for_usearch=usearch_labels_np)
                console.print(f"    Index created with {len(vector_index)} item(s). " f"Metric: {vector_index.metric}, DType: {str(vector_index.dtype)}")

                console.print(f"  Searching index for top {top} similar chunk(s)...")
                search_matches = search_inmemory_index(
                    index=vector_index,
                    query_embedding=query_embedding,
                    k=top,
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

        if relative_paths and output != OutputMode.paths:
            console.print(
                "[yellow]Warning: --relative-paths is only effective with --output paths. "
                "Paths will be displayed according to the selected output mode.[/yellow]"
            )

        if not search_matches:
            if output == OutputMode.paths:
                console.print(
                    format_paths(
                        file_paths=[],
                        use_relative=False,
                        base_path=None,
                        console=console,
                    )
                )  # Handles "No matching files found."
            else:  # OutputMode.show or other future modes that might show "no results"
                console.print("  No relevant chunks found for your query in the processed file(s).")
        else:
            if output == OutputMode.paths:
                paths_from_matches: List[Path] = []
                for result in search_matches:
                    matched_chunk_id = result.label
                    assert store is not None
                    retrieved_details = store.retrieve_chunk_for_display(matched_chunk_id)
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
                    else:  # path_to_search is a file
                        current_base_path_for_relativity = path_to_search.parent

                output_string = format_paths(
                    file_paths=paths_from_matches,
                    use_relative=actual_use_relative,
                    base_path=current_base_path_for_relativity,
                    console=console,
                )
                if output_string:
                    console.print(output_string)

            elif output == OutputMode.show:
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
                        console.print("---")  # Visual separator
                        console.print(output_string)
                    else:
                        console.print(f"[bold yellow]Warning: Could not retrieve details for chunk_id {matched_chunk_id} " f"from DB.[/bold yellow]")

    finally:
        if store:
            console.print("\n[bold]Cleanup: Closing In-Memory Database[/bold]")
            store.close()
            console.print("  Database connection closed.")


@app.command()
def index(
    path_to_index: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        resolve_path=True,  # Ensures path is absolute and symlinks resolved
        help="The path to the file or directory to index.",
    ),
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        help="Wipe existing data and rebuild the project's index from scratch.",
    ),
    project: str = typer.Option(
        "default",
        "--project",
        help="Project name to index.",
    ),
) -> None:
    """
    Creates or updates a persistent index for the specified path within a project.
    If --rebuild is provided, existing data for that project will be deleted before indexing.
    """
    console.print(f"Starting indexing for path: [green]{path_to_index}[/green] into project '[magenta]{project}[/magenta]'")
    if rebuild:
        console.print(f"[bold yellow]Warning: This will wipe and rebuild the '{project}' project index.[/bold yellow]")
        if not typer.confirm(
            f"Are you sure you want to wipe and rebuild the '{project}' project index?",
            default=False,
        ):
            console.print("Aborted indexing.")
            raise typer.Abort()

    try:
        global_simgrep_config: SimgrepConfig = load_or_create_global_config()
        project_cfg = global_simgrep_config.projects.get(project)
        if project_cfg is None:
            console.print(f"[bold red]Error: Project '{project}' not found.[/bold red]")
            raise typer.Exit(code=1)

        project_db_file = project_cfg.db_path
        project_usearch_file = project_cfg.usearch_index_path

        # Prepare configuration for the Indexer
        indexer_config = IndexerConfig(
            project_name=project,
            db_path=project_db_file,
            usearch_index_path=project_usearch_file,
            embedding_model_name=global_simgrep_config.default_embedding_model_name,
            chunk_size_tokens=global_simgrep_config.default_chunk_size_tokens,
            chunk_overlap_tokens=global_simgrep_config.default_chunk_overlap_tokens,
            file_scan_patterns=["*.txt"],  # Initially hardcode to .txt, make configurable later
        )

        indexer_instance = Indexer(config=indexer_config, console=console)
        indexer_instance.index_path(target_path=path_to_index, wipe_existing=rebuild)

        console.print(f"[bold green]Successfully indexed '{path_to_index}' into project '{project}'.[/bold green]")

    except SimgrepConfigError as e:
        console.print(f"[bold red]Configuration Error:[/bold red]\n  {e}")
        raise typer.Exit(code=1)
    except IndexerError as e:  # Custom error from Indexer
        console.print(f"[bold red]Indexing Error:[/bold red]\n  {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred during indexing:[/bold red]\n  {e}")
        # For detailed debugging, consider logging the full traceback
        # import traceback
        # console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(code=1)


@app.command()
def status() -> None:
    try:
        cfg: SimgrepConfig = load_or_create_global_config()
    except SimgrepConfigError:
        raise typer.Exit(code=1)

    db_file = cfg.default_project_data_dir / "metadata.duckdb"
    if not db_file.exists():
        console.print(
            f"[bold yellow]Warning: Default persistent index not found at '{cfg.default_project_data_dir}'.[/bold yellow]\n"
            "Please run 'simgrep index <path>' first to create an index."
        )
        raise typer.Exit(code=1)

    store: Optional[MetadataStore] = None
    try:
        store = MetadataStore(persistent=True, db_path=db_file)
        files_count, chunks_count = get_index_counts(store.conn)
        console.print(f"Default Project: {files_count} files indexed, {chunks_count} chunks.")
    except MetadataDBError as e:
        console.print(f"[bold red]Error retrieving status: {e}[/bold red]")
        raise typer.Exit(code=1)
    finally:
        if store:
            store.close()


@project_app.command("create")
def project_create(name: str) -> None:
    """Create a new named project."""
    try:
        cfg = load_or_create_global_config()
    except SimgrepConfigError:
        raise typer.Exit(code=1)

    global_db_path = cfg.db_directory / "global_metadata.duckdb"
    conn = connect_global_db(global_db_path)
    try:
        if get_project_by_name(conn, name) is not None:
            console.print(f"[bold red]Project '{name}' already exists.[/bold red]")
            raise typer.Exit(code=1)

        proj_cfg = add_project_to_config(cfg, name)
        insert_project(
            conn,
            project_name=name,
            db_path=str(proj_cfg.db_path),
            usearch_index_path=str(proj_cfg.usearch_index_path),
            embedding_model_name=proj_cfg.embedding_model,
        )
        console.print(f"[green]Project '{name}' created.[/green]")
    except (MetadataDBError, SimgrepConfigError) as e:
        console.print(f"[bold red]Error creating project: {e}[/bold red]")
        raise typer.Exit(code=1)
    finally:
        conn.close()


@project_app.command("list")
def project_list() -> None:
    """List all configured projects."""
    try:
        cfg = load_or_create_global_config()
    except SimgrepConfigError:
        raise typer.Exit(code=1)

    global_db_path = cfg.db_directory / "global_metadata.duckdb"
    conn = connect_global_db(global_db_path)
    try:
        projects = get_all_projects(conn)
    except MetadataDBError as e:
        console.print(f"[bold red]Error listing projects: {e}[/bold red]")
        raise typer.Exit(code=1)
    finally:
        conn.close()

    for proj in projects:
        label = " (default)" if proj == "default" else ""
        console.print(f"{proj}{label}")


if __name__ == "__main__":
    app()
