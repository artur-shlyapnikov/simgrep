import os
import sys
import warnings
from importlib.metadata import version
from pathlib import Path
from typing import (
    List,
    Optional,
)

import typer
from rich.console import Console

try:
    from .config import (
        DEFAULT_K_RESULTS,
        SimgrepConfigError,
        initialize_global_config,
        load_global_config,
    )
    from .core.context import SimgrepContext
    from .core.errors import MetadataDBError, SimgrepError
    from .ephemeral_searcher import EphemeralSearcher
    from .indexer import Indexer, IndexerConfig, IndexerError
    from .metadata_db import (
        add_project_path,
        connect_global_db,
        create_project_scaffolding,
        get_project_by_name,
        get_project_config,
    )
    from .models import OutputMode, SimgrepConfig
    from .project_manager import ProjectManager
    from .repository import MetadataStore
    from .services.search_service import SearchService
    from .ui.formatters import format_count, format_json, format_paths, format_show_basic
    from .utils import (
        find_project_root,
        get_project_name_from_local_config,
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
            initialize_global_config,
            load_global_config,
        )
        from simgrep.core.context import SimgrepContext
        from simgrep.core.errors import MetadataDBError, SimgrepError
        from simgrep.ephemeral_searcher import EphemeralSearcher
        from simgrep.indexer import Indexer, IndexerConfig, IndexerError
        from simgrep.metadata_db import (
            add_project_path,
            connect_global_db,
            create_project_scaffolding,
            get_project_by_name,
            get_project_config,
        )
        from simgrep.models import (
            OutputMode,
            SimgrepConfig,
        )
        from simgrep.project_manager import ProjectManager
        from simgrep.repository import MetadataStore
        from simgrep.services.search_service import SearchService
        from simgrep.ui.formatters import format_count, format_json, format_paths, format_show_basic
        from simgrep.utils import (
            find_project_root,
            get_project_name_from_local_config,
        )
    else:
        raise

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="sentence_transformers.SentenceTransformer",
)


app = typer.Typer(
    name="simgrep",
    help="A command-line tool for semantic search in local files.",
    add_completion=False,
    no_args_is_help=True,
)
project_app = typer.Typer(help="Manage simgrep projects.")
app.add_typer(project_app, name="project")
console = Console()


def get_active_project(project_option: Optional[str]) -> str:
    """Determines the project name from CLI option, local config, or default."""
    if project_option:
        return project_option

    project_root = find_project_root()
    if project_root:
        name = get_project_name_from_local_config(project_root)
        if name:
            console.print(f"[dim]Detected project '{name}' from local config.[/dim]")
            return name

    return "default"


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version_flag: Optional[bool] = typer.Option(None, "--version", "-v", help="Show simgrep version and exit.", is_eager=True),
) -> None:
    """
    simgrep CLI application.
    """
    if version_flag:
        try:
            ver = version("simgrep")
            console.print(f"simgrep version {ver}")
        except Exception:
            console.print("simgrep version (could not be determined)")
        raise typer.Exit()

    # The main callback now only handles the version flag.
    # Each command is responsible for loading its required configuration.
    # This makes the CLI more explicit and robust.


@app.command()
def init(
    global_init: bool = typer.Option(
        False,
        "--global",
        help="Initialize the global simgrep configuration in the home directory.",
    ),
) -> None:
    """
    Initializes simgrep. Use --global for first-time setup.
    Run in a directory to initialize it as a simgrep project.
    """
    if global_init:
        config_for_path = SimgrepConfig()
        if config_for_path.config_file.exists():
            if not typer.confirm(
                f"Global config file already exists at {config_for_path.config_file}. Overwrite?",
                default=False,
            ):
                console.print("Aborted.")
                raise typer.Abort()

        try:
            initialize_global_config(overwrite=True)
            console.print(f"[green]Global simgrep configuration initialized at {config_for_path.db_directory}[/green]")
        except SimgrepConfigError as e:
            console.print(f"[bold red]Error initializing global config: {e}[/bold red]")
            raise typer.Exit(code=1)
    else:
        cwd = Path.cwd()
        project_name = cwd.name.lower().replace(" ", "-")
        simgrep_dir = cwd / ".simgrep"

        if simgrep_dir.exists():
            console.print(f"Project '{project_name}' seems to be already initialized at {cwd}.")
            raise typer.Exit()

        try:
            global_cfg = load_global_config()
        except SimgrepConfigError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(code=1)

        global_db_path = global_cfg.db_directory / "global_metadata.duckdb"
        conn = connect_global_db(global_db_path)
        try:
            if get_project_by_name(conn, project_name) is not None:
                console.print(f"[bold red]Error: A project named '{project_name}' already exists globally.[/bold red]")
                console.print("You can either rename your directory or manage projects manually.")
                raise typer.Exit(code=1)

            proj_cfg = create_project_scaffolding(conn, global_cfg, project_name)

            project_info = get_project_by_name(conn, proj_cfg.name)
            if not project_info:
                console.print(f"[bold red]Internal Error: Failed to retrieve project '{proj_cfg.name}' after creation.[/bold red]")
                raise typer.Exit(code=1)
            project_id = project_info[0]
            add_project_path(conn, project_id, str(cwd))

            simgrep_dir.mkdir()
            (simgrep_dir / "config.toml").write_text(f'project_name = "{project_name}"\n')

            console.print(f"[green]Initialized simgrep project '{project_name}' in {cwd}[/green]")
            console.print("Next steps:")
            console.print("  1. Run 'simgrep index' to build the search index for this project.")

        finally:
            conn.close()


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
        help="The path to a text file or directory for ephemeral search. If omitted, searches the active persistent project.",
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
        help="Output mode. 'paths' lists unique file paths. 'json' provides detailed structured output. 'count' shows number of matches.",
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
    project: Optional[str] = typer.Option(
        None,
        "--project",
        help="Project name to use for persistent search. Autodetected if in a project directory.",
    ),
    file_filter: Optional[List[str]] = typer.Option(
        None,
        "--file-filter",
        help="Filter results to files matching glob pattern(s) (e.g., '*.py'). Applied after semantic search.",
    ),
    keyword: Optional[str] = typer.Option(
        None,
        "--keyword",
        help="Additionally filter result chunks by a case-insensitive keyword.",
    ),
    min_score: float = typer.Option(
        0.1,
        "--min-score",
        help="Minimum similarity score for a result to be included (0.0 to 1.0).",
        min=0.0,
        max=1.0,
    ),
) -> None:
    """Searches for a query within files.

    If ``path_to_search`` is a directory, files are discovered using the given
    ``patterns`` (default ``['*.txt']``). If ``path_to_search`` is omitted the
    persistent index is searched instead.
    """
    is_machine_readable_output = output in (OutputMode.json, OutputMode.paths)

    if path_to_search is None:
        persistent_store: Optional[MetadataStore] = None
        try:
            global_simgrep_config = load_global_config()
            active_project = get_active_project(project)

            if not is_machine_readable_output:
                console.print(f"Searching for: '[bold blue]{query_text}[/bold blue]' in project '[magenta]{active_project}[/magenta]'")

            global_db_path = global_simgrep_config.db_directory / "global_metadata.duckdb"
            with connect_global_db(global_db_path) as conn:
                project_cfg = get_project_config(conn, active_project)

            if project_cfg is None:
                console.print(f"[bold red]Error: Project '{active_project}' not found.[/bold red]")
                raise typer.Exit(code=1)

            if not project_cfg.db_path.exists() or not project_cfg.usearch_index_path.exists():
                console.print(
                    f"[bold yellow]Warning: Persistent index for project '{active_project}' not found.[/bold yellow]\n"
                    f"Please run 'simgrep index --project {active_project}' first."
                )
                if output == OutputMode.paths:
                    console.print("No matching files found.")
                raise typer.Exit(code=1)

            context = SimgrepContext.from_defaults(
                model_name=project_cfg.embedding_model,
                chunk_size=global_simgrep_config.default_chunk_size_tokens,
                chunk_overlap=global_simgrep_config.default_chunk_overlap_tokens,
            )
            persistent_store = MetadataStore(persistent=True, db_path=project_cfg.db_path)
            embedder = context.embedder
            vector_index = context.index_factory(embedder.ndim)
            vector_index.load(project_cfg.usearch_index_path)

            search_service = SearchService(store=persistent_store, embedder=embedder, index=vector_index)

            results = search_service.search(
                query=query_text,
                k=top,
                min_score=min_score,
                file_filter=file_filter,
                keyword_filter=keyword,
            )

            if not results:
                if output == OutputMode.paths:
                    console.print(format_paths(file_paths=[], use_relative=False, base_path=None, console=console))
                elif output == OutputMode.json:
                    print("[]")
                elif output == OutputMode.count_results:
                    console.print(format_count([]))
                else:
                    if not is_machine_readable_output:
                        console.print("  No relevant chunks found in the persistent index (after filtering).")
                raise typer.Exit(0)

            if output == OutputMode.show:
                console.print(f"\n[bold cyan]Search Results (Top {len(results)}):[/bold cyan]")
                for r in results:
                    if r.file_path and r.chunk_text:
                        console.print("---")
                        console.print(format_show_basic(file_path=r.file_path, chunk_text=r.chunk_text, score=r.score))
            elif output == OutputMode.paths:
                base = Path.cwd() if relative_paths else None
                output_paths = [r.file_path for r in results if r.file_path]
                print(format_paths(file_paths=output_paths, use_relative=relative_paths, base_path=base, console=console))
            elif output == OutputMode.json:
                print(format_json(results))
            elif output == OutputMode.count_results:
                console.print(format_count(results))

        except (SimgrepError, SimgrepConfigError) as e:
            console.print(f"[bold red]Error during persistent search: {e}[/bold red]")
            raise typer.Exit(code=1)
        finally:
            if persistent_store:
                persistent_store.close()
        return

    global_simgrep_config = load_global_config()
    context = SimgrepContext.from_defaults(
        model_name=global_simgrep_config.default_embedding_model_name,
        chunk_size=global_simgrep_config.default_chunk_size_tokens,
        chunk_overlap=global_simgrep_config.default_chunk_overlap_tokens,
    )
    searcher = EphemeralSearcher(context=context, console=console)
    searcher.search(
        query_text=query_text,
        path_to_search=path_to_search,
        patterns=patterns,
        output_mode=output,
        top=top,
        relative_paths=relative_paths,
        min_score=min_score,
        file_filter=file_filter,
        keyword_filter=keyword,
    )


@app.command()
def index(
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        help="Wipe existing data and rebuild the project's index from scratch.",
    ),
    project: Optional[str] = typer.Option(
        None,
        "--project",
        help="Project name to index. Autodetected if in a project directory.",
    ),
    patterns: Optional[List[str]] = typer.Option(
        None,
        "--pattern",
        "-p",
        help="Glob pattern(s) for files to index. Can be used multiple times. Defaults to project settings or '*.txt'.",
    ),
    workers: Optional[int] = typer.Option(
        None,
        "--workers",
        "-w",
        help="Number of concurrent workers for indexing. Defaults to CPU count.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Skip confirmation prompts and assume 'yes'.",
    ),
) -> None:
    """
    Creates or updates a persistent index for all paths in a project.
    If --rebuild is provided, existing data for that project will be deleted before indexing.
    """
    try:
        global_simgrep_config = load_global_config()
    except SimgrepConfigError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)

    active_project = get_active_project(project)
    console.print(f"Starting indexing for project '[magenta]{active_project}[/magenta]'")
    if rebuild:
        console.print(f"[bold yellow]Warning: This will wipe and rebuild the '{active_project}' project index.[/bold yellow]")
        if not yes:
            if not typer.confirm(
                f"Are you sure you want to wipe and rebuild the '{active_project}' project index?",
                default=False,
            ):
                console.print("Aborted indexing.")
                raise typer.Abort()

    try:
        global_db_path = global_simgrep_config.db_directory / "global_metadata.duckdb"
        conn = connect_global_db(global_db_path)
        try:
            project_cfg = get_project_config(conn, active_project)
        finally:
            conn.close()

        if project_cfg is None:
            console.print(f"[bold red]Error: Project '{active_project}' not found.[/bold red]")
            raise typer.Exit(code=1)

        if not project_cfg.indexed_paths:
            console.print(f"[yellow]Warning: Project '{active_project}' has no paths to index.[/yellow]")
            console.print(f"Use 'simgrep project add-path <path> --project {active_project}' to add paths.")
            raise typer.Exit()

        project_db_file = project_cfg.db_path
        project_usearch_file = project_cfg.usearch_index_path

        scan_patterns = patterns if patterns else ["*.txt"]

        indexer_config = IndexerConfig(
            project_name=active_project,
            db_path=project_db_file,
            usearch_index_path=project_usearch_file,
            embedding_model_name=project_cfg.embedding_model,
            chunk_size_tokens=global_simgrep_config.default_chunk_size_tokens,
            chunk_overlap_tokens=global_simgrep_config.default_chunk_overlap_tokens,
            file_scan_patterns=scan_patterns,
            max_index_workers=workers if workers is not None else (os.cpu_count() or 4),
        )

        context = SimgrepContext.from_defaults(
            model_name=indexer_config.embedding_model_name,
            chunk_size=indexer_config.chunk_size_tokens,
            chunk_overlap=indexer_config.chunk_overlap_tokens,
        )

        indexer_instance = Indexer(config=indexer_config, context=context, console=console)
        indexer_instance.run_index(target_paths=project_cfg.indexed_paths, wipe_existing=rebuild)

        console.print(f"[bold green]Successfully indexed project '{active_project}'.[/bold green]")

    except SimgrepConfigError as e:
        console.print(f"[bold red]Configuration Error:[/bold red]\n  {e}")
        raise typer.Exit(code=1)
    except IndexerError as e:  # Custom error from Indexer
        console.print(f"[bold red]Indexing Error:[/bold red]\n  {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred during indexing:[/bold red]\n  {e}")
        raise typer.Exit(code=1)


@app.command()
def status() -> None:
    try:
        cfg = load_global_config()
    except SimgrepConfigError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)

    db_file = cfg.default_project_data_dir / "metadata.duckdb"
    if not db_file.exists():
        console.print("Default Project: 0 files indexed, 0 chunks.")
        raise typer.Exit()

    store: Optional[MetadataStore] = None
    try:
        store = MetadataStore(persistent=True, db_path=db_file)
        files_count, chunks_count = store.get_index_counts()
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
        cfg = load_global_config()
        manager = ProjectManager(cfg)
        manager.create_project(name)
        console.print(f"[green]Project '{name}' created.[/green]")
    except (SimgrepConfigError, MetadataDBError) as e:
        console.print(f"[bold red]Error creating project: {e}[/bold red]")
        raise typer.Exit(code=1)


@project_app.command("add-path")
def project_add_path(
    path_to_add: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="The file or directory path to add to the project for indexing.",
    ),
    project: Optional[str] = typer.Option(
        None,
        "--project",
        help="The name of the project to add the path to. Autodetected if in a project directory.",
    ),
) -> None:
    """Adds a path to a project. Simgrep will scan this path during indexing."""
    try:
        cfg = load_global_config()
        manager = ProjectManager(cfg)
    except SimgrepConfigError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)

    active_project = get_active_project(project)

    try:
        manager.add_path(active_project, path_to_add)
        console.print(f"Added path '[green]{path_to_add}[/green]' to project '[magenta]{active_project}[/magenta]'.")
    except MetadataDBError as e:
        console.print(f"[bold red]Error adding path to project: {e}[/bold red]")
        raise typer.Exit(code=1)


@project_app.command("list")
def project_list() -> None:
    """List all configured projects."""
    try:
        cfg = load_global_config()
        manager = ProjectManager(cfg)
        projects = manager.list_projects()
    except (SimgrepConfigError, MetadataDBError) as e:
        console.print(f"[bold red]Error listing projects: {e}[/bold red]")
        raise typer.Exit(code=1)

    for proj in projects:
        label = " (default)" if proj == "default" else ""
        console.print(f"{proj}{label}")


if __name__ == "__main__":
    app()