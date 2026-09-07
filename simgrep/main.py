from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Optional

import typer
from rich.console import Console

from simgrep.config import load_app_config, set_config_value
from simgrep.errors import (
    ConfigError,
    ProjectError,
    RerankError,
    SearchError,
    SimgrepError,
)
from simgrep.models import (
    AppConfig,
    ClustersOptions,
    DebtOptions,
    DebtReport,
    DiffOptions,
    DiffOutcome,
    DiversityMode,
    FreshnessMode,
    IndexOptions,
    LexicalFallbackMode,
    PathBoost,
    ProjectConfig,
    RenderOptions,
    RerankMatch,
    RerankOptions,
    RerankReport,
    ResultFormat,
    SearchOptions,
    SimilarOptions,
)
from simgrep.output import (
    CLUSTER_FORMATS,
    DIFF_FORMATS,
    PACK_FORMATS,
    render_cluster_outcome,
    render_diff_outcome,
    render_pack_report,
    render_search_results,
)
from simgrep.pack import PackOutcome
from simgrep.project import add_indexed_path, find_active_project, init_project, project_covers_path, remove_indexed_path, require_active_project

if TYPE_CHECKING:
    from simgrep.runtime import Runtime, RuntimeFactory


_LAZY_ENGINE_IMPORTS: dict[str, str] = {
    "ClustersEngine": "simgrep.clusters_engine",
    "DiffEngine": "simgrep.diff_engine",
    "IndexEngine": "simgrep.indexing",
    "Runtime": "simgrep.runtime",
    "RuntimeFactory": "simgrep.runtime",
    "SearchEngine": "simgrep.search",
    "Store": "simgrep.store",
    "resolve_anchor": "simgrep.search",
}


def __getattr__(name: str) -> Any:
    """Lazily materialize engine symbols (PEP 562).

    Keeps `import simgrep.main` free of the heavy engine chain (numpy, duckdb,
    pathspec) so daemon-served searches skip ~90ms of imports.
    Runtime construction now resolves through ``simgrep.execution.factory``;
    the runtime test seam is ``monkeypatch.setattr("simgrep.execution.RuntimeFactory", ...)``.
    """
    module_name = _LAZY_ENGINE_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


app = typer.Typer(
    name="simgrep",
    help="Semantic grep for local files.",
    add_completion=False,
    no_args_is_help=True,
    invoke_without_command=True,
)
project_app = typer.Typer(help="Manage local project paths.")
models_app = typer.Typer(help="Manage model cache.")
config_app = typer.Typer(help="Manage config.")
app.add_typer(project_app, name="project")
app.add_typer(models_app, name="models")
app.add_typer(config_app, name="config")

console = Console()
stderr_console = Console(stderr=True)


class _IndexProgress:
    def phase(self, message: str) -> None:
        stderr_console.print(f"simgrep: {message}", markup=False, highlight=False)


def _fail(prefix: str, exc: Exception) -> None:
    stderr_console.print(f"[bold red]{prefix}: {exc}[/bold red]")
    hint = getattr(exc, "hint", None)
    if hint:
        stderr_console.print(f"Hint: {hint}")


def _runtime_for_project(factory: RuntimeFactory, project: ProjectConfig, app_config: AppConfig) -> Runtime:
    project_runtime_config = project
    if project.model == app_config.model and project.chunk_size == app_config.chunk_size and project.chunk_overlap == app_config.chunk_overlap:
        return factory.for_app(app_config)
    return factory.for_project(project_runtime_config)


@app.callback()
def main(
    ctx: typer.Context,
    version_flag: Optional[bool] = typer.Option(None, "--version", "-v", help="Show version and exit.", is_eager=True),
    project_root: Path | None = typer.Option(
        None,
        "--project-root",
        "-C",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Run command as if started in this directory.",
    ),
) -> None:
    if project_root is not None:
        os.chdir(project_root)
    if version_flag:
        from importlib.metadata import PackageNotFoundError, version

        try:
            console.print(f"simgrep version {version('simgrep')}")
        except PackageNotFoundError:
            console.print("simgrep version (editable)")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


@app.command()
def init(
    path: Path = typer.Argument(Path("."), file_okay=False, dir_okay=True, resolve_path=True, help="Project root."),
    yes: bool = typer.Option(False, "--yes", help="Overwrite existing project config."),
) -> None:
    """Create a simgrep project in a directory."""
    try:
        cfg = init_project(path, load_app_config(), yes=yes)
    except (SimgrepError, ConfigError, ProjectError) as exc:
        _fail("Error", exc)
        raise typer.Exit(code=1) from exc
    console.print(f"Initialized simgrep project '{cfg.name}' in {cfg.root}")
    console.print('Next: run `simgrep index`, then `simgrep search "your query"`.')


@app.command()
def index(
    rebuild: bool = typer.Option(False, "--rebuild", help="Rebuild metadata and vector index."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan without writing."),
    include: list[str] | None = typer.Option(None, "--include", help="Only include matching globs."),
    exclude: list[str] | None = typer.Option(None, "--exclude", help="Exclude matching globs."),
    pattern: list[str] | None = typer.Option(None, "--pattern", "-p", help="File scan glob."),
    workers: int = typer.Option(4, "--workers", min=1, help="Extraction workers."),
) -> None:
    """Build or update the persistent index."""
    started = perf_counter()
    try:
        app_config = load_app_config()
        project = require_active_project()
        # Dry-run never touches the runtime: skip model load and index construction.
        from simgrep.execution import factory as _resolve_factory
        from simgrep.main import IndexEngine

        runtime = None if dry_run else _runtime_for_project(_resolve_factory(), project, app_config)
        stats = IndexEngine(runtime).index_project(
            project,
            app_config,
            IndexOptions(
                rebuild=rebuild,
                dry_run=dry_run,
                include_globs=tuple(include or ()),
                exclude_globs=tuple(exclude or ()),
                patterns=tuple(pattern or ()),
                max_workers=workers,
            ),
        )
    except (SimgrepError, ConfigError, ProjectError) as exc:
        _fail("Error", exc)
        raise typer.Exit(code=1) from exc
    if dry_run:
        console.print(f"Would index {stats.files_indexed} file(s).")
    else:
        console.print(f"Indexed {stats.files_indexed} file(s), {stats.chunks_indexed} chunk(s) in {perf_counter() - started:.2f}s.")
    if stats.files_skipped_unchanged:
        console.print(f"Unchanged: {stats.files_skipped_unchanged}")
    if stats.files_pruned_deleted:
        console.print(f"Deleted: {stats.files_pruned_deleted}")


@app.command()
def search(
    query: str | None = typer.Argument(None, help="Search query."),
    path: Path | None = typer.Argument(None, file_okay=True, dir_okay=True, resolve_path=True, help="Optional path scope."),
    persistent: bool = typer.Option(False, "--persistent", help="Require persistent project search."),
    ephemeral: bool = typer.Option(False, "--ephemeral", help="Force one-off indexing."),
    format: ResultFormat = typer.Option(ResultFormat.rich, "--format", "-o", case_sensitive=False, metavar="FORMAT", help="Output format."),
    top: int = typer.Option(5, "--top", "--k", min=1, help="Number of results."),
    min_score: float = typer.Option(0.0, "--min-score", min=0.0, max=1.0, help="Minimum score."),
    candidates: int | None = typer.Option(None, "--candidates", min=1, help="Semantic candidate pool."),
    lexical_top: int | None = typer.Option(None, "--lexical-top", min=0, help="Lexical candidate count."),
    lexical_weight: float | None = typer.Option(None, "--lexical-weight", min=0.0, max=1.0, help="Lexical rank weight."),
    hybrid: bool = typer.Option(True, "--hybrid/--no-hybrid", help="Enable hybrid ranking."),
    expr: str | None = typer.Option(None, "--expr", help="Boolean semantic expression: AND / OR / NOT (uppercase), quotes for phrases, parens for grouping."),
    lexical_fallback: LexicalFallbackMode = typer.Option(
        LexicalFallbackMode.fill, "--lexical-fallback", case_sensitive=False, metavar="MODE", help="Lexical-only hits: fill, always, or never."
    ),
    diversity: DiversityMode = typer.Option(
        DiversityMode.window, "--diversity", case_sensitive=False, metavar="MODE", help="Diversify results by window, file, or none."
    ),
    freshness: FreshnessMode | None = typer.Option(None, "--freshness", case_sensitive=False, metavar="MODE", help="Stale index handling."),
    file_filter: list[str] | None = typer.Option(None, "--file-filter", help="Filter result file glob."),
    keyword: str | None = typer.Option(None, "--keyword", help="Filter result text."),
    include: list[str] | None = typer.Option(None, "--include", help="Include path globs."),
    exclude: list[str] | None = typer.Option(None, "--exclude", help="Exclude path globs."),
    pattern: list[str] | None = typer.Option(None, "--pattern", "-p", help="Ephemeral scan glob."),
    prefer: list[str] | None = typer.Option(None, "--prefer", help="Boost path glob."),
    prefer_weight: float = typer.Option(0.15, "--prefer-weight", min=0.0, max=1.0),
    context: int = typer.Option(0, "--context", "-c", min=0, help="Context lines around each hit."),
    max_chars: int | None = typer.Option(None, "--max-chars", min=1, help="Truncate snippets to this many characters."),
    relative_paths: bool = typer.Option(True, "--relative-paths/--absolute-paths"),
    line_numbers: bool = typer.Option(True, "--line-numbers/--no-line-numbers", help="Show line ranges."),
    show_scores: bool = typer.Option(True, "--show-scores/--no-scores", help="Show match scores."),
    why: bool = typer.Option(False, "--why/--no-why", help="Show per-hit scoring breakdown."),
    whole_unit: bool = typer.Option(False, "--whole-unit", help="Expand each hit to its enclosing semantic unit (function/class/block/paragraph)."),
    rerank: bool = typer.Option(False, "--rerank/--no-rerank", help="Rerank the top hits by cross-encoder query fit."),
    rerank_model: str = typer.Option("cross-encoder/ms-marco-MiniLM-L-6-v2", "--rerank-model", help="Cross-encoder model used by --rerank."),
    rerank_top: int = typer.Option(25, "--rerank-top", min=1, help="Window of hybrid-ordered hits reranked when --rerank is on."),
) -> None:
    """Semantic hybrid search over local files."""
    if persistent and ephemeral:
        stderr_console.print("[bold red]Error: --persistent and --ephemeral cannot be combined.[/bold red]")
        raise typer.Exit(code=1)
    if expr is not None and query is not None:
        stderr_console.print("[bold red]Error: query and --expr are mutually exclusive.[/bold red]")
        raise typer.Exit(code=1)
    if expr is None and (query is None or not query.strip()):
        stderr_console.print("[bold red]Error: query cannot be empty.[/bold red]")
        raise typer.Exit(code=1)
    try:
        app_config = load_app_config()
        active = find_active_project(path or Path.cwd())
        use_ephemeral = ephemeral
        if path is not None and not use_ephemeral:
            if active is None or not project_covers_path(active, path):
                if persistent:
                    raise SearchError("Persistent search requires an active project covering the requested path.")
                use_ephemeral = True
        if path is None and active is None:
            raise SearchError("No active project found.", hint="Run `simgrep init` or pass a PATH for ephemeral search.")
        if persistent and active is None:
            raise SearchError("Persistent search requires an active project.")

        if expr is not None and (lexical_top is not None or lexical_weight is not None):
            stderr_console.print("[bold red]Error: lexical options are not supported with --expr (pure semantic scoring).[/bold red]")
            raise typer.Exit(code=1)
        if expr is not None:
            resolved_lexical_top = 0
            resolved_lexical_weight = 0.0
        else:
            resolved_lexical_top = lexical_top if lexical_top is not None else (app_config.lexical_top if hybrid else 0)
            resolved_lexical_weight = lexical_weight if lexical_weight is not None else (app_config.lexical_weight if hybrid else 0.0)
        effective_query = expr if expr is not None else query
        assert effective_query is not None
        options = SearchOptions(
            query=effective_query,
            top=top,
            min_score=min_score,
            candidate_top=candidates,
            lexical_top=resolved_lexical_top,
            lexical_weight=resolved_lexical_weight,
            diversity=diversity,
            scope_path=path.resolve() if path is not None and not use_ephemeral else None,
            file_filter=tuple(file_filter or ()),
            keyword_filter=keyword,
            include_globs=tuple(include or ()),
            exclude_globs=tuple(exclude or ()),
            path_boosts=tuple(PathBoost(p, prefer_weight) for p in (prefer or ())),
            expr=expr,
        )
        from simgrep.execution import RerankRequest, execute_search
        from simgrep.indexing import progress_scope

        reporter = _IndexProgress() if format in {ResultFormat.rich, ResultFormat.compact} else None
        with progress_scope(reporter):
            outcome = execute_search(
                app_config=app_config,
                path=path,
                options=options,
                ephemeral=ephemeral,
                freshness=freshness or None,
                persistent=persistent,
                patterns=tuple(pattern or ()) or None,
                include_globs=tuple(include or ()),
                exclude_globs=tuple(exclude or ()),
                whole_unit=whole_unit,
                rerank=RerankRequest(top=rerank_top, model=rerank_model) if rerank else None,
            )
        render_search_results(
            outcome.results,
            options=RenderOptions(
                format=format,
                relative_paths=relative_paths,
                base_path=outcome.base_path,
                show_scores=show_scores,
                show_line_numbers=line_numbers,
                context_lines=context,
                max_chars=max_chars if max_chars is not None else app_config.max_chars,
                query=effective_query,
                show_why=why,
            ),
            console=Console(quiet=format in {ResultFormat.json, ResultFormat.jsonl, ResultFormat.paths, ResultFormat.count, ResultFormat.grep}),
        )
    except (SimgrepError, ConfigError, ProjectError) as exc:
        _fail("Error", exc)
        raise typer.Exit(code=getattr(exc, "exit_code", 1)) from exc


@app.command()
def similar(
    source: str = typer.Argument(..., help="Like anchor: '-' (stdin), '@path', 'path:start-end', or literal text."),
    target_dir: Path | None = typer.Argument(None, file_okay=False, dir_okay=True, resolve_path=True, help="Optional ephemeral target directory."),
    unlike: str | None = typer.Option(None, "--unlike", help="Unlike anchor; accepts the same SOURCE forms."),
    unlike_weight: float = typer.Option(0.5, "--unlike-weight", help="Contrastive weight lambda in [0.0, 1.0]."),
    include_self: bool = typer.Option(False, "--include-self", help="Keep chunks overlapping the anchor's own file span."),
    persistent: bool = typer.Option(False, "--persistent", help="Require persistent project search."),
    ephemeral: bool = typer.Option(False, "--ephemeral", help="Force one-off indexing."),
    format: ResultFormat = typer.Option(ResultFormat.rich, "--format", "-o", case_sensitive=False, metavar="FORMAT", help="Output format."),
    top: int = typer.Option(5, "--top", "--k", min=1, help="Number of results."),
    min_score: float = typer.Option(0.0, "--min-score", min=0.0, max=1.0, help="Minimum score."),
    scope: Path | None = typer.Option(None, "--scope", file_okay=True, dir_okay=True, resolve_path=True, help="Restrict results to this path."),
    file_filter: list[str] | None = typer.Option(None, "--file-filter", help="Filter result file glob."),
    include: list[str] | None = typer.Option(None, "--include-glob", help="Include path globs."),
    exclude: list[str] | None = typer.Option(None, "--exclude-glob", help="Exclude path globs."),
    diversity: DiversityMode = typer.Option(
        DiversityMode.window, "--diversity", case_sensitive=False, metavar="MODE", help="Diversify results by window, file, or none."
    ),
    freshness: FreshnessMode | None = typer.Option(None, "--freshness", case_sensitive=False, metavar="MODE", help="Stale index handling."),
    why: bool = typer.Option(False, "--why/--no-why", help="Show per-hit scoring breakdown."),
) -> None:
    """Find chunks semantically similar to a text or file."""
    if persistent and ephemeral:
        stderr_console.print("[bold red]Error: --persistent and --ephemeral cannot be combined.[/bold red]")
        raise typer.Exit(code=1)
    if not 0.0 <= unlike_weight <= 1.0:
        stderr_console.print("[bold red]Error: --unlike-weight must be between 0.0 and 1.0.[/bold red]")
        raise typer.Exit(code=1)
    try:
        app_config = load_app_config()
        stdin_text = sys.stdin.read() if ("-" in (source, unlike)) and sys.stdin is not None else None
        from simgrep.main import resolve_anchor

        like_anchor = resolve_anchor(source, stdin_text=stdin_text)
        unlike_anchor = resolve_anchor(unlike, stdin_text=stdin_text) if unlike is not None else None
        options = SimilarOptions(
            search=SearchOptions(
                query=like_anchor.text,
                top=top,
                min_score=min_score,
                lexical_top=app_config.lexical_top,
                lexical_weight=app_config.lexical_weight,
                diversity=diversity,
                scope_path=scope.resolve() if scope is not None else None,
                file_filter=tuple(file_filter or ()),
                include_globs=tuple(include or ()),
                exclude_globs=tuple(exclude or ()),
            ),
            anchor=like_anchor,
            unlike=unlike_anchor,
            unlike_weight=unlike_weight,
            include_self=include_self,
        )
        from simgrep.execution import execute_similar
        from simgrep.indexing import progress_scope

        reporter = _IndexProgress() if format in {ResultFormat.rich, ResultFormat.compact} else None
        with progress_scope(reporter):
            outcome = execute_similar(
                app_config=app_config,
                target_dir=target_dir,
                options=options,
                freshness=freshness or None,
                ephemeral=ephemeral,
                persistent=persistent,
                include_globs=tuple(include or ()),
                exclude_globs=tuple(exclude or ()),
            )
        render_search_results(
            outcome.results,
            options=RenderOptions(
                format=format,
                relative_paths=True,
                base_path=outcome.base_path,
                show_line_numbers=True,
                max_chars=app_config.max_chars,
                query=like_anchor.text,
                show_why=why,
            ),
            console=Console(quiet=format in {ResultFormat.json, ResultFormat.jsonl, ResultFormat.paths, ResultFormat.count, ResultFormat.grep}),
        )
    except (SimgrepError, ConfigError, ProjectError) as exc:
        _fail("Error", exc)
        raise typer.Exit(code=getattr(exc, "exit_code", 1)) from exc


_CLUSTER_FORMATS = ", ".join(CLUSTER_FORMATS)


@app.command()
def clusters(
    path: Path | None = typer.Argument(None, file_okay=True, dir_okay=True, resolve_path=True, help="Optional path to cluster."),
    threshold: float = typer.Option(0.8, "--threshold", help="Minimum intra-cluster pairwise similarity in (0.0, 1.0]."),
    min_size: int = typer.Option(2, "--min-size", min=2, help="Minimum cluster size."),
    top: int = typer.Option(20, "--top", min=1, help="Maximum number of clusters shown."),
    max_chunks: int = typer.Option(50000, "--max-chunks", min=1, help="Hard cap on chunks clustered (O(N^2) guard)."),
    same_file: bool = typer.Option(False, "--same-file", help="Include duplicates within a single file."),
    persistent: bool = typer.Option(False, "--persistent", help="Require persistent project clustering."),
    ephemeral: bool = typer.Option(False, "--ephemeral", help="Force one-off indexing."),
    freshness: FreshnessMode | None = typer.Option(None, "--freshness", case_sensitive=False, metavar="MODE", help="Stale index handling."),
    absolute_paths: bool = typer.Option(False, "--absolute-paths", help="Print absolute file paths."),
    format: str = typer.Option("rich", "--format", "-o", case_sensitive=False, help=f"Output format: {_CLUSTER_FORMATS}."),
) -> None:
    """Find groups of semantically duplicated chunks across the corpus."""
    if persistent and ephemeral:
        stderr_console.print("[bold red]Error: --persistent and --ephemeral cannot be combined.[/bold red]")
        raise typer.Exit(code=1)
    fmt = format.lower()
    if fmt not in CLUSTER_FORMATS:
        stderr_console.print(f"[bold red]Error: --format must be one of {_CLUSTER_FORMATS}.[/bold red]")
        raise typer.Exit(code=1)
    if not 0.0 < threshold <= 1.0:
        stderr_console.print("[bold red]Error: --threshold must be greater than 0.0 and at most 1.0.[/bold red]")
        raise typer.Exit(code=1)
    try:
        app_config = load_app_config()
        options = ClustersOptions(
            threshold=threshold,
            min_size=min_size,
            top=top,
            same_file=same_file,
            max_chunks=max_chunks,
        )
        from simgrep.execution import execute_clusters

        outcome, base_path = execute_clusters(
            app_config=app_config,
            path=path,
            options=options,
            freshness=freshness or None,
            ephemeral=ephemeral,
            persistent=persistent,
        )
        render_cluster_outcome(outcome, format=fmt, relative_paths=not absolute_paths, base_path=base_path)
    except (SimgrepError, ConfigError, ProjectError) as exc:
        _fail("Error", exc)
        raise typer.Exit(code=getattr(exc, "exit_code", 1)) from exc


_DIFF_FORMATS = ", ".join(sorted(DIFF_FORMATS))


@app.command()
def diff(
    path_a: Path = typer.Argument(..., file_okay=True, dir_okay=True, resolve_path=True, help="Old tree."),
    path_b: Path = typer.Argument(..., file_okay=True, dir_okay=True, resolve_path=True, help="New tree."),
    threshold: float = typer.Option(0.8, "--threshold", help="Match similarity threshold in (0.0, 1.0]."),
    top: int = typer.Option(50, "--top", min=1, help="Maximum number of added/removed chunks listed."),
    max_chunks: int = typer.Option(50000, "--max-chunks", min=1, help="Guard for total chunks across both trees."),
    absolute_paths: bool = typer.Option(False, "--absolute-paths", help="Print absolute file paths."),
    format: str = typer.Option("rich", "--format", "-o", case_sensitive=False, help=f"Output format: {_DIFF_FORMATS}."),
) -> None:
    """Semantically diff two trees: what appeared, disappeared, or merely moved."""
    fmt = format.lower()
    if fmt not in DIFF_FORMATS:
        stderr_console.print(f"[bold red]Error: --format must be one of {_DIFF_FORMATS}.[/bold red]")
        raise typer.Exit(code=1)
    if not 0.0 < threshold <= 1.0:
        stderr_console.print("[bold red]Error: --threshold must be greater than 0.0 and at most 1.0.[/bold red]")
        raise typer.Exit(code=1)
    try:
        app_config = load_app_config()
        options = DiffOptions(threshold=threshold, top=top, max_chunks=max_chunks)
        from simgrep.execution import factory as _resolve_factory
        from simgrep.main import DiffEngine

        runtime = _resolve_factory().for_app(app_config)
        outcome: DiffOutcome = DiffEngine(runtime).diff_paths(path_a, path_b, app_config, options)
        render_diff_outcome(
            outcome,
            fmt=fmt,
            absolute_paths=absolute_paths,
            path_a=path_a,
            path_b=path_b,
        )
    except (SimgrepError, ConfigError, ProjectError) as exc:
        _fail("Error", exc)
        raise typer.Exit(code=getattr(exc, "exit_code", 1)) from exc


_EXPAND_FORMATS = ("rich", "json", "text")
_EXPAND_FAMILIES = ("dedent", "brace", "paragraph")


@app.command()
def expand(
    path: Path = typer.Argument(..., file_okay=True, dir_okay=False, resolve_path=True, help="File to expand."),
    line: int = typer.Argument(..., min=1, help="1-based line inside the unit to expand."),
    format: str = typer.Option("rich", "--format", "-o", case_sensitive=False, help="Output format: rich, json, text."),
    max_chars: int | None = typer.Option(None, "--max-chars", min=1, help="Cap unit length in chars; adds a truncation marker."),
    language: str | None = typer.Option(None, "--language", case_sensitive=False, help="Override family detection: dedent, brace, paragraph."),
) -> None:
    """Expand PATH:LINE to its enclosing semantic unit (function/class/block/paragraph).

    Lexical scope analysis only — deterministic, no index required.
    Exit codes: 0 ok, 1 runtime (unreadable file), 2 usage (bad line/format/family).
    """
    fmt = format.lower()
    if fmt not in _EXPAND_FORMATS:
        stderr_console.print(f"[bold red]Error: --format must be one of {_EXPAND_FORMATS}.[/bold red]")
        raise typer.Exit(code=2)
    family_override = language.lower() if language is not None else None
    if family_override is not None and family_override not in _EXPAND_FAMILIES:
        stderr_console.print(f"[bold red]Error: --language must be one of {_EXPAND_FAMILIES}.[/bold red]")
        raise typer.Exit(code=2)
    try:
        import json

        from simgrep.errors import ExpandError
        from simgrep.expand import cap_unit, read_text_raw, unit_bounds, unit_family  # Lazy import.
        from simgrep.files import infer_language
        from simgrep.text import compute_line_starts, offset_to_line

        if not path.exists():
            raise ExpandError(f"Path not found: {path}", hint="Pass an existing PATH.", exit_code=2)
        text = read_text_raw(path)
        line_starts = compute_line_starts(text)
        # Mirror the engine's strict-\n offset model: splitlines() would count
        # lone CR / form feeds as breaks that have no char offsets.
        total_lines = len(line_starts) - (1 if text.endswith("\n") else 0)
        if line > total_lines:
            raise ExpandError(f"line {line} out of range", hint=f"file has {total_lines} lines", exit_code=2)
        family = family_override if family_override is not None else unit_family(path)
        anchor = line_starts[line - 1]
        try:
            start, end = unit_bounds(text, anchor, family=family)
        except ValueError as exc:
            raise ExpandError(str(exc), hint=f"file has {total_lines} lines", exit_code=2) from exc
        start_line = offset_to_line(line_starts, start)
        end_line = offset_to_line(line_starts, max(end - 1, start))
        shown_start, shown_end = start, end
        truncated = False
        if max_chars is not None:
            shown_start, shown_end = cap_unit(text, start, end, max_chars=max_chars, anchor=anchor)
            truncated = shown_end < end
        unit_text = text[shown_start:shown_end]
        if truncated:
            unit_text += "..."
        if fmt == "json":
            print(
                json.dumps(
                    {
                        "path": str(path),
                        "start_line": start_line,
                        "end_line": end_line,
                        "start_char": start,
                        "end_char": end,
                        "language": infer_language(path),
                        "family": family,
                        "text": unit_text,
                        "truncated": truncated,
                    }
                )
            )
        elif fmt == "text":
            print(unit_text, end="" if unit_text.endswith("\n") else "\n")
        else:
            shown_last_line = offset_to_line(line_starts, max(shown_end - 1, shown_start))
            console.print(
                f"{path}:{start_line}-{end_line} ({family}, {end_line - start_line + 1} lines)",
                markup=False,
                highlight=False,
            )
            shown_first_line = offset_to_line(line_starts, shown_start)
            hidden_lines = (shown_first_line - start_line) + (end_line - shown_last_line)
            hidden_lines = max(hidden_lines, 1) if truncated else hidden_lines
            for number in range(shown_first_line, shown_last_line + 1):
                line_offset = line_starts[number - 1]
                next_offset = line_starts[number] if number < len(line_starts) else len(text)
                content = text[line_offset:next_offset].rstrip("\n")
                console.print(f"{number:>6}  {content}", markup=False, highlight=False)
            if truncated:
                console.print(f"[+{hidden_lines} more lines]", markup=False, highlight=False)
    except (SimgrepError, ConfigError, ProjectError) as exc:
        _fail("Error", exc)
        raise typer.Exit(code=getattr(exc, "exit_code", 1)) from exc


_PACK_FORMATS = ", ".join(PACK_FORMATS)


def _split_pack_args(args: list[str]) -> tuple[list[str], Path | None]:
    """Split the variadic into queries plus an optional trailing TARGET_DIR.

    The last argument is TARGET_DIR iff it is an existing directory; a lone
    argument is always a query (QUERY is required by the product contract).
    """
    if len(args) > 1:
        candidate = Path(args[-1])
        if candidate.is_dir():
            return args[:-1], candidate.resolve()
    return args, None


def _run_pack(
    queries: list[str],
    target_dir: Path | None,
    *,
    budget: int,
    lam: float,
    per_query: int,
    persistent: bool,
    ephemeral: bool,
) -> PackOutcome:
    """Per-query searches -> label-dedup union -> token-budgeted packing (application layer)."""
    from simgrep.execution import execute_pack

    return execute_pack(queries, target_dir, budget=budget, lam=lam, per_query=per_query, ephemeral=ephemeral, persistent=persistent)


@app.command()
def pack(
    args: list[str] = typer.Argument(..., help="One or more QUERY terms; optionally end with TARGET_DIR (an existing directory)."),
    budget: int = typer.Option(3000, "--budget", min=100, max=200000, help="Token budget."),
    per_query: int = typer.Option(8, "--per-query", min=1, max=50, help="Pool size per query."),
    lam: float = typer.Option(0.7, "--lam", min=0.0, max=1.0, help="Relevance vs diversity weight."),
    persistent: bool = typer.Option(False, "--persistent", help="Require persistent project search."),
    ephemeral: bool = typer.Option(False, "--ephemeral", help="Force one-off indexing."),
    format: str = typer.Option("rich", "--format", "-o", case_sensitive=False, help=f"Output format: {_PACK_FORMATS}."),
) -> None:
    """Assemble queries into one paste-ready context block under a token budget.

    Per-query top-N pools are unioned (deduped by label), then greedily packed
    via MMR-style gains until the budget is exhausted.
    Exit codes: 0 ok, 1 no candidates, 2 usage error.
    """
    if persistent and ephemeral:
        stderr_console.print("[bold red]Error: --persistent and --ephemeral cannot be combined.[/bold red]")
        raise typer.Exit(code=2)
    fmt = format.lower()
    if fmt not in PACK_FORMATS:
        stderr_console.print(f"[bold red]Error: --format must be one of {_PACK_FORMATS}.[/bold red]")
        raise typer.Exit(code=2)
    queries, target_dir = _split_pack_args(args)
    if not queries:
        stderr_console.print("[bold red]Error: at least one QUERY is required.[/bold red]")
        raise typer.Exit(code=2)
    try:
        outcome = _run_pack(queries, target_dir, budget=budget, lam=lam, per_query=per_query, persistent=persistent, ephemeral=ephemeral)
        if not outcome.pool_size:
            stderr_console.print("[yellow]No candidates found.[/yellow]")
            raise typer.Exit(code=1)
        print(render_pack_report(outcome, queries, format=fmt))
    except (SimgrepError, ConfigError, ProjectError) as exc:
        _fail("Error", exc)
        raise typer.Exit(code=getattr(exc, "exit_code", 1)) from exc


@app.command()
def debt(
    target_dir: Path | None = typer.Argument(None, file_okay=False, dir_okay=True, resolve_path=True, help="Optional ephemeral target directory."),
    threshold: float = typer.Option(0.8, "--threshold", min=0.01, max=1.0, help="Cluster cosine: two marker chunks join into one theme at or above this."),
    min_size: int = typer.Option(2, "--min-size", min=1, help="Markers-per-theme floor; smaller components stay scattered."),
    top: int = typer.Option(20, "--top", min=1, max=200, help="Max themes shown."),
    max_members: int = typer.Option(8, "--max-members", min=1, max=50, help="Max matches shown per theme."),
    max_age: float | None = typer.Option(None, "--max-age", min=0.01, help="Gate (days): fail when any dated theme's oldest member commit exceeds this."),
    format: str = typer.Option("rich", "--format", "-o", case_sensitive=False, help="Output format: rich, json, jsonl."),
) -> None:
    """Radar debt markers (TODO/FIXME/XXX/HACK/WORKAROUND) as semantic themes with git ages.

    Marker chunks are clustered by cosine similarity into themes, labeled,
    ranked, and gated on --max-age for CI.
    Exit codes: 0 ok / no markers, 1 gate failure (--max-age exceeded) or error, 2 bounds violation.
    """
    from simgrep.output import _DEBT_FORMATS_TEXT, DEBT_FORMATS, render_debt_report

    fmt = format.lower()
    if fmt not in DEBT_FORMATS:
        stderr_console.print(f"[bold red]Error: --format must be one of {_DEBT_FORMATS_TEXT}.[/bold red]")
        raise typer.Exit(code=2)
    try:
        from simgrep.debt_engine import DebtEngine
        from simgrep.execution import factory as _resolve_factory

        app_config = load_app_config()
        options = DebtOptions(threshold=threshold, min_size=min_size, top=top, max_members=max_members, max_age_days=max_age)
        active = find_active_project(target_dir or Path.cwd())
        if target_dir is None and active is None:
            raise SearchError("No active project found.", hint="Run `simgrep init` or pass a TARGET_DIR for ephemeral scan.")
        factory = _resolve_factory()
        report: DebtReport
        if target_dir is not None and (active is None or not project_covers_path(active, target_dir)):
            runtime = factory.for_app(app_config)
            report = DebtEngine(runtime).debt_path(target_dir, app_config, options)
        else:
            assert active is not None
            runtime = _runtime_for_project(factory, active, app_config)
            report = DebtEngine(runtime).debt_project(active, app_config, options, freshness=app_config.freshness)
        print(render_debt_report(report, format=fmt))
        if report.passed is False:
            raise typer.Exit(code=1)
    except (SimgrepError, ConfigError, ProjectError) as exc:
        _fail("Error", exc)
        raise typer.Exit(code=getattr(exc, "exit_code", 1)) from exc


@app.command()
def rerank(
    query: str = typer.Argument(..., help="Query the chunks are scored against."),
    files: list[str] | None = typer.Argument(None, help="Files to read, chunk, and score; omit when --files-from supplies them."),
    files_from: str | None = typer.Option(None, "--files-from", help="Read more file paths from '-' (stdin) or a paths file, one per line."),
    model: str = typer.Option("cross-encoder/ms-marco-MiniLM-L-6-v2", "--model", help="Cross-encoder model name."),
    max_chunks: int = typer.Option(512, "--max-chunks", min=1, help="Cap on total chunks scored."),
    format: str = typer.Option("rich", "--format", "-o", case_sensitive=False, help="Output format: rich, json, jsonl."),
) -> None:
    """Score FILE chunks against QUERY with a local cross-encoder; rank each file's best chunk.

    Zero-infrastructure surface: no index, no store, no embedder. Composable:
    ``grep -rl TODO . | simgrep rerank "error handling" --files-from -``.
    Exit codes: 0 ranked / 1 zero readable files, cap exceeded, or model failure / 2 usage.
    """
    from simgrep.output import _RERANK_FORMATS_TEXT, RERANK_FORMATS, render_rerank_report

    fmt = format.lower()
    if fmt not in RERANK_FORMATS:
        stderr_console.print(f"[bold red]Error: --format must be one of {_RERANK_FORMATS_TEXT}.[/bold red]")
        raise typer.Exit(code=2)
    try:
        from simgrep.adapters.reranker import CrossEncoderReranker  # Lazy: heavy import.
        from simgrep.execution import factory as _resolve_factory
        from simgrep.rerank import best_per_file, chunk_file_texts, ensure_chunk_cap

        paths = [Path(name) for name in (files or ())]
        if files_from is not None:
            if files_from == "-":
                extra = sys.stdin.read().splitlines()
            else:
                try:
                    extra = Path(files_from).read_text(encoding="utf-8").splitlines()
                except OSError as exc:
                    raise RerankError(f"Cannot read --files-from paths file {files_from!r}: {exc}", hint="pass '-' to read paths from stdin") from exc
            paths.extend(Path(line.strip()) for line in extra if line.strip())
        if not paths:
            stderr_console.print("[bold red]Error: no input files given (FILE args or --files-from).[/bold red]")
            raise typer.Exit(code=2)
        app_config = load_app_config()
        runtime = _resolve_factory().for_app(app_config)  # rerank needs no project — ephemeral runtime

        readable: list[tuple[Path, str]] = []
        for candidate in paths:
            try:
                readable.append((candidate, candidate.read_text(encoding="utf-8")))
            except (OSError, UnicodeDecodeError) as exc:
                stderr_console.print(f"[yellow]Warning: skipping unreadable {candidate}: {exc}[/yellow]")
        if not readable:
            raise RerankError("No readable input files.", hint="check the FILE arguments; --files-from - reads paths from stdin")

        scored: list[tuple[Path, Any]] = []
        for source_path, text in readable:
            for chunk in chunk_file_texts(text, runtime.chunker):
                ensure_chunk_cap(len(scored) + 1, max_chunks)
                scored.append((source_path, chunk))
        options = RerankOptions(query=query, model=model, max_chunks=max_chunks)
        chunk_reranker = getattr(runtime, "reranker", None) or CrossEncoderReranker(options.model)
        documents = [chunk.text for _, chunk in scored]
        cross_scores = chunk_reranker.score(query, documents)
        matches = [
            RerankMatch(
                file_path=str(source_path),
                line_start=chunk.line_start or 1,
                line_end=chunk.line_end or chunk.line_start or 1,
                score=float(score),
                snippet=" ".join(chunk.text.split())[:120],
            )
            for (source_path, chunk), score in zip(scored, cross_scores, strict=True)
        ]
        report = RerankReport(
            query=query,
            model=options.model,
            matches=best_per_file(matches),
            files_seen=len(readable),
            chunks_scored=len(documents),
        )
        print(render_rerank_report(report, format=fmt))
    except (SimgrepError, ConfigError, ProjectError) as exc:
        _fail("Error", exc)
        raise typer.Exit(code=getattr(exc, "exit_code", 1)) from exc


@app.command()
def status() -> None:
    """Show the current project index state."""
    try:
        project = require_active_project()
        if not project.metadata_db_path.exists():
            console.print(f"{project.name}: no index. Run `simgrep index` first.")
            return
        from simgrep.main import Store

        store = Store.open(project.metadata_db_path, read_only=True)
        try:
            counts = store.counts(project.name)
        finally:
            store.close()
    except (SimgrepError, ProjectError) as exc:
        _fail("Error", exc)
        raise typer.Exit(code=1) from exc
    state = counts.index_state.value if counts.index_state else "unknown"
    console.print(f"{project.name}: {counts.files_count} file(s), {counts.chunks_count} chunk(s), state={state}")


@app.command()
def mcp() -> None:
    """Run the MCP stdio server (newline-delimited JSON-RPC 2.0 on stdin/stdout)."""
    from simgrep.mcp_server import serve

    raise typer.Exit(code=serve())


@app.command()
def repl() -> None:
    """Interactive semantic query loop for an indexed project."""
    try:
        app_config = load_app_config()
        project = require_active_project()
        from simgrep.execution import factory as _resolve_factory
        from simgrep.main import SearchEngine

        runtime = _runtime_for_project(_resolve_factory(), project, app_config)
        engine = SearchEngine(runtime)
    except (SimgrepError, ProjectError, ConfigError) as exc:
        _fail("Error", exc)
        raise typer.Exit(code=1) from exc
    console.print("simgrep repl. Empty query exits.")
    while True:
        try:
            query = typer.prompt("query", default="", show_default=False)
        except (EOFError, typer.Abort):
            break
        if not query:
            break
        try:
            started = perf_counter()
            outcome = engine.search_project(project, app_config, SearchOptions(query=query), app_config.freshness)
            render_search_results(outcome.results, options=RenderOptions(format=ResultFormat.compact, base_path=project.root), console=console)
            console.print(f"{len(outcome.results)} hit(s) in {perf_counter() - started:.2f}s.")
        except (SimgrepError, ProjectError, ConfigError) as exc:
            _fail("Error", exc)
            continue


@project_app.command("add-path")
def project_add_path(
    path: Path = typer.Argument(..., exists=True, resolve_path=True),
    allow_outside_root: bool = typer.Option(False, "--allow-outside-root"),
) -> None:
    """Register a directory for indexing."""
    try:
        project = add_indexed_path(require_active_project(), path, allow_outside_root=allow_outside_root)
    except (SimgrepError, ProjectError) as exc:
        _fail("Error", exc)
        raise typer.Exit(code=1) from exc
    console.print(f"Indexed paths: {len(project.indexed_paths)}")


@project_app.command("remove-path")
def project_remove_path(path: Path = typer.Argument(..., resolve_path=True), allow_outside_root: bool = typer.Option(False, "--allow-outside-root")) -> None:
    """Drop a directory from the project."""
    try:
        before = require_active_project()
        after = remove_indexed_path(before, path, allow_outside_root=allow_outside_root)
    except (SimgrepError, ProjectError) as exc:
        _fail("Error", exc)
        raise typer.Exit(code=1) from exc
    console.print(f"Indexed paths: {len(after.indexed_paths)}")


@project_app.command("info")
def project_info() -> None:
    """Show project settings and indexed paths."""
    try:
        project = require_active_project()
    except ProjectError as exc:
        _fail("Error", exc)
        raise typer.Exit(code=1) from exc
    console.print(f"name: {project.name}")
    console.print(f"root: {project.root}")
    console.print(f"model: {project.model}")
    for path in project.indexed_paths:
        console.print(f"path: {path}")


@models_app.command("status")
def models_status(model: str | None = typer.Argument(None)) -> None:
    """Check whether a model is in the local cache."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import LocalEntryNotFoundError

    try:
        selected = model or load_app_config().model
    except SimgrepError as exc:
        _fail("Error", exc)
        raise typer.Exit(code=1) from exc
    try:
        snapshot_download(selected, local_files_only=True)
        console.print(f"{selected}: cached")
    except (FileNotFoundError, LocalEntryNotFoundError):
        console.print(f"{selected}: not cached")


@models_app.command("cache")
def models_cache(model: str | None = typer.Argument(None)) -> None:
    """Download a model for offline use."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    try:
        selected = model or load_app_config().model
        AutoTokenizer.from_pretrained(selected)  # type: ignore[no-untyped-call]
        snapshot_download(selected)
    except SimgrepError as exc:
        _fail("Error", exc)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        _fail("Error", SimgrepError(f"Failed to cache model '{selected}': {exc}", hint="Check the model id, e.g. 'simgrep models status <model>'."))
        raise typer.Exit(code=1) from exc
    console.print(f"{selected}: cached")


@config_app.command("list")
def config_list() -> None:
    """Show every config key and its value."""
    try:
        cfg = load_app_config()
    except SimgrepError as exc:
        _fail("Error", exc)
        raise typer.Exit(code=1) from exc
    for key, value in cfg.__dict__.items():
        console.print(f"{key} = {value}")


@config_app.command("get")
def config_get(key: str) -> None:
    """Read one config value."""
    try:
        cfg = load_app_config()
    except SimgrepError as exc:
        _fail("Error", exc)
        raise typer.Exit(code=1) from exc
    if not hasattr(cfg, key):
        _fail("Error", ConfigError(f"Unknown config key: {key}"))
        raise typer.Exit(code=1)
    console.print(getattr(cfg, key))


@config_app.command("set")
def config_set(key: str, value: str) -> None:
    """Persist one config value."""
    try:
        cfg = set_config_value(key, value)
    except ConfigError as exc:
        _fail("Error", exc)
        raise typer.Exit(code=1) from exc
    console.print(f"{key} = {getattr(cfg, key)}")


@app.command()
def doctor() -> None:
    """Sanity-check configuration, model cache, and index."""
    try:
        cfg = load_app_config()
    except SimgrepError as exc:
        _fail("Error", exc)
        raise typer.Exit(code=1) from exc
    from importlib.metadata import version

    console.print(f"simgrep: {version('simgrep')}")
    console.print("config: ok")
    console.print(f"model: {cfg.model}")
    project = find_active_project()
    console.print(f"project: {project.root if project else 'none'}")
    models_status(cfg.model)
    if project is not None and project.metadata_db_path.exists():
        from simgrep.main import Store

        try:
            store = Store.open(project.metadata_db_path, read_only=True)
            try:
                counts = store.counts(project.name)
            finally:
                store.close()
            console.print(f"index: {counts.files_count} file(s), {counts.chunks_count} chunk(s)")
        except SimgrepError as exc:
            _fail("Error", exc)
            raise typer.Exit(code=1) from exc
    elif project is not None:
        console.print("index: missing. Run `simgrep index`.")


@app.command()
def reset(yes: bool = typer.Option(False, "--yes", help="Confirm reset.")) -> None:
    """Delete local index artifacts and keep project configuration."""
    if not yes and not typer.confirm("Delete local simgrep index artifacts?", default=False):
        raise typer.Abort()
    try:
        project = require_active_project()
    except ProjectError as exc:
        _fail("Error", exc)
        raise typer.Exit(code=1) from exc
    for path in (project.metadata_db_path, project.vector_index_path, project.index_lock_path):
        path.unlink(missing_ok=True)
    tmp_dir = project.simgrep_dir / "tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    wal_path = Path(str(project.metadata_db_path) + ".wal")
    wal_path.unlink(missing_ok=True)
    tmp_spill_path = Path(str(project.metadata_db_path) + ".tmp")
    if tmp_spill_path.is_dir():
        shutil.rmtree(tmp_spill_path)
    else:
        tmp_spill_path.unlink(missing_ok=True)
    console.print("Reset local simgrep index artifacts.")


if __name__ == "__main__":
    app()
