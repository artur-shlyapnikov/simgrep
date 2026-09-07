"""Transport-neutral tool registry: the single application boundary shared by
the MCP stdio transport and the HTTP API.

``ToolSpec`` handlers return structured ``JsonValue`` payloads — the machine
projection owned by :mod:`simgrep.records`. Serialization is each transport's
concern: the MCP adapter encodes text content, the HTTP adapter sends the
object directly. The registry lives apart from ``mcp_server`` so the HTTP
transport never imports an MCP module.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

JsonValue = dict[str, Any] | list[Any] | str | int | float | bool | None

SERVER_NAME = "simgrep"


def server_version() -> str:
    try:
        return version("simgrep")
    except PackageNotFoundError:
        return "0.1.0"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], "JsonValue"]


_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "string": lambda value: isinstance(value, str),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
    "array": lambda value: isinstance(value, list),
}


def validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
    """Validate tool arguments against a JSON-Schema subset; return human-readable errors."""
    errors: list[str] = []
    properties = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in arguments:
            errors.append(f"Missing required argument '{key}'.")
    for key, value in arguments.items():
        prop = properties.get(key)
        if not isinstance(prop, dict):
            errors.append(f"Unknown argument '{key}'.")
            continue
        expected = prop.get("type")
        check = _TYPE_CHECKS.get(str(expected))
        if check is not None and not check(value):
            errors.append(f"Argument '{key}' must be of type {expected}.")
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            minimum = prop.get("minimum")
            maximum = prop.get("maximum")
            if isinstance(minimum, (int, float)) and value < minimum:
                errors.append(f"Argument '{key}' must be >= {minimum}.")
                continue
            if isinstance(maximum, (int, float)) and value > maximum:
                errors.append(f"Argument '{key}' must be <= {maximum}.")
        enum_values = prop.get("enum")
        if isinstance(enum_values, list) and value not in enum_values:
            errors.append(f"Argument '{key}' must be one of {enum_values}.")
        items = prop.get("items")
        item_type = str(items["type"]) if isinstance(items, dict) and "type" in items else None
        item_check = _TYPE_CHECKS.get(item_type) if item_type is not None else None
        if isinstance(value, list) and item_check is not None:
            errors.extend(f"Argument '{key}[{index}]' must be of type {item_type}." for index, item in enumerate(value) if not item_check(item))
    return errors


# --- shared handler helpers (lazy imports keep startup light) ---------------


def _project_runtime(factory: Any, project: Any, app_config: Any) -> Any:
    """Reuse the app runtime when project overrides match the global config."""
    if project.model == app_config.model and project.chunk_size == app_config.chunk_size and project.chunk_overlap == app_config.chunk_overlap:
        return factory.for_app(app_config)
    return factory.for_project(project)


def _format_outcome(results: Any, *, base_path: Any, query: str, show_why: bool, max_chars: int) -> JsonValue:
    from simgrep.models import RenderOptions, ResultFormat
    from simgrep.output import enrich_result, format_json

    options = RenderOptions(
        format=ResultFormat.json,
        relative_paths=True,
        base_path=base_path,
        show_scores=True,
        show_line_numbers=True,
        context_lines=0,
        max_chars=max_chars,
        query=query,
        show_why=show_why,
    )
    display = [enrich_result(result, options) for result in results]
    import json as _json

    parsed: JsonValue = _json.loads(format_json(display, show_scores=True, show_why=show_why))
    return parsed


# --- tool handlers ----------------------------------------------------------


def _tool_search(arguments: dict[str, Any]) -> JsonValue:
    from pathlib import Path

    from simgrep.config import load_app_config
    from simgrep.errors import SearchError
    from simgrep.models import DiversityMode, SearchOptions
    from simgrep.project import find_active_project, project_covers_path

    expr = arguments.get("expr")
    use_expr = expr is not None and bool(expr.strip())
    query = arguments.get("query", "")
    if use_expr:
        if query.strip():
            raise SearchError("query and expr are mutually exclusive.", hint="Pass either 'query' or 'expr', not both.")
        query = expr
    elif expr is not None and not query.strip():
        raise SearchError("Expression cannot be empty.", hint="Provide non-empty expr text or omit it to search by query.")
    elif not query.strip():
        raise SearchError("Query cannot be empty.", hint="Provide non-empty query text.")
    app_config = load_app_config()
    raw_path = arguments.get("path")
    path = Path(raw_path).expanduser().resolve() if raw_path else None
    persistent = bool(arguments.get("persistent", False))

    # Coverage check stays visible here only to build scope_path and keep the
    # historical error ordering; mode/runtime/freshness policy lives in execution.
    active = find_active_project(path or Path.cwd())
    use_ephemeral = False
    if path is not None:
        if active is None or not project_covers_path(active, path):
            if persistent:
                raise SearchError("Persistent search requires an active project covering the requested path.")
            use_ephemeral = True
    elif active is None:
        raise SearchError("No active project found.", hint="Run `simgrep init` or pass a PATH for ephemeral search.")

    diversity = DiversityMode(arguments["diversity"]) if "diversity" in arguments else DiversityMode.window
    include_globs = tuple(arguments.get("include") or ())
    exclude_globs = tuple(arguments.get("exclude") or ())
    options = SearchOptions(
        query=query,
        top=int(arguments.get("top", 5)),
        min_score=float(arguments.get("min_score", 0.0)),
        expr=expr if use_expr else None,
        lexical_top=0 if use_expr else app_config.lexical_top,
        lexical_weight=0.0 if use_expr else app_config.lexical_weight,
        diversity=diversity,
        scope_path=None if use_ephemeral else Path(arguments["scope"]).resolve() if arguments.get("scope") else None,
        file_filter=(arguments["file_filter"],) if arguments.get("file_filter") else (),
        include_globs=include_globs,
        exclude_globs=exclude_globs,
    )
    from simgrep.execution import execute_search

    outcome = execute_search(
        app_config=app_config,
        path=path,
        options=options,
        ephemeral=False,
        persistent=persistent,
        include_globs=include_globs,
        exclude_globs=exclude_globs,
        whole_unit=bool(arguments.get("whole_unit", False)),
    )
    show_why = bool(arguments.get("show_why", False))
    return _format_outcome(outcome.results, base_path=outcome.base_path, query=query, show_why=show_why, max_chars=app_config.max_chars)


def _tool_similar(arguments: dict[str, Any]) -> JsonValue:
    from simgrep.config import load_app_config
    from simgrep.errors import SearchError
    from simgrep.models import SearchOptions, SimilarOptions
    from simgrep.search import resolve_anchor

    source = arguments["source"]
    unlike = arguments.get("unlike")
    if source == "-" or unlike == "-":
        raise SearchError("stdin ('-') anchors are unavailable over MCP.", hint="Pass '@file', 'path:start-end', or literal anchor text instead.")

    like_anchor = resolve_anchor(source)
    unlike_anchor = resolve_anchor(unlike) if unlike else None
    app_config = load_app_config()
    from simgrep.execution import execute_similar

    options = SimilarOptions(
        search=SearchOptions(
            query=like_anchor.text,
            top=int(arguments.get("top", 5)),
            min_score=float(arguments.get("min_score", 0.0)),
            lexical_top=app_config.lexical_top,
            lexical_weight=app_config.lexical_weight,
        ),
        anchor=like_anchor,
        unlike=unlike_anchor,
        unlike_weight=float(arguments.get("unlike_weight", 0.5)),
        include_self=bool(arguments.get("include_self", False)),
    )
    outcome = execute_similar(app_config=app_config, target_dir=None, options=options)
    return _format_outcome(outcome.results, base_path=outcome.base_path, query=like_anchor.text, show_why=False, max_chars=app_config.max_chars)


def _tool_clusters(arguments: dict[str, Any]) -> JsonValue:
    from pathlib import Path

    from simgrep.config import load_app_config
    from simgrep.models import ClustersOptions

    app_config = load_app_config()
    raw_path = arguments.get("path")
    path = Path(raw_path).expanduser().resolve() if raw_path else None
    options = ClustersOptions(
        threshold=float(arguments.get("threshold", 0.8)),
        min_size=int(arguments.get("min_size", 2)),
        top=int(arguments.get("top", 20)),
        same_file=bool(arguments.get("same_file", False)),
    )
    # Canonical threshold validity lives in ClustersEngine.run_batch.

    from simgrep.execution import execute_clusters
    from simgrep.records import cluster_record

    outcome, base_path = execute_clusters(app_config=app_config, path=path, options=options)
    return [cluster_record(cluster, relative_paths=True, base_path=base_path) for cluster in outcome.clusters]


def _tool_status(arguments: dict[str, Any]) -> JsonValue:
    del arguments
    from simgrep.project import require_active_project
    from simgrep.store import Store

    project = require_active_project()
    files = chunks = 0
    state: str | None = None
    if project.metadata_db_path.exists():
        store = Store.open(project.metadata_db_path, read_only=True)
        try:
            counts = store.counts(project.name)
        finally:
            store.close()
        files = counts.files_count
        chunks = counts.chunks_count
        state = counts.index_state.value if counts.index_state else None
    payload = {
        "project_root": str(project.root),
        "indexed_paths": [str(path) for path in project.indexed_paths],
        "files": files,
        "chunks": chunks,
        "index_state": state,
        "model": project.model,
    }
    return payload


def _tool_index(arguments: dict[str, Any]) -> JsonValue:
    del arguments
    from dataclasses import asdict

    from simgrep.config import load_app_config
    from simgrep.execution import factory as resolve_factory
    from simgrep.indexing import IndexEngine
    from simgrep.models import IndexOptions
    from simgrep.project import require_active_project
    from simgrep.runtime import assert_safe_bulk_entry

    assert_safe_bulk_entry()
    app_config = load_app_config()
    project = require_active_project()
    runtime = _project_runtime(resolve_factory(), project, app_config)
    stats = IndexEngine(runtime).index_project(project, app_config, IndexOptions())
    return asdict(stats)


def _tool_diff(arguments: dict[str, Any]) -> JsonValue:
    from dataclasses import asdict
    from pathlib import Path

    from simgrep.config import load_app_config
    from simgrep.diff_engine import DiffEngine
    from simgrep.execution import factory as resolve_factory
    from simgrep.models import DiffOptions

    app_config = load_app_config()
    path_a = Path(arguments["a"]).expanduser().resolve()
    path_b = Path(arguments["b"]).expanduser().resolve()
    options = DiffOptions(
        threshold=float(arguments.get("threshold", 0.8)),
        top=int(arguments.get("top", 50)),
        max_chunks=int(arguments.get("max_chunks", 50_000)),
    )
    # Canonical threshold validity lives in DiffEngine.diff_paths.

    # Diff is purely ephemeral: no project, no freshness, no lock.
    factory = resolve_factory()
    runtime = factory.for_app(app_config)
    outcome = DiffEngine(runtime).diff_paths(path_a, path_b, app_config, options)
    return asdict(outcome)


def _tool_expand(arguments: dict[str, Any]) -> JsonValue:
    """Enclosing semantic unit for a file line; payload identical to `simgrep expand --format json`."""
    from pathlib import Path

    from simgrep.errors import ExpandError
    from simgrep.expand import cap_unit, read_text_raw, unit_bounds, unit_family
    from simgrep.files import infer_language
    from simgrep.text import compute_line_starts, offset_to_line

    path = Path(str(arguments["path"])).expanduser().resolve()
    family_override = arguments.get("language")
    if family_override is not None:
        family_override = str(family_override).lower()
        if family_override not in ("dedent", "brace", "paragraph"):
            raise ExpandError(
                f"Unknown language family: {family_override}.",
                exit_code=2,
                hint="language accepts only 'dedent', 'brace', or 'paragraph'.",
            )
    if not path.exists():
        raise ExpandError(f"Path not found: {path}", hint="Pass an existing PATH.", exit_code=2)
    text = read_text_raw(path)  # ExpandError(exit_code=1) on unreadable files
    line_starts = compute_line_starts(text)
    # Mirror compute_line_starts' strict-\n model: splitlines() also splits on
    # Unicode separators (\r, \v, ...) and would miscount lone-CR line endings.
    total_lines = len(line_starts) - (1 if text.endswith("\n") else 0)
    line = int(arguments["line"])
    if line < 1:
        raise ExpandError(f"line must be >= 1, got {line}.", exit_code=2, hint="Line numbers are 1-based.")
    if line > total_lines:
        raise ExpandError(f"line {line} out of range", hint=f"file has {total_lines} lines", exit_code=2)
    anchor = line_starts[line - 1]
    family = family_override if family_override is not None else unit_family(path)
    try:
        start, end = unit_bounds(text, anchor, family=family)
    except ValueError as exc:
        raise ExpandError(str(exc), hint=f"file has {total_lines} lines", exit_code=2) from exc
    start_line = offset_to_line(line_starts, start)
    end_line = offset_to_line(line_starts, max(end - 1, start))
    shown_start, shown_end = start, end
    truncated = False
    max_chars = arguments.get("max_chars")
    if max_chars is not None:
        shown_start, shown_end = cap_unit(text, start, end, max_chars=int(max_chars), anchor=anchor)
        truncated = shown_end < end
    unit_text = text[shown_start:shown_end]
    if truncated:
        unit_text += "..."
    payload = {
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
    return payload


def _tool_pack(arguments: dict[str, Any]) -> JsonValue:
    """Budgeted context assembly; payload identical to `simgrep pack --format json`."""
    from pathlib import Path

    from simgrep.errors import SearchError
    from simgrep.execution import execute_pack

    raw_queries = arguments.get("queries")
    if not isinstance(raw_queries, list):
        raise SearchError("queries must be an array of strings.", hint='Pass at least one query, e.g. ["auth flow"].')
    queries = [str(query) for query in raw_queries]
    if not queries or any(not query.strip() for query in queries):
        raise SearchError(
            "queries must contain at least one non-empty string.",
            hint='Pass at least one non-empty query, e.g. ["auth flow"].',
        )

    raw_path = arguments.get("path")
    path = Path(raw_path).expanduser().resolve() if raw_path else None
    budget = int(arguments.get("budget", 3000))
    per_query = int(arguments.get("per_query", 8))
    lam = float(arguments.get("lam", 0.7))

    from simgrep.records import pack_record

    packed = execute_pack(queries, path, budget=budget, lam=lam, per_query=per_query)
    if not packed.pool_size:
        raise SearchError(
            "No candidates found for the given queries.",
            hint="Broaden the queries or lower per-query filters.",
        )
    return pack_record(packed, queries)


def _tool_debt(arguments: dict[str, Any]) -> JsonValue:
    """Cluster TODO/FIXME/HACK markers into semantic themes; payload identical to `simgrep debt --format json`."""
    from pathlib import Path

    from simgrep.config import load_app_config
    from simgrep.debt_engine import DebtEngine
    from simgrep.errors import DebtError, SearchError
    from simgrep.execution import factory as resolve_factory
    from simgrep.models import DebtOptions
    from simgrep.project import find_active_project, project_covers_path

    app_config = load_app_config()
    raw_path = arguments.get("path")
    path = Path(str(raw_path)).expanduser().resolve() if raw_path else None
    raw_max_age = arguments.get("max_age_days")
    options = DebtOptions(
        threshold=float(arguments.get("threshold", 0.8)),
        min_size=int(arguments.get("min_size", 2)),
        top=int(arguments.get("top", 20)),
        max_members=int(arguments.get("max_members", 8)),
        max_age_days=float(raw_max_age) if raw_max_age is not None else None,
    )
    if not 0.01 <= options.threshold <= 1:
        raise DebtError(
            f"threshold must be a number between 0.01 and 1, got {options.threshold}.",
            exit_code=2,
            hint="Pass a cosine join threshold in (0, 1], e.g. 0.8.",
        )
    if options.min_size < 1:
        raise DebtError(
            f"min_size must be an int >= 1, got {options.min_size}.",
            exit_code=2,
            hint="Raise the markers-per-theme floor, e.g. 2.",
        )
    if not 1 <= options.top <= 200:
        raise DebtError(
            f"top must be an int between 1 and 200, got {options.top}.",
            exit_code=2,
            hint="Pass a theme cap from 1 to 200, e.g. 20.",
        )
    if not 1 <= options.max_members <= 50:
        raise DebtError(
            f"max_members must be an int between 1 and 50, got {options.max_members}.",
            exit_code=2,
            hint="Pass a per-theme match cap from 1 to 50, e.g. 8.",
        )
    if options.max_age_days is not None and options.max_age_days <= 0:
        raise DebtError(
            f"max_age_days must be a positive number, got {options.max_age_days}.",
            exit_code=2,
            hint="Pass the CI gate in days, e.g. 90.",
        )

    active = find_active_project(path or Path.cwd())
    use_ephemeral = False
    if path is not None:
        if active is None or not project_covers_path(active, path):
            use_ephemeral = True
    elif active is None:
        raise SearchError("No active project found.", hint="Run `simgrep init` or pass a path.")

    factory = resolve_factory()
    engine_runtime = factory.for_app(app_config) if use_ephemeral else _project_runtime(factory, active, app_config)
    engine = DebtEngine(engine_runtime)
    if use_ephemeral:
        assert path is not None
        report = engine.debt_path(path, app_config, options)
    else:
        assert active is not None
        report = engine.debt_project(active, app_config, options, freshness=app_config.freshness)

    from simgrep.records import debt_record

    return debt_record(report)


_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query text."},
        "expr": {
            "type": "string",
            "description": 'Boolean semantic expression (UPPERCASE AND/OR/NOT, quotes for phrases). Send with query:"" when using expr.',
        },
        "path": {"type": "string", "description": "Optional path scope; ephemeral search unless covered by the active project."},
        "top": {"type": "integer", "minimum": 1, "description": "Number of results (default 5)."},
        "min_score": {"type": "number", "minimum": 0, "maximum": 1, "description": "Minimum score."},
        "scope": {"type": "string", "description": "Restrict persistent results to this path."},
        "file_filter": {"type": "string", "description": "Filter result file glob."},
        "include": {"type": "array", "items": {"type": "string"}, "description": "Include path globs."},
        "exclude": {"type": "array", "items": {"type": "string"}, "description": "Exclude path globs."},
        "diversity": {"type": "string", "enum": ["none", "window", "file", "package"], "description": "Result deduplication mode (default window)."},
        "show_why": {"type": "boolean", "description": "Include ranking explanations."},
        "whole_unit": {"type": "boolean", "description": "Expand every hit to its enclosing semantic unit (function/class/block/paragraph) before rendering."},
        "persistent": {"type": "boolean", "description": "Require persistent project search."},
    },
    "required": ["query"],
}

_SIMILAR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "source": {"type": "string", "description": "Like anchor: '@file', 'path:start-end', or literal text ('-' stdin unsupported over MCP)."},
        "unlike": {"type": "string", "description": "Unlike anchor; same forms minus stdin."},
        "unlike_weight": {"type": "number", "minimum": 0, "maximum": 1, "description": "Contrastive weight lambda (default 0.5)."},
        "include_self": {"type": "boolean", "description": "Keep chunks overlapping the anchor's own file span."},
        "top": {"type": "integer", "minimum": 1, "description": "Number of results (default 5)."},
        "min_score": {"type": "number", "minimum": 0, "maximum": 1, "description": "Minimum score."},
    },
    "required": ["source"],
}


_CLUSTERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Optional path; ephemeral clustering unless covered by the active project."},
        "threshold": {"type": "number", "minimum": 0, "maximum": 1, "description": "Minimum pairwise similarity (default 0.8)."},
        "min_size": {"type": "integer", "minimum": 2, "description": "Smallest reported cluster size (default 2)."},
        "top": {"type": "integer", "minimum": 1, "description": "Cap on returned clusters (default 20)."},
        "same_file": {"type": "boolean", "description": "Include duplicates within a single file (default false)."},
    },
}


DIFF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "a": {"type": "string", "description": "Old tree path."},
        "b": {"type": "string", "description": "New tree path."},
        "threshold": {"type": "number", "exclusiveMinimum": 0, "maximum": 1, "description": "Match threshold (default 0.8)."},
        "top": {"type": "integer", "minimum": 1, "description": "Max listed added/removed chunks (default 50)."},
        "max_chunks": {
            "type": "integer",
            "minimum": 1,
            "description": "Guard for total chunks across both trees (default 50000).",
        },
    },
    "required": ["a", "b"],
    "additionalProperties": False,
}


_EXPAND_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "File to expand; must exist."},
        "line": {"type": "integer", "minimum": 1, "description": "1-based line inside the desired unit."},
        "max_chars": {"type": "integer", "minimum": 200, "maximum": 200000, "description": "Character budget for the returned text (default: no cap)."},
        "language": {"type": "string", "enum": ["dedent", "brace", "paragraph"], "description": "Override family detection (default: by file suffix)."},
    },
    "required": ["path", "line"],
}


_PACK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "description": "One or more queries; their top hits form the deduplicated packing pool (1+).",
        },
        "path": {"type": "string", "description": "Optional path scope; ephemeral packing unless covered by the active project."},
        "budget": {"type": "integer", "minimum": 100, "maximum": 200000, "description": "Token budget for the assembled block (default 3000)."},
        "per_query": {"type": "integer", "minimum": 1, "maximum": 50, "description": "Pool size per query (default 8)."},
        "lam": {"type": "number", "minimum": 0, "maximum": 1, "description": "Relevance vs diversity weight for MMR selection (default 0.7)."},
    },
    "required": ["queries"],
}

_DEBT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Optional project directory; ephemeral corpus unless covered by the active project."},
        "threshold": {"type": "number", "minimum": 0.01, "maximum": 1, "description": "Cosine threshold joining marker chunks into themes (default 0.8)."},
        "min_size": {"type": "integer", "minimum": 1, "description": "Markers-per-theme floor; smaller components stay scattered (default 2)."},
        "top": {"type": "integer", "minimum": 1, "maximum": 200, "description": "Max ranked themes (default 20)."},
        "max_members": {"type": "integer", "minimum": 1, "maximum": 50, "description": "Max matches listed per theme (default 8)."},
        "max_age_days": {
            "type": "number",
            "minimum": 0.01,
            "description": "CI gate in days: `passed` reports whether every dated theme is younger than this (default: no gate).",
        },
    },
}

_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}

TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="search",
        description="Semantic grep over local files. Returns a JSON array of scored result records.",
        input_schema=_SEARCH_SCHEMA,
        handler=_tool_search,
    ),
    ToolSpec(
        name="similar",
        description="Query-by-example: find chunks like a source anchor, optionally demoting an 'unlike' anchor. Same JSON array shape as search.",
        input_schema=_SIMILAR_SCHEMA,
        handler=_tool_similar,
    ),
    ToolSpec(
        name="clusters",
        description="Semantic duplicate detection: returns a JSON array of clone clusters with score, duplicated_lines, and members.",
        input_schema=_CLUSTERS_SCHEMA,
        handler=_tool_clusters,
    ),
    ToolSpec(
        name="status",
        description="Report the active project's index state: project_root, indexed_paths, files, chunks, index_state, model.",
        input_schema=_EMPTY_SCHEMA,
        handler=_tool_status,
    ),
    ToolSpec(
        name="index",
        description="Incrementally index the active project (same flow as `simgrep index`). Note: the first call may load the embedding model and be slow.",
        input_schema=_EMPTY_SCHEMA,
        handler=_tool_index,
    ),
    ToolSpec(
        name="diff",
        description="Semantic tree comparison of two paths with rename insensitivity:"
        " returns a JSON object with matched/added/removed chunk entries and per-file rollups.",
        input_schema=DIFF_SCHEMA,
        handler=_tool_diff,
    ),
    ToolSpec(
        name="expand",
        description="Return the enclosing semantic unit (function/class/block/paragraph) for a" " file line. Lexical, deterministic, no index needed.",
        input_schema=_EXPAND_SCHEMA,
        handler=_tool_expand,
    ),
    ToolSpec(
        name="pack",
        description="Budgeted context assembly: run one or more queries, deduplicate their hits,"
        " and greedily select a paste-ready, budget-fitting block via MMR."
        " Returns the same JSON payload as `simgrep pack --format json`.",
        input_schema=_PACK_SCHEMA,
        handler=_tool_pack,
    ),
    ToolSpec(
        name="debt",
        description="Semantic debt-marker radar: cluster TODO/FIXME/HACK markers into themes with git ages,"
        " ranked by size and age, with an optional max-age CI gate."
        " Returns the same JSON payload as `simgrep debt --format json`.",
        input_schema=_DEBT_SCHEMA,
        handler=_tool_debt,
    ),
)
TOOLS_BY_NAME: dict[str, ToolSpec] = {tool.name: tool for tool in TOOLS}
