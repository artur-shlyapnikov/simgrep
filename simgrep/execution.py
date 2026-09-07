"""Transport-neutral application layer.

One copy of the policy that CLI and MCP used to duplicate independently: active-project
resolution, persistent-vs-ephemeral mode selection, runtime selection, freshness defaults,
daemon offload and post-search transforms. Transports parse input and render output; these
functions decide how a command runs. Error text stays byte-identical to the historical CLI
messages so exit-code contracts are unchanged.

The corpus boundary is :func:`open_resolved_corpus`: after it yields a
:class:`ResolvedCorpus`, feature code knows nothing about active-project lookup, coverage
fallback, runtime selection, or freshness — every ``execute_*`` dispatcher routes scope,
runtime and freshness decisions through it.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

from simgrep.models import (
    AppConfig,
    ClustersOptions,
    ClustersOutcome,
    EphemeralIndexOptions,
    FreshnessMode,
    ProjectConfig,
    ScanOptions,
    SearchOptions,
    SearchOutcome,
    SimilarOptions,
)
from simgrep.pack import PackOutcome

if TYPE_CHECKING:
    from simgrep.runtime import RuntimeFactory


def runtime_for_project(factory: Any, project: ProjectConfig, app_config: AppConfig) -> Any:
    """Reuse the app runtime when project model/chunk settings match the global config."""
    if project.model == app_config.model and project.chunk_size == app_config.chunk_size and project.chunk_overlap == app_config.chunk_overlap:
        return factory.for_app(app_config)
    return factory.for_project(project)


@dataclass(frozen=True)
class RerankRequest:
    """Cross-encoder rerank parameters (transport flag surface)."""

    top: int
    model: str


@dataclass(frozen=True)
class _ScopeErrors:
    """Per-command exception factories preserving exact historical messages."""

    missing_project: Callable[[], Exception]
    needs_coverage: Callable[[], Exception]
    needs_active: Callable[[], Exception]


@dataclass(frozen=True)
class _Scope:
    project: ProjectConfig | None
    ephemeral_path: Path | None


def _resolve_scope(
    *,
    app_config: AppConfig,
    path: Path | None,
    ephemeral: bool,
    persistent: bool,
    errors: _ScopeErrors,
    allow_missing_path: bool,
    cwd_fallback: bool = False,
) -> _Scope:
    """Active-project lookup -> coverage -> persistent-vs-ephemeral decision."""
    from simgrep.project import find_active_project, project_covers_path

    del app_config  # resolution only needs paths today; kept for symmetric call sites
    active = find_active_project(path or Path.cwd())
    use_ephemeral = bool(ephemeral)
    if path is not None and not use_ephemeral:
        if active is None or not project_covers_path(active, path):
            if persistent:
                raise errors.needs_coverage()
            use_ephemeral = True
    if path is None and active is None:
        # Historical order: the missing-project message wins over the persistent one.
        if cwd_fallback:
            return _Scope(project=None, ephemeral_path=Path.cwd())
        if not allow_missing_path:
            raise errors.needs_active()
        raise errors.missing_project()
    if persistent and active is None:
        raise errors.needs_active()
    if use_ephemeral:
        return _Scope(project=None, ephemeral_path=path if path is not None else Path.cwd())
    assert active is not None
    return _Scope(project=active, ephemeral_path=None)


def _scan_for(app_config: AppConfig, patterns: tuple[str, ...] | None, include_globs: tuple[str, ...], exclude_globs: tuple[str, ...]) -> EphemeralIndexOptions:
    return EphemeralIndexOptions(
        scan=ScanOptions(
            patterns=tuple(patterns or app_config.file_patterns),
            include_globs=tuple(include_globs),
            exclude_globs=tuple(exclude_globs),
            max_file_size_bytes=app_config.max_file_size_bytes,
            follow_symlinks=app_config.follow_symlinks,
        )
    )


@dataclass(frozen=True)
class CorpusRequest:
    """Which corpus a command wants: a path scope plus mode flags."""

    path: Path | None
    ephemeral: bool = False
    persistent: bool = False


@dataclass(frozen=True)
class ResolvedCorpus:
    """An opened corpus plus the base path to project display paths against."""

    reader: Any  # simgrep.corpus.CorpusReader
    base_path: Path
    project: ProjectConfig | None


@contextmanager
def open_resolved_corpus(
    request: CorpusRequest,
    app_config: AppConfig,
    *,
    errors: _ScopeErrors,
    freshness: FreshnessMode | None = None,
    patterns: tuple[str, ...] | None = None,
    include_globs: tuple[str, ...] = (),
    exclude_globs: tuple[str, ...] = (),
    allow_missing_path: bool = True,
    cwd_fallback: bool = False,
) -> Iterator[ResolvedCorpus]:
    """Resolve and open the corpus for ``request``: the single owner of scope,
    runtime selection and freshness policy. Feature engines consume the yielded
    :class:`ResolvedCorpus` and never see project lookup or fallback rules."""
    from simgrep.corpus import CorpusAccess

    scope = _resolve_scope(
        app_config=app_config,
        path=request.path,
        ephemeral=request.ephemeral,
        persistent=request.persistent,
        errors=errors,
        allow_missing_path=allow_missing_path,
        cwd_fallback=cwd_fallback,
    )
    runtime_factory = factory()
    if scope.project is not None:
        runtime = runtime_for_project(runtime_factory, scope.project, app_config)
        with CorpusAccess(runtime).open_project(scope.project, app_config, freshness=freshness or app_config.freshness) as reader:
            yield ResolvedCorpus(reader=reader, base_path=scope.project.root, project=scope.project)
    else:
        assert scope.ephemeral_path is not None
        runtime = runtime_factory.for_app(app_config)
        scan = _scan_for(app_config, patterns, include_globs, exclude_globs)
        with CorpusAccess(runtime).open_ephemeral([scope.ephemeral_path], app_config, scan) as reader:
            yield ResolvedCorpus(reader=reader, base_path=scope.ephemeral_path, project=None)


# Canonical test seam: ``monkeypatch.setattr("simgrep.execution.RuntimeFactory", ...)``.
# Resolved lazily (PEP 562) so importing execution never pulls the heavy model stack.
_LAZY_IMPORTS = {"RuntimeFactory": "simgrep.runtime"}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def factory() -> RuntimeFactory:
    """Transport-neutral RuntimeFactory resolution (see module test seam above)."""
    if "RuntimeFactory" in globals():
        patched: Any = globals()["RuntimeFactory"]
        result: RuntimeFactory = patched()
        return result
    from simgrep.runtime import RuntimeFactory

    return RuntimeFactory()


def _search_error(message: str, hint: str | None) -> Exception:
    from simgrep.errors import SearchError

    return SearchError(message, hint=hint) if hint else SearchError(message)


def _search_scope_errors() -> _ScopeErrors:
    return _ScopeErrors(
        missing_project=lambda: _search_error("No active project found.", "Run `simgrep init` or pass a PATH for ephemeral search."),
        needs_coverage=lambda: _search_error("Persistent search requires an active project covering the requested path.", None),
        needs_active=lambda: _search_error("Persistent search requires an active project.", None),
    )


def _scan_scope_errors(noun: str, *, target_word: str = "TARGET_DIR", activity: str = "scan") -> _ScopeErrors:
    """Error family shared by the analytics commands ("ephemeral <activity>" wording)."""

    def missing() -> Exception:
        return _search_error("No active project found.", f"Run `simgrep init` or pass a {target_word} for ephemeral {activity}.")

    def needs_coverage() -> Exception:
        return _search_error(f"Persistent {noun} requires an active project covering the requested path.", None)

    def needs_active() -> Exception:
        return _search_error(f"Persistent {noun} requires an active project.", None)

    return _ScopeErrors(missing_project=missing, needs_coverage=needs_coverage, needs_active=needs_active)


def replace_outcome_results(outcome: SearchOutcome, results: list[Any]) -> SearchOutcome:
    from dataclasses import replace

    return replace(outcome, results=results)


def execute_search(
    *,
    app_config: AppConfig,
    path: Path | None,
    options: SearchOptions,
    freshness: FreshnessMode | None = None,
    ephemeral: bool = False,
    persistent: bool = False,
    patterns: tuple[str, ...] | None = None,
    include_globs: tuple[str, ...] = (),
    exclude_globs: tuple[str, ...] = (),
    whole_unit: bool = False,
    rerank: RerankRequest | None = None,
) -> SearchOutcome:
    """Search an active covering project (daemon-first when fresh-auto) or an ephemeral corpus."""
    from dataclasses import replace as dc_replace

    from simgrep.search import SearchEngine

    if rerank is not None:
        # Torch-first libomp order: the fallback search below may construct a
        # USearchIndex, and the reranker branch imports torch only afterwards
        # (CrossEncoderReranker -> sentence_transformers). Flag it now so the
        # usearch guard loads torch first if torch is still pending; flag only,
        # nothing is loaded here and daemon-served flows are unaffected.
        from simgrep.adapters import vector

        vector.mark_torch_pending()
    scope = _resolve_scope(
        app_config=app_config,
        path=path,
        ephemeral=ephemeral,
        persistent=persistent,
        errors=_search_scope_errors(),
        allow_missing_path=True,
    )
    if scope.project is not None and options.scope_path is None and path is not None:
        options = dc_replace(options, scope_path=path.resolve())
    effective_freshness = freshness or app_config.freshness
    selected_runtime: Any = None
    outcome: SearchOutcome | None = None
    if scope.project is not None and effective_freshness is FreshnessMode.auto:
        from simgrep import daemon  # Local: served queries skip heavy engine imports entirely.

        outcome = daemon.try_search(scope.project, app_config, options, effective_freshness)
    if outcome is None:
        runtime_factory = factory()
        if scope.project is not None:
            selected_runtime = runtime_for_project(runtime_factory, scope.project, app_config)
            outcome = SearchEngine(selected_runtime).search_project(scope.project, app_config, options, effective_freshness)
        else:
            assert scope.ephemeral_path is not None
            selected_runtime = runtime_factory.for_app(app_config)
            outcome = SearchEngine(selected_runtime).search_path(
                scope.ephemeral_path, app_config, options, _scan_for(app_config, patterns, include_globs, exclude_globs)
            )
    if whole_unit:
        from simgrep.expand import expand_results, read_text_raw

        outcome.results[:] = expand_results(outcome.results, fetch=read_text_raw)
    if rerank is not None:
        from simgrep.adapters.reranker import CrossEncoderReranker
        from simgrep.rerank import rerank_results

        resolved_reranker = getattr(selected_runtime, "reranker", None)
        if resolved_reranker is None:
            resolved_reranker = CrossEncoderReranker(rerank.model)
        outcome = replace_outcome_results(outcome, rerank_results(outcome.results, options.query, resolved_reranker.score, rerank.top).results)
    return outcome


def execute_similar(
    *,
    app_config: AppConfig,
    target_dir: Path | None,
    options: SimilarOptions,
    freshness: FreshnessMode | None = None,
    ephemeral: bool = False,
    persistent: bool = False,
    include_globs: tuple[str, ...] = (),
    exclude_globs: tuple[str, ...] = (),
) -> SearchOutcome:
    """Similar-file search mirroring ``execute_search``'s scope policy."""
    from simgrep.search import SearchEngine

    def missing() -> Exception:
        return _search_error("No active project found.", "Run `simgrep init` or pass a TARGET_DIR for ephemeral search.")

    base_errors = _search_scope_errors()
    scope = _resolve_scope(
        app_config=app_config,
        path=target_dir,
        ephemeral=ephemeral,
        persistent=persistent,
        errors=_ScopeErrors(missing_project=missing, needs_coverage=base_errors.needs_coverage, needs_active=base_errors.needs_active),
        allow_missing_path=True,
    )
    runtime_factory = factory()
    if scope.project is not None:
        runtime = runtime_for_project(runtime_factory, scope.project, app_config)
        return SearchEngine(runtime).similar_project(scope.project, app_config, options, freshness or app_config.freshness)
    assert scope.ephemeral_path is not None
    runtime = runtime_factory.for_app(app_config)
    return SearchEngine(runtime).similar_path(scope.ephemeral_path, app_config, options, _scan_for(app_config, None, include_globs, exclude_globs))


def execute_clusters(
    *,
    app_config: AppConfig,
    path: Path | None,
    options: ClustersOptions,
    freshness: FreshnessMode | None = None,
    ephemeral: bool = False,
    persistent: bool = False,
) -> tuple[ClustersOutcome, Path]:
    """Duplicate-chunk clusters plus the base path renderers should relativize against."""
    from simgrep.clusters_engine import ClustersEngine

    runtime = factory().for_app(app_config)

    with open_resolved_corpus(
        CorpusRequest(path=path, ephemeral=ephemeral, persistent=persistent),
        app_config,
        freshness=freshness,
        errors=_scan_scope_errors("clustering", target_word="PATH", activity="clustering"),
    ) as corpus:
        return ClustersEngine(runtime).run_batch(corpus.reader.snapshot(), options), corpus.base_path


def execute_pack(
    queries: list[str],
    target_dir: Path | None,
    *,
    budget: int,
    lam: float,
    per_query: int,
    ephemeral: bool = False,
    persistent: bool = False,
) -> PackOutcome:
    """Per-query searches -> label-dedup union -> token-budgeted candidate packing."""
    from simgrep.config import load_app_config
    from simgrep.models import SearchResult
    from simgrep.pack import PackCandidate, pack_candidates
    from simgrep.search import SearchEngine

    app_config = load_app_config()
    if not queries or any(not query.strip() for query in queries):
        raise _search_error(
            "queries must contain at least one non-empty string.",
            'Pass at least one non-empty query, e.g. ["auth flow"].',
        )

    def missing() -> Exception:
        return _search_error("No active project found.", "Run `simgrep init` or pass a TARGET_DIR for ephemeral search.")

    scope = _resolve_scope(
        app_config=app_config,
        path=target_dir,
        ephemeral=ephemeral,
        persistent=persistent,
        errors=_ScopeErrors(
            missing_project=missing, needs_coverage=lambda: _search_scope_errors().needs_coverage(), needs_active=lambda: _search_scope_errors().needs_active()
        ),
        allow_missing_path=True,
    )
    merged: dict[int, SearchResult] = {}
    base_path = target_dir if target_dir is not None else Path.cwd()
    for query in queries:
        options = SearchOptions(
            query=query,
            top=per_query,
            lexical_top=app_config.lexical_top,
            lexical_weight=app_config.lexical_weight,
        )
        if scope.project is not None:
            runtime = runtime_for_project(factory(), scope.project, app_config)
            outcome = SearchEngine(runtime).search_project(scope.project, app_config, options, app_config.freshness)
        else:
            assert scope.ephemeral_path is not None
            runtime = factory().for_app(app_config)
            # Historical pack scan omits glob filters on both transports.
            scan = EphemeralIndexOptions(
                scan=ScanOptions(
                    patterns=tuple(app_config.file_patterns),
                    max_file_size_bytes=app_config.max_file_size_bytes,
                    follow_symlinks=app_config.follow_symlinks,
                )
            )
            outcome = SearchEngine(runtime).search_path(scope.ephemeral_path, app_config, options, scan)
        for result in outcome.results:
            known = merged.get(result.label)
            if known is None or result.score > known.score:
                merged[result.label] = result
        base_path = outcome.base_path
    pool: list[PackCandidate] = []
    for label, result in merged.items():
        display_path: Path = result.file_path
        try:
            display_path = result.file_path.relative_to(base_path)
        except ValueError:
            pass
        pool.append(
            PackCandidate(
                label=label,
                path=str(display_path),
                line_start=result.line_start or 1,
                line_end=result.line_end or result.line_start or 1,
                score=result.score,
                text=result.chunk_text,
            )
        )
    return pack_candidates(pool, budget, lam=lam)


def execute_debt(
    *,
    app_config: AppConfig,
    target_dir: Path | None,
    options: Any,
    ephemeral: bool = False,
    persistent: bool = False,
) -> Any:
    """Debt-marker radar over the resolved corpus."""
    from simgrep.debt_engine import DebtEngine

    runtime = factory().for_app(app_config)

    with open_resolved_corpus(
        CorpusRequest(path=target_dir, ephemeral=ephemeral, persistent=persistent),
        app_config,
        errors=_scan_scope_errors("debt scan"),
    ) as corpus:
        return DebtEngine(runtime).run_batch(corpus.reader.snapshot(), corpus.base_path, options)
