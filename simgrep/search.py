from __future__ import annotations

import dataclasses
import os
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np

from simgrep import daemon as daemon
from simgrep.corpus import CorpusAccess, CorpusReader, StoredChunk
from simgrep.errors import SearchError
from simgrep.models import (
    Anchor,
    AppConfig,
    EphemeralIndexOptions,
    FreshnessMode,
    ProjectConfig,
    ResultFilters,
    SearchOptions,
    SearchOutcome,
    SearchResult,
    SimilarOptions,
)
from simgrep.query_expr import collect_leaves, evaluate, parse
from simgrep.ranking import _normalize_semantic, rank_candidates
from simgrep.text import compute_line_starts


class SearchEngine:
    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.access = CorpusAccess(runtime)

    def search_path(
        self,
        path: Path,
        app_config: AppConfig,
        options: SearchOptions,
        scan_options: EphemeralIndexOptions | None = None,
    ) -> SearchOutcome:
        with self.access.open_ephemeral([path], app_config, scan_options or EphemeralIndexOptions()) as reader:
            results, semantic_count = self.search_reader(reader, options)
            counts = reader.counts()
            return SearchOutcome(
                results=results,
                base_path=reader.base_path,
                files_seen=counts.files_count,
                chunks_searched=counts.chunks_count,
                semantic_candidates=semantic_count,
            )

    def search_project(self, project: ProjectConfig, app_config: AppConfig, options: SearchOptions, freshness: FreshnessMode) -> SearchOutcome:
        # Daemon offload is owned by simgrep.execution.execute_search; the engine
        # always serves locally so the daemon itself can reuse this path.

        with self.access.open_project(project, app_config, freshness=freshness) as reader:
            results, semantic_count = self.search_reader(reader, options)
            counts = reader.counts(project.name)
            return SearchOutcome(
                results=results,
                base_path=project.root,
                files_seen=counts.files_count,
                chunks_searched=counts.chunks_count,
                semantic_candidates=semantic_count,
            )

    def similar_path(
        self,
        path: Path,
        app_config: AppConfig,
        options: SimilarOptions,
        scan_options: EphemeralIndexOptions | None = None,
    ) -> SearchOutcome:
        with self.access.open_ephemeral([path], app_config, scan_options or EphemeralIndexOptions()) as reader:
            results, semantic_count = self._similar_reader(reader, options)
            counts = reader.counts()
            return SearchOutcome(
                results=results,
                base_path=reader.base_path,
                files_seen=counts.files_count,
                chunks_searched=counts.chunks_count,
                semantic_candidates=semantic_count,
            )

    def similar_project(self, project: ProjectConfig, app_config: AppConfig, options: SimilarOptions, freshness: FreshnessMode) -> SearchOutcome:
        with self.access.open_project(project, app_config, freshness=freshness) as reader:
            results, semantic_count = self._similar_reader(reader, options)
            counts = reader.counts(project.name)
            return SearchOutcome(
                results=results,
                base_path=project.root,
                files_seen=counts.files_count,
                chunks_searched=counts.chunks_count,
                semantic_candidates=semantic_count,
            )

    def search_reader(self, reader: CorpusReader, options: SearchOptions) -> tuple[list[SearchResult], int]:
        """Canonical search pipeline over an open reader: the one implementation
        shared by in-process and daemon-resident corpora."""
        if options.expr is not None:
            return self._search_expr_reader(reader, options)
        semantic_pairs: list[tuple[int, float]] = []
        if reader.chunk_count > 0:
            semantic_pairs = [(int(hit.label), float(hit.score)) for hit in self._search_vectors(reader, options.query, effective_candidate_top(options))]
        filters = ResultFilters.from_search_options(options)
        semantic_rows = reader.lookup([label for label, _ in semantic_pairs], filters)
        lexical_rows: list[tuple[StoredChunk, float]] = []
        if options.lexical_top > 0 and options.lexical_weight > 0:
            lexical_rows = reader.lexical(tokenize_query(options.query), options.lexical_top, filters)
        return (
            rank_candidates(query=options.query, semantic_matches=semantic_pairs, semantic_rows=semantic_rows, lexical_rows=lexical_rows, options=options),
            len(semantic_pairs),
        )

    def _search_expr_reader(self, reader: CorpusReader, options: SearchOptions) -> tuple[list[SearchResult], int]:
        text = options.expr
        assert text is not None  # guaranteed by the _search_reader branch
        expr = parse(text)
        k = effective_candidate_top(options)
        leaves = list(collect_leaves(expr))
        # One batched encode for all leaves instead of one model round-trip per leaf.
        vectors = np.asarray(getattr(self.runtime, "query_embedder", self.runtime.embedder).encode(leaves, is_query=True))
        leaf_scores = {leaf: {int(h.label): _normalize_semantic(float(h.score)) for h in reader.search(vectors[row], k=k)} for row, leaf in enumerate(leaves)}
        return expr_results(reader, text, leaf_scores, options)

    def _similar_reader(self, reader: CorpusReader, options: SimilarOptions) -> tuple[list[SearchResult], int]:
        search_options = options.search
        k = effective_candidate_top(search_options)
        semantic_pairs: list[tuple[int, float]] = []
        contrast_unlike: dict[int, float] | None = None
        contrast_like: dict[int, float] | None = None
        if reader.chunk_count > 0:
            like_hits = self._search_vectors(reader, options.anchor.text, k)
            like_scores = {(int(hit.label)): float(hit.score) for hit in like_hits}
            semantic_pairs = list(like_scores.items())
            if options.unlike is not None:
                unlike_scores = {(int(hit.label)): float(hit.score) for hit in self._search_vectors(reader, options.unlike.text, k)}
                semantic_pairs = combine_candidate_scores(like_scores, unlike_scores, options.unlike_weight)
                contrast_unlike = {label: options.unlike_weight * unlike_scores.get(label, 0.0) for label, _ in semantic_pairs}
                contrast_like = {label: like_scores.get(label, 0.0) for label, _ in semantic_pairs}
        filters = ResultFilters.from_search_options(search_options)
        semantic_rows = reader.lookup([label for label, _ in semantic_pairs], filters)
        lexical_rows: list[tuple[StoredChunk, float]] = []
        if search_options.lexical_top > 0 and search_options.lexical_weight > 0:
            lexical_rows = reader.lexical(tokenize_query(options.anchor.text), search_options.lexical_top, filters)
        if options.anchor.has_span:
            assert options.anchor.origin is not None and options.anchor.start_char is not None and options.anchor.end_char is not None
            overlaps_anchor = self_match_predicate(
                origin=options.anchor.origin,
                anchor_start=options.anchor.start_char,
                anchor_end=options.anchor.end_char,
                include_self=options.include_self,
                base_path=reader.base_path,
            )
            semantic_rows = [row for row in semantic_rows if overlaps_anchor(row)]
            lexical_rows = [pair for pair in lexical_rows if overlaps_anchor(pair[0])]
        results = rank_candidates(
            query=options.anchor.text,
            semantic_matches=semantic_pairs,
            semantic_rows=semantic_rows,
            lexical_rows=lexical_rows,
            options=search_options,
            contrast_unlike=contrast_unlike,
            contrast_like=contrast_like,
        )
        return results, len(semantic_pairs)

    def _search_vectors(self, reader: CorpusReader, query: str, k: int) -> list[Any]:
        query_vec = np.asarray(getattr(self.runtime, "query_embedder", self.runtime.embedder).encode([query], is_query=True))
        hits: list[Any] = reader.search(query_vec[0], k=k)
        return hits


LINE_RANGE_RE = re.compile(r"^(.+):(\d+)(?:-(\d+))?$")


def resolve_anchor(source: str, *, stdin_text: str | None = None) -> Anchor:
    """Resolve a SOURCE argument to an anchor, honoring the spec precedence order."""
    if source == "-":
        if stdin_text is None:
            raise SearchError("Anchor '-' requires piped stdin.", hint="Pipe the anchor text into the command.")
        return _anchor_from_text(stdin_text)
    if source.startswith("@"):
        return _anchor_from_spec(source[1:])
    match = LINE_RANGE_RE.match(source)
    if match is not None:
        candidate = Path(match.group(1))
        if candidate.is_file():
            end_line = int(match.group(3)) if match.group(3) is not None else int(match.group(2))
            return _anchor_from_file(candidate, (int(match.group(2)), end_line))
        if _looks_like_file_path(match.group(1)):
            raise SearchError(
                f"Anchor file not found: {match.group(1)}",
                hint="Anchor 'path:line[-end]' needs an existing file, e.g. greet.py:6 or greet.py:6-9.",
            )
    return _anchor_from_text(source)


def _looks_like_file_path(text: str) -> bool:
    """Heuristic: file-like anchors have no whitespace plus a separator or extension."""
    if not text or any(char.isspace() for char in text):
        return False
    return "/" in text or "." in text


def _anchor_from_spec(spec: str) -> Anchor:
    """Resolve an ``@``-prefixed spec: whole file, or ``file:line[-end]`` span."""
    match = LINE_RANGE_RE.match(spec)
    if match is not None and Path(match.group(1)).is_file():
        end_line = int(match.group(3)) if match.group(3) is not None else int(match.group(2))
        return _anchor_from_file(Path(match.group(1)), (int(match.group(2)), end_line))
    return _anchor_from_file(Path(spec), None)


def _anchor_from_text(text: str) -> Anchor:
    if not text.strip():
        raise SearchError("Anchor text is empty.", hint="Provide non-empty anchor text.")
    return Anchor(text=text)


def _anchor_from_file(path: Path, line_range: tuple[int, int] | None) -> Anchor:
    if not path.is_file():
        raise SearchError(
            f"Anchor file not found: {path}",
            hint="Pass an existing file; use 'path', 'path:line[-end]', or '@path'.",
        )
    text = _read_anchor_text(path)
    if line_range is None:
        anchor_text = text
        start_char, end_char = 0, len(text)
    else:
        start_line, end_line = line_range
        if start_line < 1 or start_line > end_line:
            raise SearchError(f"Invalid anchor line range: {path}:{start_line}-{end_line}", hint="Use 1-based start <= end.")
        total_lines = len(text.splitlines())
        if start_line > total_lines:
            raise SearchError(f"Anchor start line beyond end of file: {path}:{start_line}-{end_line}")
        line_starts = compute_line_starts(text)
        clamped_end = min(end_line, total_lines)
        start_char = line_starts[start_line - 1]
        end_char = line_starts[clamped_end] if clamped_end < len(line_starts) else len(text)
        anchor_text = text[start_char:end_char]
    if not anchor_text.strip():
        raise SearchError(f"Anchor text is empty: {path}")
    return Anchor(text=anchor_text, origin=path.absolute(), start_char=start_char, end_char=end_char)


def _read_anchor_text(path: Path) -> str:
    """Decode anchor bytes exactly like the indexer's extractor (utf-8-sig BOM
    strip, no universal-newline translation) so char spans reconcile with stored
    chunk offsets. Non-UTF-8 files surface a user-facing error instead of a raw
    traceback."""
    try:
        return path.read_bytes().decode("utf-8-sig")
    except (UnicodeDecodeError, OSError) as exc:
        raise SearchError(f"Anchor file is not valid UTF-8 text: {path}", hint="Re-save the anchor file as UTF-8.") from exc


def combine_candidate_scores(like: dict[int, float], unlike: dict[int, float], weight: float) -> list[tuple[int, float]]:
    """Union of candidate labels scored s_like - weight * s_unlike, sorted by score
    descending (label ascending as tiebreak); missing side contributes 0."""
    combined: dict[int, float] = {}
    for label, score in like.items():
        combined[label] = score - weight * unlike.get(label, 0.0)
    for label, score in unlike.items():
        if label not in combined:
            combined[label] = -weight * score
    return sorted(combined.items(), key=lambda item: (-item[1], item[0]))


def spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def self_match_predicate(
    *,
    origin: Path,
    anchor_start: int,
    anchor_end: int,
    include_self: bool,
    base_path: Path | None = None,
) -> Callable[[StoredChunk], bool]:
    """Build a predicate that rejects chunks overlapping the anchor span in the
    anchor's own file. Applied before ranking so remaining candidates backfill --top."""
    if include_self:
        return lambda _chunk: True

    def _keep(chunk: StoredChunk) -> bool:
        candidate = Path(chunk.file_path)
        if not candidate.is_absolute() and base_path is not None:
            candidate = base_path / candidate
        if _same_file(candidate, origin) and spans_overlap(chunk.start_char, chunk.end_char, anchor_start, anchor_end):
            return False
        return True

    return _keep


def filter_self_matches(
    rows: list[StoredChunk],
    origin: Path,
    anchor_start: int,
    anchor_end: int,
    *,
    include_self: bool,
    base_path: Path | None = None,
) -> list[StoredChunk]:
    """Drop candidate chunks overlapping the anchor span in the anchor's own file."""
    keep = self_match_predicate(origin=origin, anchor_start=anchor_start, anchor_end=anchor_end, include_self=include_self, base_path=base_path)
    return [row for row in rows if keep(row)]


def _same_file(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except OSError:
        return os.path.normpath(str(left)) == os.path.normpath(str(right))


def expr_results(
    reader: CorpusReader,
    expr_text: str,
    leaf_scores: dict[str, dict[int, float]],
    options: SearchOptions,
) -> tuple[list[SearchResult], int]:
    """Evaluate ``expr_text`` over pre-built per-leaf score maps (public seam).

    Used by ``search --expr`` so dominance semantics live in
    exactly one place. ``leaf_scores`` maps each expression leaf to
    ``{chunk label: normalized semantic score}``.
    """
    combined = evaluate(parse(expr_text), leaf_scores)
    # DOMINANCE NOT, FILE GRANULARITY: combined == 0.0 marks a chunk the
    # negated leaf dominates (or that no leaf surfaces). Files are the
    # user's mental unit, so every chunk of a file containing a zeroed
    # chunk is dropped before ranking — a straddling sibling that would
    # survive on its own positive score must not resurface the excluded
    # concept (nor let rank_candidates re-normalize the zero to ~0.5).
    pairs = sorted(combined.items(), key=lambda item: (-item[1], item[0]))
    filters = ResultFilters.from_search_options(options)
    rows = reader.lookup([label for label, _ in pairs], filters)
    file_of = {row.label: row.file_path for row in rows}
    excluded_files = {file_of[label] for label, score in combined.items() if score == 0.0 and label in file_of}
    kept_pairs = [(label, score) for label, score in pairs if file_of.get(label) not in excluded_files]
    kept_labels = {label for label, _ in kept_pairs}
    pure = dataclasses.replace(options, lexical_weight=0.0)
    results = rank_candidates(
        query=expr_text, semantic_matches=kept_pairs, semantic_rows=[row for row in rows if row.label in kept_labels], lexical_rows=[], options=pure
    )
    return results, len(kept_pairs)


def effective_candidate_top(options: SearchOptions) -> int:
    if options.candidate_top is not None:
        return max(options.candidate_top, options.top)
    base = max(options.top * 40, 200)
    if options.scope_path or options.file_filter or options.include_globs or options.exclude_globs:
        base = max(options.top * 120, 1000)
    return base


def tokenize_query(query: str) -> list[str]:
    if not query:
        return []
    split_acronyms = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", query)
    lowered = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", split_acronyms).lower()
    snake_kebab_split = re.sub(r"[_\-]+", " ", lowered)
    parts = re.split(r"[^a-z0-9]+", snake_kebab_split)
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if not part or len(part) == 1 or part in seen:
            continue
        seen.add(part)
        out.append(part)
        if len(out) >= 8:
            break
    return out
