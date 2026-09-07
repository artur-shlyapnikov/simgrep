from __future__ import annotations

import json
import time
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from simgrep.models import (
    ClusterMember,
    ClustersOutcome,
    DebtReport,
    DiffEntry,
    DiffOutcome,
    DisplaySearchResult,
    FileRollup,
    RenderOptions,
    RerankReport,
    ResultFormat,
    SearchResult,
    SemanticCluster,
)
from simgrep.pack import PackOutcome
from simgrep.records import (
    _debt_theme_record,
    _rerank_match_record,
    cluster_display_path,
    cluster_member_display,
    cluster_record,
    debt_record,
    display_record,
    pack_record,
    rerank_record,
)
from simgrep.text import compute_line_starts, offset_to_line


def _display_path(path: Path, options: RenderOptions) -> str:
    if not options.relative_paths or options.base_path is None:
        return str(path.resolve())
    try:
        return str(path.resolve().relative_to(options.base_path.resolve()))
    except ValueError:
        return str(path.resolve())


def _line_slice(text: str, line_no: int) -> str:
    lines = text.splitlines(keepends=True)
    if 1 <= line_no <= len(lines):
        return lines[line_no - 1].replace("\r\n", "\n")
    return ""


def _truncate(text: str, max_chars: int | None) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    candidate = text[: max_chars - 3]
    space_idx = candidate.rfind(" ")
    if space_idx > 0:
        candidate = candidate[:space_idx]
    return candidate + "..."


def enrich_result(result: SearchResult, options: RenderOptions) -> DisplaySearchResult:
    snippet = result.chunk_text
    line_start = result.line_start
    line_end = result.line_end
    before: list[str] = []
    after: list[str] = []
    stale = False
    if options.context_lines > 0:
        if not result.file_path.exists():
            stale = True
        else:
            full_text: str | None = None
            try:
                # Decode bytes without universal-newline translation: stored chunk
                # offsets refer to the RAW file text, so line mapping must use the
                # same text (mirrors search._read_anchor_text).
                full_text = result.file_path.read_bytes().decode("utf-8")
            except UnicodeDecodeError:
                try:
                    full_text = result.file_path.read_bytes().decode("latin-1")
                except OSError:
                    stale = True
            except OSError:
                stale = True
            if full_text is not None:
                starts = compute_line_starts(full_text)
                if result.end_char > len(full_text):
                    stale = True
                else:
                    line_start = offset_to_line(starts, result.start_char)
                    line_end = offset_to_line(starts, max(result.end_char - 1, result.start_char))
                    matched = [_line_slice(full_text, line_no) for line_no in range(line_start, line_end + 1)]
                    snippet = "".join(matched).replace("\r\n", "\n") or snippet
                    before = [_line_slice(full_text, line_no) for line_no in range(max(1, line_start - options.context_lines), line_start)]
                    after = [_line_slice(full_text, line_no) for line_no in range(line_end + 1, line_end + 1 + options.context_lines)]
    if not options.show_line_numbers and options.format is not ResultFormat.grep:
        # grep format's contract is `path:line:text`: always emit TRUE line
        # numbers, mirroring grep(1), even when --no-line-numbers is set.
        line_start = None
        line_end = None
    snippet = _truncate(snippet, options.max_chars)
    return DisplaySearchResult(
        search_result=result,
        display_path=_display_path(result.file_path, options),
        line_start=line_start,
        line_end=line_end,
        snippet=snippet,
        context_before=tuple(before),
        context_after=tuple(after),
        stale_offsets=stale,
    )


def format_paths(results: list[SearchResult], options: RenderOptions) -> str:
    paths = sorted({_display_path(result.file_path, options) for result in results})
    return "\n".join(paths)


def format_json(results: list[DisplaySearchResult], *, show_scores: bool, show_why: bool) -> str:
    return json.dumps([display_record(result, show_scores=show_scores, show_why=show_why) for result in results], indent=2)


def format_jsonl(results: list[DisplaySearchResult], *, show_scores: bool, show_why: bool) -> str:
    return "\n".join(json.dumps(display_record(result, show_scores=show_scores, show_why=show_why)) for result in results)


def format_grep(result: DisplaySearchResult, *, show_scores: bool) -> str:
    line = result.line_start or 1
    score = f":{result.search_result.score:.3f}" if show_scores else ""
    snippet = result.snippet.replace("\n", " ").strip()
    return f"{result.display_path}:{line}{score}: {snippet}"


def format_compact(result: DisplaySearchResult, *, show_scores: bool, show_line_numbers: bool) -> str:
    prefix = result.display_path
    if show_line_numbers and result.line_start is not None:
        if result.line_end is not None and result.line_end != result.line_start:
            prefix = f"{prefix}:{result.line_start}-{result.line_end}"
        else:
            prefix = f"{prefix}:{result.line_start}"
    if show_scores:
        prefix = f"{prefix}  score={result.search_result.score:.3f}"
    return f"{prefix}  {result.snippet.splitlines()[0] if result.snippet else ''}"


def render_search_results(results: list[SearchResult], *, options: RenderOptions, console: Console | None = None) -> None:
    out = console or Console()
    if options.format == ResultFormat.paths:
        text = format_paths(results, options)
        if text:
            print(text)
        return
    if options.format == ResultFormat.count:
        print(str(len(results)))
        return
    if not results:
        if options.format == ResultFormat.json:
            print("[]")
        elif options.format in {ResultFormat.jsonl, ResultFormat.grep}:
            return
        else:
            out.print("No matches after filters.")
        return
    display = [enrich_result(result, options) for result in results]
    if options.format == ResultFormat.json:
        print(format_json(display, show_scores=options.show_scores, show_why=options.show_why))
        return
    if options.format == ResultFormat.jsonl:
        text = format_jsonl(display, show_scores=options.show_scores, show_why=options.show_why)
        if text:
            print(text)
        return
    if options.format == ResultFormat.grep:
        for result in display:
            print(format_grep(result, show_scores=options.show_scores))
        return
    if options.format == ResultFormat.compact:
        for result in display:
            out.print(escape(format_compact(result, show_scores=options.show_scores, show_line_numbers=options.show_line_numbers)))
        return
    out.print(f"Search Results ({len(display)}):")
    for result in display:
        prefix = escape(result.display_path)
        if options.show_line_numbers and result.line_start is not None:
            if result.line_end is not None and result.line_end != result.line_start:
                prefix = f"{prefix}:{result.line_start}-{result.line_end}"
            else:
                prefix = f"{prefix}:{result.line_start}"
        if options.show_scores:
            prefix = f"{prefix}  score={result.search_result.score:.3f}"
        out.print(prefix)
        for line in result.context_before:
            out.print(f"  {escape(line.rstrip())}")
        snippet_lines = result.snippet.rstrip().splitlines()
        if snippet_lines:
            out.print("  " + "\n  ".join(escape(line) for line in snippet_lines))
        for line in result.context_after:
            out.print(f"  {escape(line.rstrip())}")
        if options.show_why and result.search_result.why:
            why = "; ".join(f"{key}={value}" for key, value in result.search_result.why.items() if value not in (None, [], ()))
            if why:
                out.print(f"[dim]why: {escape(why)}[/dim]")


CLUSTER_FORMATS = ("rich", "compact", "paths", "json", "jsonl", "count")


def render_cluster_outcome(
    outcome: ClustersOutcome,
    *,
    format: str,
    relative_paths: bool = True,
    base_path: Path | None = None,
    console: Console | None = None,
) -> None:
    """Render a :class:`ClustersOutcome` in one of ``CLUSTER_FORMATS``.

    Machine formats (``json``/``jsonl``/``count``/``paths``) write payload-only
    text to stdout; warnings stay on stderr (repo convention).
    """
    out = console or Console()
    fmt = format.lower()
    clusters = list(outcome.clusters)

    def display_path(path_str: str) -> str:
        return cluster_display_path(path_str, relative_paths=relative_paths, base_path=base_path)

    def json_record(cluster: SemanticCluster) -> dict[str, object]:
        return cluster_record(cluster, relative_paths=relative_paths, base_path=base_path)

    def member_display(member: ClusterMember) -> str:
        return cluster_member_display(member, relative_paths=relative_paths, base_path=base_path)

    if fmt == "count":
        print(str(outcome.total_found))
        return
    if fmt == "paths":
        paths = sorted({display_path(member.file_path) for cluster in clusters for member in cluster.members})
        if paths:
            print("\n".join(paths))
        return
    if fmt == "json":
        print(json.dumps([json_record(c) for c in clusters], indent=2))
        return
    if fmt == "jsonl":
        lines = [json.dumps(json_record(c)) for c in clusters]
        if lines:
            print("\n".join(lines))
        return
    if not clusters:
        out.print("No duplicate clusters found.")
        return
    if fmt == "compact":
        for index, cluster in enumerate(clusters, start=1):
            prefix = f"[{index}] score={cluster.score:.3f}"
            for member in cluster.members:
                print(f"{prefix}  {member_display(member)}")
        return
    shown_note = f", {len(clusters)} shown" if len(clusters) < outcome.total_found else ""
    out.print(f"Semantic Clusters ({outcome.total_found} found{shown_note})")
    for cluster in clusters:
        out.print(escape(f"score={cluster.score:.2f} · {cluster.duplicated_lines}" f" duplicated lines · {len(cluster.members)} members"))
        for member in cluster.members:
            out.print(f"  {escape(member_display(member))}")


DIFF_FORMATS = frozenset({"rich", "json", "jsonl", "count"})


def _diff_display_path(path_str: str, *, relative_paths: bool, base_path: Path | None) -> str:
    path = Path(path_str)
    if not relative_paths or base_path is None:
        return str(path.resolve())
    try:
        return str(path.resolve().relative_to(base_path.resolve()))
    except ValueError:
        return str(path.resolve())


def _diff_entry_record(
    entry: DiffEntry,
    *,
    kind: str | None = None,
    relative_paths: bool = True,
    base_path: Path | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "label": entry.label,
        "file_path": _diff_display_path(entry.file_path, relative_paths=relative_paths, base_path=base_path),
        "line_start": entry.line_start,
        "line_end": entry.line_end,
    }
    if kind is not None:
        return {"kind": kind, **record}
    return record


def _diff_rollup_record(rollup: FileRollup, *, relative_paths: bool = True, base_path: Path | None = None) -> dict[str, object]:
    return {
        "file_path": _diff_display_path(rollup.file_path, relative_paths=relative_paths, base_path=base_path),
        "added": rollup.added,
        "removed": rollup.removed,
        "matched": rollup.matched,
    }


def render_diff_outcome(
    outcome: DiffOutcome,
    *,
    fmt: str,
    absolute_paths: bool = False,
    path_a: Path | None = None,
    path_b: Path | None = None,
    console: Console | None = None,
) -> None:
    """Render a :class:`DiffOutcome` in one of ``DIFF_FORMATS``.

    Machine formats (``json``/``jsonl``/``count``) write payload-only text to
    stdout; warnings stay on stderr (repo convention). Paths are relativized
    against the current working directory unless ``absolute_paths`` is set.
    """
    out = console or Console()
    format_key = fmt.lower()
    relative_paths = not absolute_paths
    base_path = Path.cwd()

    def entry_record(entry: DiffEntry, *, kind: str | None = None) -> dict[str, object]:
        return _diff_entry_record(entry, kind=kind, relative_paths=relative_paths, base_path=base_path)

    added_total = sum(rollup.added for rollup in outcome.files)
    removed_total = sum(rollup.removed for rollup in outcome.files)
    summary = f"{outcome.matched} matched, {added_total} added, {removed_total} removed"

    if format_key == "count":
        print(summary)
        return
    if format_key == "json":
        payload = {
            "matched": outcome.matched,
            "chunks_a": outcome.chunks_a,
            "chunks_b": outcome.chunks_b,
            "threshold": outcome.threshold,
            "added": [entry_record(entry) for entry in outcome.added],
            "removed": [entry_record(entry) for entry in outcome.removed],
            "files": [_diff_rollup_record(rollup, relative_paths=relative_paths, base_path=base_path) for rollup in outcome.files],
        }
        print(json.dumps(payload, indent=2))
        return
    if format_key == "jsonl":
        lines = [json.dumps(entry_record(entry, kind="added")) for entry in outcome.added]
        lines.extend(json.dumps(entry_record(entry, kind="removed")) for entry in outcome.removed)
        if lines:
            print("\n".join(lines))
        return

    def display_tree(path: Path | None) -> str:
        if path is None:
            return "?"
        return _diff_display_path(str(path), relative_paths=relative_paths, base_path=base_path)

    header_a = display_tree(path_a)
    header_b = display_tree(path_b)
    out.print(f"Semantic Diff: {escape(header_a)} -> {escape(header_b)}")
    out.print(escape(summary))
    if not outcome.added and not outcome.removed:
        out.print("Trees are semantically identical.")
        return
    for section, entries in (("Added", outcome.added), ("Removed", outcome.removed)):
        total = added_total if section == "Added" else removed_total
        shown_note = f", showing first {len(entries)}" if len(entries) < total else ""
        out.print(f"{section} ({total}{shown_note})")
        if not entries:
            out.print("  (none)")
        for entry in entries:
            path = _diff_display_path(entry.file_path, relative_paths=relative_paths, base_path=base_path)
            out.print(f"  {escape(f'{path}:{entry.line_start}-{entry.line_end} [label {entry.label}]')}")
    table = Table(title="Per-file changes")
    table.add_column("File")
    table.add_column("Added", justify="right")
    table.add_column("Removed", justify="right")
    table.add_column("Matched", justify="right")
    for rollup in outcome.files:
        table.add_row(
            escape(_diff_display_path(rollup.file_path, relative_paths=relative_paths, base_path=base_path)),
            str(rollup.added),
            str(rollup.removed),
            str(rollup.matched),
        )
    out.print(table)


PACK_FORMATS = ("rich", "markdown", "json")


def _pack_budget_line(outcome: PackOutcome) -> str:
    return f"packaged {outcome.used_tokens}/{outcome.budget} tokens, {len(outcome.selections)} of {outcome.pool_size} chunks"


def _pack_rich_text(outcome: PackOutcome) -> str:
    blocks = []
    for selection in outcome.selections:
        cand = selection.candidate
        blocks.append(f"{cand.path}:{cand.line_start}-{cand.line_end} (score={cand.score:.3f}, ~{selection.tokens} tok)\n{cand.text.rstrip(chr(10))}")
    lines = [*blocks, "", _pack_budget_line(outcome)]
    return "\n".join(lines)


def _pack_markdown_text(outcome: PackOutcome) -> str:
    parts: list[str] = []
    for index, selection in enumerate(outcome.selections):
        cand = selection.candidate
        if index:
            parts += ["---", ""]
        parts += [
            f"### {cand.path}:{cand.line_start}-{cand.line_end} (score={cand.score:.3f}, ~{selection.tokens} tok)",
            "",
            "```plain",
            cand.text.rstrip("\n"),
            "```",
            "",
        ]
    parts.append(_pack_budget_line(outcome))
    return "\n".join(parts)


def render_pack_report(outcome: PackOutcome, queries: list[str], *, format: str) -> str:
    """Render a :class:`PackOutcome` as one of ``PACK_FORMATS``.

    Selections appear in greedy pick order; the trailing budget line reports
    ``used/budget`` tokens and selections-of-pool. ``json`` is a single pinned
    payload object.
    """
    fmt = format.lower()
    if fmt == "json":
        return json.dumps(pack_record(outcome, queries), indent=2)
    if fmt == "markdown":
        return _pack_markdown_text(outcome)
    return _pack_rich_text(outcome)


DEBT_FORMATS = ("rich", "json", "jsonl")

_DEBT_FORMATS_TEXT = ", ".join(DEBT_FORMATS)


def _debt_summary_record(report: DebtReport) -> dict[str, object]:
    """jsonl tail record: the pinned scalars under a ``kind`` discriminator."""
    record = debt_record(report)
    del record["themes"]
    return {"kind": "summary", **record}


def _debt_gate_suffix(report: DebtReport) -> str:
    if report.max_age_days is None:
        return ""
    verdict = "PASS" if report.passed else "FAIL"
    return f" [gate max={report.max_age_days:g}d: {verdict}]"


def _debt_rich_text(report: DebtReport) -> str:
    if report.markers_found == 0:
        return "No debt markers found."
    now = time.time()  # display-only age rendering; reports store raw epochs.
    lines = [f"Debt Themes ({len(report.themes)}) — {report.markers_found} markers, " f"{report.scattered} scattered{_debt_gate_suffix(report)}"]
    for theme in report.themes:
        if theme.oldest_epoch is None:
            age = "oldest ?"
        else:
            age = f"oldest {int(max(0, now - theme.oldest_epoch)) // 86_400}d"
        lines.append(f"\n{theme.label} — {theme.size} members, {age}")
        lines.extend(f"  {match.file_path}:{match.line_start}  {match.marker}  {match.snippet}" for match in theme.matches)
    if report.truncated:
        lines.append("\n(theme list truncated by --top)")
    return "\n".join(lines)


def render_debt_report(report: DebtReport, *, format: str) -> str:
    """Render a :class:`DebtReport` as one of ``DEBT_FORMATS``.

    Themes appear ranked best-first; each lists its capped member matches.
    ``json`` is a single pinned payload object; ``jsonl`` emits one record per
    theme followed by a final ``kind: summary`` record.
    """
    fmt = format.lower()
    if fmt == "json":
        return json.dumps(debt_record(report), indent=2)
    if fmt == "jsonl":
        lines = [json.dumps(_debt_theme_record(theme)) for theme in report.themes]
        lines.append(json.dumps(_debt_summary_record(report)))
        return "\n".join(lines)
    return _debt_rich_text(report)


RERANK_FORMATS = ("rich", "json", "jsonl")

_RERANK_FORMATS_TEXT = ", ".join(RERANK_FORMATS)


def _rerank_rich_text(report: RerankReport) -> str:
    if not report.matches:
        return f"No chunks scored for query {report.query!r}."
    lines = [f"Rerank {report.query!r} — best chunk per file " f"({len(report.matches)} file(s), {report.chunks_scored} chunk(s) scored, model {report.model})"]
    for rank, match in enumerate(report.matches, 1):
        lines.append(f"  {rank}. {match.score:.4f}  {match.file_path}:{match.line_start}-{match.line_end}  {match.snippet}")
    if report.truncated:
        lines.append("\n(matches truncated)")
    return "\n".join(lines)


def render_rerank_report(report: RerankReport, *, format: str) -> str:
    """Render a :class:`RerankReport` as one of ``RERANK_FORMATS``.

    Matches appear ranked best-first; ``json`` is a single pinned payload object
    and ``jsonl`` emits one match record per line.
    """
    fmt = format.lower()
    if fmt == "json":
        return json.dumps(rerank_record(report), indent=2)
    if fmt == "jsonl":
        return "\n".join(json.dumps(_rerank_match_record(match)) for match in report.matches)
    return _rerank_rich_text(report)
