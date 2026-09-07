"""Canonical report->machine-record projection consumed by terminal JSON rendering
and non-CLI transports alike; path relativization, null-field policy and score
inclusion live here."""

from __future__ import annotations

from pathlib import Path

from simgrep.models import (
    ClusterMember,
    DebtReport,
    DebtTheme,
    DisplaySearchResult,
    RerankMatch,
    RerankReport,
    SemanticCluster,
)
from simgrep.pack import PackOutcome


def display_record(result: DisplaySearchResult, *, show_scores: bool, show_why: bool) -> dict[str, object]:
    record: dict[str, object] = {
        "path": result.display_path,
        "score": result.search_result.score if show_scores else None,
        "start_char": result.search_result.start_char,
        "end_char": result.search_result.end_char,
        "text": result.snippet,
        "stale_offsets": result.stale_offsets,
    }
    if show_why and result.search_result.why:
        record["why"] = result.search_result.why
    if result.line_start is not None:
        record["line_start"] = result.line_start
        record["line_end"] = result.line_end
    if result.context_before:
        record["context_before"] = list(result.context_before)
    if result.context_after:
        record["context_after"] = list(result.context_after)
    return record


def cluster_display_path(path_str: str, *, relative_paths: bool, base_path: Path | None) -> str:
    path = Path(path_str)
    if not relative_paths or base_path is None:
        return str(path.resolve())
    try:
        return str(path.resolve().relative_to(base_path.resolve()))
    except ValueError:
        return str(path.resolve())


def cluster_member_display(member: ClusterMember, *, relative_paths: bool, base_path: Path | None) -> str:
    path = cluster_display_path(member.file_path, relative_paths=relative_paths, base_path=base_path)
    return f"{path}:{member.line_start}-{member.line_end}"


def cluster_record(cluster: SemanticCluster, *, relative_paths: bool, base_path: Path | None) -> dict[str, object]:
    return {
        "score": cluster.score,
        "duplicated_lines": cluster.duplicated_lines,
        "members": [
            {
                "label": member.label,
                "file_path": cluster_display_path(member.file_path, relative_paths=relative_paths, base_path=base_path),
                "line_start": member.line_start,
                "line_end": member.line_end,
            }
            for member in cluster.members
        ],
    }


def pack_record(outcome: PackOutcome, queries: list[str]) -> dict[str, object]:
    """Pinned payload shape (CLI ``--format json`` == MCP payload, parity rule)."""
    return {
        "queries": list(queries),
        "budget_tokens": outcome.budget,
        "used_tokens": outcome.used_tokens,
        "pool_size": outcome.pool_size,
        "dropped": outcome.dropped,
        "selections": [
            {
                "path": selection.candidate.path,
                "line_start": selection.candidate.line_start,
                "line_end": selection.candidate.line_end,
                "score": selection.candidate.score,
                "tokens": selection.tokens,
                "truncated": selection.truncated,
                "text": selection.candidate.text,
            }
            for selection in outcome.selections
        ],
    }


def debt_record(report: DebtReport) -> dict[str, object]:
    """Pinned payload shape (CLI ``--format json`` == MCP payload, parity rule)."""
    return {
        "themes": [_debt_theme_record(theme) for theme in report.themes],
        "scattered": report.scattered,
        "markers_found": report.markers_found,
        "chunks_scanned": report.chunks_scanned,
        "truncated": report.truncated,
        "threshold": report.threshold,
        "max_age_days": report.max_age_days,
        "passed": report.passed,
    }


def _debt_theme_record(theme: DebtTheme) -> dict[str, object]:
    return {
        "label": theme.label,
        "size": theme.size,
        "oldest_epoch": theme.oldest_epoch,
        "matches": [
            {
                "file_path": match.file_path,
                "line_start": match.line_start,
                "marker": match.marker,
                "snippet": match.snippet,
            }
            for match in theme.matches
        ],
    }


def _rerank_match_record(match: RerankMatch) -> dict[str, object]:
    """Pinned per-match payload shape (pinned key order)."""
    return {
        "file_path": match.file_path,
        "line_start": match.line_start,
        "line_end": match.line_end,
        "score": match.score,
        "snippet": match.snippet,
    }


def rerank_record(report: RerankReport) -> dict[str, object]:
    """Pinned payload shape (CLI ``--format json`` == MCP payload, parity rule)."""
    return {
        "query": report.query,
        "model": report.model,
        "matches": [_rerank_match_record(match) for match in report.matches],
        "files_seen": report.files_seen,
        "chunks_scored": report.chunks_scored,
        "truncated": report.truncated,
    }
