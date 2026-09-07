from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from simgrep.corpus import StoredChunk
from simgrep.models import DiversityMode, FileRole, LexicalFallbackMode, PathBoost, SearchOptions, SearchResult


def _normalize_semantic(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    raw = float(value)
    if -1.0 <= raw <= 1.0:
        return max(0.0, min(1.0, (raw + 1.0) / 2.0))
    if raw > 1.0:
        return raw / (raw + 1.0)
    return 0.0


def _normalize_lexical(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    raw = max(0.0, float(value))
    if raw == 0.0:
        return 0.0
    return raw / (raw + 1.0)


def _tokenize(query: str) -> list[str]:
    if not query:
        return []
    split_acronyms = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", query)
    lowered = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", split_acronyms).lower()
    snake_kebab_split = re.sub(r"[_\-]+", " ", lowered)
    parts = re.split(r"[^a-z0-9]+", snake_kebab_split)
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if not part or part in seen:
            continue
        seen.add(part)
        out.append(part)
        if len(out) >= 8:
            break
    return out


def _path_boost(path: Optional[Path], boosts: tuple[PathBoost, ...]) -> float:
    if path is None or not boosts:
        return 0.0
    path_str = path.as_posix()
    name = path.name
    best = 0.0
    for entry in boosts:
        if fnmatch.fnmatch(path_str, entry.pattern) or fnmatch.fnmatch(name, entry.pattern):
            best = max(best, float(entry.weight))
    return best


@dataclass
class _Row:
    label: int
    file_path: Optional[Path]
    chunk_text: Optional[str]
    start_char_offset: Optional[int]
    end_char_offset: Optional[int]
    line_start: Optional[int]
    line_end: Optional[int]
    file_role: FileRole
    language: str
    semantic_raw: Optional[float] = None
    semantic_norm: float = 0.0
    lexical_raw: Optional[float] = None
    lexical_norm: float = 0.0
    final_score: float = 0.0
    role_multiplier: float = 1.0
    lexical_only: bool = False


_WINDOW_RESULTS = 2


def _diversify(rows: list[_Row], mode: DiversityMode, top: int) -> list[_Row]:
    if mode == DiversityMode.none:
        return rows[:top]
    selected: list[_Row] = []
    seen_files: set[str] = set()
    per_package: dict[str, int] = {}
    for row in rows:
        key = str(row.file_path or "<unknown>")
        package = str(row.file_path.parent if row.file_path else Path("<unknown>"))
        if mode == DiversityMode.file and key in seen_files:
            continue
        if mode == DiversityMode.package and per_package.get(package, 0) >= 2:
            continue
        if mode == DiversityMode.window:
            recent = {str(r.file_path or "<unknown>") for r in selected[-_WINDOW_RESULTS:]}
            if key in recent:
                continue
        seen_files.add(key)
        per_package[package] = per_package.get(package, 0) + 1
        selected.append(row)
        if len(selected) >= top:
            break
    return selected


def _build_why(
    row: _Row,
    tokens: list[str],
    boosts: tuple[PathBoost, ...],
    contrast_unlike: dict[int, float] | None,
    contrast_like: dict[int, float] | None,
) -> dict[str, Any]:
    why: dict[str, Any] = {
        "semantic_norm": round(row.semantic_norm, 4),
        "lexical_norm": round(row.lexical_norm, 4),
        "lexical_only": row.lexical_only,
        "role_multiplier": round(row.role_multiplier, 2),
        "path_boost": round(_path_boost(row.file_path, boosts), 4),
        "query_terms": tokens,
    }
    if contrast_unlike is not None:
        # semantic_like carries the true like-side score; semantic_unlike the weighted
        # term lambda * s_unlike, so semantic_like - semantic_unlike reconciles with
        # the combined contribution in row.semantic_raw.
        if contrast_like is not None:
            why["semantic_like"] = round(contrast_like.get(row.label, 0.0), 4)
        else:
            why["semantic_like"] = round(float(row.semantic_raw), 4) if row.semantic_raw is not None else 0.0
        why["semantic_unlike"] = round(contrast_unlike.get(row.label, 0.0), 4)
    return why


def rank_candidates(
    *,
    query: str,
    semantic_matches: Iterable[tuple[int, float]],
    semantic_rows: Sequence[StoredChunk],
    lexical_rows: Sequence[tuple[StoredChunk, float]],
    options: SearchOptions,
    contrast_unlike: dict[int, float] | None = None,
    contrast_like: dict[int, float] | None = None,
) -> list[SearchResult]:
    if options.top <= 0:
        return []
    semantic_scores: dict[int, float] = {}
    for label, score in semantic_matches:
        semantic_scores[int(label)] = float(score)

    rows: dict[int, _Row] = {}

    def _row_for(chunk: StoredChunk) -> _Row:
        item = rows.get(chunk.label)
        if item is None:
            item = _Row(
                label=chunk.label,
                file_path=chunk.file_path,
                chunk_text=chunk.text,
                start_char_offset=chunk.start_char,
                end_char_offset=chunk.end_char,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                file_role=chunk.role,
                language=chunk.language,
            )
            rows[chunk.label] = item
        return item

    for chunk in semantic_rows:
        item = _row_for(chunk)
        item.semantic_raw = semantic_scores.get(item.label)
        item.semantic_norm = _normalize_semantic(item.semantic_raw)

    for chunk, score in lexical_rows:
        item = _row_for(chunk)
        item.lexical_raw = float(score)
        item.lexical_norm = _normalize_lexical(item.lexical_raw)
        item.lexical_only = item.semantic_raw is None

    role_multiplier = {
        FileRole.source: 1.08,
        FileRole.test: 0.94,
        FileRole.docs: 0.86,
        FileRole.config: 0.9,
        FileRole.generated: 0.7,
    }
    semantic_items: list[_Row] = []
    lexical_only_items: list[_Row] = []
    for item in rows.values():
        item.role_multiplier = role_multiplier.get(item.file_role, 1.0)
        boost = _path_boost(item.file_path, options.path_boosts)
        if item.semantic_raw is None:
            lexical_only_items.append(item)
            continue
        fused = ((1.0 - options.lexical_weight) * item.semantic_norm) + (options.lexical_weight * item.lexical_norm)
        item.final_score = min(1.0, fused * item.role_multiplier + boost)
        semantic_items.append(item)

    lexical_only_items.sort(key=lambda r: (-r.lexical_norm, r.label))
    if options.lexical_fallback == LexicalFallbackMode.off:
        lexical_only_items = []
    elif options.lexical_fallback == LexicalFallbackMode.fill and semantic_items:
        cap = min(max(min(x.final_score for x in semantic_items) - 0.001, 0.0), 0.35)
        for item in lexical_only_items:
            item.final_score = min(item.lexical_norm * item.role_multiplier + _path_boost(item.file_path, options.path_boosts), cap)
    elif options.lexical_fallback == LexicalFallbackMode.empty and semantic_items:
        for item in lexical_only_items:
            item.final_score = 0.0
    else:
        for item in lexical_only_items:
            item.final_score = item.lexical_norm * item.role_multiplier + _path_boost(item.file_path, options.path_boosts)

    tokens = _tokenize(query)

    def _token_coverage_ok(row: _Row) -> bool:
        if not tokens or options.lexical_weight <= 0 or len(tokens) < 3:
            return True
        text = (row.chunk_text or "").lower()
        matched = sum(1 for token in tokens if token in text)
        required = max(1, len(tokens) - 1)
        return matched >= required

    merged = semantic_items + [x for x in lexical_only_items if _token_coverage_ok(x)]
    merged = [x for x in merged if x.final_score >= options.min_score]
    merged.sort(key=lambda r: (-r.final_score, -int(r.semantic_raw is not None), -r.semantic_norm, -r.lexical_norm, r.label))
    merged = _diversify(merged, options.diversity, options.top)
    return [
        SearchResult(
            label=row.label,
            score=row.final_score,
            file_path=row.file_path or Path("<unknown>"),
            file_role=row.file_role,
            language=row.language,
            chunk_text=row.chunk_text or "",
            start_char=row.start_char_offset or 0,
            end_char=row.end_char_offset or 0,
            line_start=row.line_start,
            line_end=row.line_end,
            why=_build_why(row, tokens, options.path_boosts, contrast_unlike, contrast_like),
        )
        for row in merged
    ]
