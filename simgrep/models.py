from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

SCHEMA_VERSION = 1
DEFAULT_MODEL = "ibm-granite/granite-embedding-30m-english"
DEFAULT_FILE_SCAN_PATTERNS: tuple[str, ...] = (
    "*.txt",
    "*.md",
    "*.rst",
    "*.py",
    "*.js",
    "*.ts",
    "*.tsx",
    "*.jsx",
    "*.java",
    "*.go",
    "*.rs",
    "*.c",
    "*.cpp",
    "*.h",
    "*.hpp",
    "*.cs",
    "*.rb",
    "*.php",
    "*.swift",
    "*.kt",
    "*.scala",
    "*.sh",
    "*.bash",
    "*.zsh",
    "*.toml",
    "*.yaml",
    "*.yml",
    "*.json",
    "*.xml",
    "*.html",
    "*.css",
    "*.sql",
    "*.dockerfile",
    "Dockerfile",
)


class ResultFormat(str, Enum):
    rich = "rich"
    compact = "compact"
    paths = "paths"
    json = "json"
    jsonl = "jsonl"
    count = "count"  # type: ignore[assignment]
    grep = "grep"


class DiversityMode(str, Enum):
    none = "none"
    window = "window"
    file = "file"
    package = "package"


class FreshnessMode(str, Enum):
    auto = "auto"
    skip = "skip"
    check = "check"


class ChangeDetectionMode(str, Enum):
    stat = "stat"
    hash = "hash"


class LexicalFallbackMode(str, Enum):
    off = "off"
    fill = "fill"
    empty = "empty"


class FileRole(str, Enum):
    source = "source"
    test = "test"
    docs = "docs"
    config = "config"
    dependency_metadata = "dependency_metadata"
    build_metadata = "build_metadata"
    generated = "generated"
    data = "data"
    unknown = "unknown"


class IndexState(str, Enum):
    ready = "ready"
    indexing = "indexing"
    failed = "failed"


@dataclass(frozen=True)
class AppConfig:
    schema_version: int = SCHEMA_VERSION
    model: str = DEFAULT_MODEL
    chunk_size: int = 128
    chunk_overlap: int = 20
    batch_size: int = 128
    max_file_size_bytes: int | None = 10_485_760
    follow_symlinks: bool = False
    file_patterns: tuple[str, ...] = DEFAULT_FILE_SCAN_PATTERNS
    lexical_top: int = 50
    lexical_weight: float = 0.25
    freshness: FreshnessMode = FreshnessMode.auto
    context_lines: int = 0
    max_chars: int = 1200


@dataclass(frozen=True)
class ProjectConfig:
    schema_version: int
    name: str
    root: Path
    indexed_paths: tuple[Path, ...]
    model: str
    chunk_size: int
    chunk_overlap: int

    @property
    def simgrep_dir(self) -> Path:
        return self.root / ".simgrep"

    @property
    def metadata_db_path(self) -> Path:
        return self.simgrep_dir / "metadata.duckdb"

    @property
    def vector_index_path(self) -> Path:
        return self.simgrep_dir / "vectors.usearch"

    @property
    def index_lock_path(self) -> Path:
        return self.simgrep_dir / "index.lock"


@dataclass(frozen=True)
class PathBoost:
    pattern: str
    weight: float = 0.15


@dataclass(frozen=True)
class SearchOptions:
    query: str
    top: int = 5
    min_score: float = 0.0
    candidate_top: int | None = None
    lexical_top: int = 50
    lexical_weight: float = 0.25
    diversity: DiversityMode = DiversityMode.window
    scope_path: Path | None = None
    file_filter: tuple[str, ...] = ()
    keyword_filter: str | None = None
    include_globs: tuple[str, ...] = ()
    exclude_globs: tuple[str, ...] = ()
    path_boosts: tuple[PathBoost, ...] = ()
    lexical_fallback: LexicalFallbackMode = LexicalFallbackMode.fill
    expr: str | None = None


@dataclass(frozen=True)
class Anchor:
    """Resolved `similar` anchor: its text plus optional file origin for self-exclusion."""

    text: str
    origin: Path | None = None
    start_char: int | None = None
    end_char: int | None = None

    @property
    def has_span(self) -> bool:
        return self.origin is not None and self.start_char is not None and self.end_char is not None


@dataclass(frozen=True)
class SimilarOptions:
    """Options for the query-by-example flow; `search.query` carries the like-anchor text."""

    search: SearchOptions
    anchor: Anchor
    unlike: Anchor | None = None
    unlike_weight: float = 0.5
    include_self: bool = False


@dataclass(frozen=True)
class ResultFilters:
    scope_path: Path | None = None
    file_filter: tuple[str, ...] = ()
    keyword_filter: str | None = None
    include_globs: tuple[str, ...] = ()
    exclude_globs: tuple[str, ...] = ()

    @classmethod
    def from_search_options(cls, options: SearchOptions) -> "ResultFilters":
        return cls(
            scope_path=options.scope_path,
            file_filter=options.file_filter,
            keyword_filter=options.keyword_filter,
            include_globs=options.include_globs,
            exclude_globs=options.exclude_globs,
        )


@dataclass(frozen=True)
class VectorHit:
    label: int
    score: float


@dataclass(frozen=True)
class SearchResult:
    label: int
    score: float
    file_path: Path
    chunk_text: str
    start_char: int
    end_char: int
    line_start: int | None
    line_end: int | None
    file_role: FileRole
    language: str
    why: dict[str, object] = field(default_factory=dict)

    @property
    def start_char_offset(self) -> int:
        return self.start_char

    @property
    def end_char_offset(self) -> int:
        return self.end_char


@dataclass(frozen=True)
class Chunk:
    id: int
    file_id: int
    text: str
    start: int
    end: int
    tokens: int
    line_start: int | None = None
    line_end: int | None = None


@dataclass(frozen=True)
class FileRecord:
    id: int
    path: Path
    size_bytes: int
    mtime_ns: int
    sha256: str | None = None
    role: FileRole = FileRole.unknown
    language: str = "unknown"
    is_test: bool = False
    is_generated: bool = False


@dataclass(frozen=True)
class ChunkRecord:
    label: int
    file_id: int
    text: str
    start_char: int
    end_char: int
    token_count: int
    line_start: int | None = None
    line_end: int | None = None
    kind: str = "mixed"


@dataclass(frozen=True)
class TermRecord:
    label: int
    term: str
    field: str
    tf: int
    weight: float


@dataclass(frozen=True)
class LexicalCandidate:
    label: int
    score: float
    file_path: Path
    chunk_text: str
    start_char: int
    end_char: int
    line_start: int | None
    line_end: int | None
    file_role: FileRole
    language: str
    matched_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProjectStatus:
    project_name: str
    files_count: int
    chunks_count: int
    index_exists: bool
    index_state: IndexState | None = None


@dataclass
class IndexStats:
    files_seen: int = 0
    files_processed: int = 0
    files_indexed: int = 0
    files_skipped_unchanged: int = 0
    files_skipped_too_large: int = 0
    files_pruned_deleted: int = 0
    ignored_count: int = 0
    unreadable_count: int = 0
    chunks_indexed: int = 0
    vectors_added: int = 0
    vectors_removed: int = 0
    index_mutated: bool = False
    errors: int = 0
    plan_seconds: float = 0.0
    total_seconds: float = 0.0


@dataclass(frozen=True)
class SearchOutcome:
    results: list[SearchResult]
    base_path: Path
    files_seen: int = 0
    chunks_searched: int = 0
    semantic_candidates: int = 0


@dataclass(frozen=True)
class ScanOptions:
    patterns: tuple[str, ...] = DEFAULT_FILE_SCAN_PATTERNS
    include_globs: tuple[str, ...] = ()
    exclude_globs: tuple[str, ...] = ()
    max_file_size_bytes: int | None = 10_485_760
    follow_symlinks: bool = False


@dataclass(frozen=True)
class IndexOptions:
    rebuild: bool = False
    dry_run: bool = False
    include_globs: tuple[str, ...] = ()
    exclude_globs: tuple[str, ...] = ()
    patterns: tuple[str, ...] = ()
    max_workers: int = 4
    change_detection: ChangeDetectionMode = ChangeDetectionMode.stat


@dataclass(frozen=True)
class EphemeralIndexOptions:
    scan: ScanOptions = field(default_factory=ScanOptions)
    max_workers: int = 4


FilePlanStatus = Literal["new", "changed", "unchanged", "deleted", "ignored", "too_large", "unreadable"]


@dataclass(frozen=True)
class FilePlanEntry:
    path: Path
    status: FilePlanStatus
    existing_file_id: int | None = None
    old_size_bytes: int | None = None
    old_mtime_ns: int | None = None
    old_hash: str | None = None
    new_hash: str | None = None
    reason: str | None = None
    size_bytes: int | None = None
    mtime_ns: int | None = None


@dataclass(frozen=True)
class FilePlan:
    entries: tuple[FilePlanEntry, ...]

    @property
    def new_count(self) -> int:
        return self._count("new")

    @property
    def changed_count(self) -> int:
        return self._count("changed")

    @property
    def unchanged_count(self) -> int:
        return self._count("unchanged")

    @property
    def deleted_count(self) -> int:
        return self._count("deleted")

    @property
    def ignored_count(self) -> int:
        return self._count("ignored")

    @property
    def too_large_count(self) -> int:
        return self._count("too_large")

    @property
    def unreadable_count(self) -> int:
        return self._count("unreadable")

    @property
    def has_mutations(self) -> bool:
        return self.new_count + self.changed_count + self.deleted_count > 0

    @property
    def has_indexable_work(self) -> bool:
        return self.new_count + self.changed_count > 0

    def _count(self, status: str) -> int:
        return sum(1 for entry in self.entries if entry.status == status)


@dataclass(frozen=True)
class RenderOptions:
    format: ResultFormat
    relative_paths: bool = True
    base_path: Path | None = None
    show_scores: bool = True
    show_line_numbers: bool = True
    context_lines: int = 0
    max_chars: int | None = 1200
    query: str = ""
    show_why: bool = False


@dataclass(frozen=True)
class DisplaySearchResult:
    search_result: SearchResult
    display_path: str
    line_start: int | None
    line_end: int | None
    snippet: str
    context_before: tuple[str, ...] = ()
    context_after: tuple[str, ...] = ()
    stale_offsets: bool = False


@dataclass(frozen=True)
class ClusterMember:
    label: int
    file_path: str
    line_start: int
    line_end: int


@dataclass(frozen=True)
class SemanticCluster:
    """A group of semantically duplicated chunks."""

    members: tuple[ClusterMember, ...]  # sorted by (file_path, line_start, label)
    score: float  # min cosine among the qualifying (>= threshold) pairs that formed the cluster, in [0, 1]
    duplicated_lines: int  # per-file union of inclusive line spans, summed


@dataclass(frozen=True)
class ClustersOptions:
    threshold: float = 0.8  # 0 < t <= 1
    min_size: int = 2  # >= 2
    top: int = 20  # >= 1, cap on returned clusters
    same_file: bool = False  # False: cross-file pairs only
    max_chunks: int = 50_000  # hard O(N^2) guard


@dataclass(frozen=True)
class ClustersOutcome:
    clusters: tuple[SemanticCluster, ...]  # after top cap
    total_found: int  # components passing min_size before cap
    chunks_scanned: int


@dataclass(frozen=True)
class DiffEntry:
    label: int
    file_path: str
    line_start: int
    line_end: int


@dataclass(frozen=True)
class FileRollup:
    file_path: str
    added: int  # chunks appearing in tree B under this path
    removed: int  # chunks last seen in tree A under this path
    matched: int  # chunks matched under this path (A-side for pairs, plus B-side)


@dataclass(frozen=True)
class DiffOptions:
    threshold: float = 0.8
    top: int = 50
    max_chunks: int = 50_000


@dataclass(frozen=True)
class DiffOutcome:
    added: tuple[DiffEntry, ...]  # from tree B, capped at top
    removed: tuple[DiffEntry, ...]  # from tree A, capped at top
    matched: int  # pair count, pre-cap
    files: tuple[FileRollup, ...]  # rollup over all chunks, pre-cap
    chunks_a: int
    chunks_b: int
    threshold: float


@dataclass(frozen=True)
class DebtOptions:
    """Options for `simgrep debt`: semantic debt-marker radar."""

    threshold: float = 0.8  # 0 < t <= 1; cosine joining two marker chunks into one theme
    min_size: int = 2  # >=2; markers-per-theme floor, smaller components stay scattered
    top: int = 20  # 1..200 themes shown
    max_members: int = 8  # 1..50 matches shown per theme
    max_age_days: float | None = None  # None or > 0; gate: no theme older than this
    max_chunks: int = 50_000  # corpus scan guard


@dataclass(frozen=True)
class DebtMatch:
    """One debt-marker occurrence."""

    file_path: str  # display path, base-relative when persistent
    line_start: int
    marker: str  # TODO | FIXME | XXX | HACK | WORKAROUND
    snippet: str  # rest of the marker line, stripped, capped at 117 chars + "..."


@dataclass(frozen=True)
class DebtTheme:
    """A semantic cluster of debt markers."""

    label: str  # top-2 frequent tokens of member chunks, " / "-joined; fallback "debt"
    size: int  # member count (>= min_size)
    matches: tuple[DebtMatch, ...]  # sorted (file_path, line_start), capped at max_members
    oldest_epoch: int | None  # min member-file last-commit epoch; None when git ages unavailable


@dataclass(frozen=True)
class DebtReport:
    """Corpus-wide debt radar outcome."""

    themes: tuple[DebtTheme, ...]  # ranked (size desc, oldest_epoch asc w/ None last, label asc), top cap
    scattered: int  # singleton markers (below min_size)
    markers_found: int  # total marker occurrences
    chunks_scanned: int  # corpus chunks considered
    truncated: bool  # True when more themes found than top
    threshold: float
    max_age_days: float | None
    passed: bool | None  # None when max_age_days None; else every dated theme within the gate


@dataclass(frozen=True)
class RerankOptions:
    query: str
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top: int = 25  # search --rerank window
    batch_size: int = 32
    max_chunks: int = 512  # standalone command cap


@dataclass(frozen=True)
class RerankMatch:
    file_path: str
    line_start: int
    line_end: int
    score: float
    snippet: str  # first 120 chars of chunk, single line


@dataclass(frozen=True)
class RerankReport:
    query: str
    model: str
    matches: tuple[RerankMatch, ...]
    files_seen: int = 0
    chunks_scored: int = 0
    truncated: bool = False
