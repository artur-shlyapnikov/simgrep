from __future__ import annotations

import fnmatch
import hashlib
import os
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pathspec

from simgrep.errors import SimgrepError
from simgrep.models import ChangeDetectionMode, FilePlan, FilePlanEntry, FilePlanStatus, FileRecord, FileRole, ProjectConfig, ScanOptions

IGNORED_DIR_NAMES = {
    ".simgrep",
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "build",
    "dist",
    "target",
    ".cargo",
    ".gradle",
    ".idea",
    ".vscode",
}
SENSITIVE_FILE_PATTERNS = {".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx", "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "*.kubeconfig"}
TEST_DIR_SEGMENTS = {"test", "tests", "spec", "specs", "integrationtest", "e2e", "integration", "e2e-tests"}
GENERATED_PATH_SEGMENTS = {"generated", "target/generated-sources", "build/generated", ".generated", "gen"}
KNOWN_LOCK_FILES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "uv.lock", "Cargo.lock", "Gemfile.lock", "go.sum", "requirements.txt"}
METADATA_FILE_NAMES = {"renovate.json", ".renovaterc", ".renovaterc.json"}


@dataclass(frozen=True)
class FileScanEntry:
    path: Path
    resolved_path: Path
    rel_path: str
    size_bytes: int
    mtime_ns: int


@dataclass(frozen=True)
class FileFeatures:
    file_path: Path
    file_role: FileRole
    language: str
    is_test: bool
    is_generated: bool


@dataclass(frozen=True)
class ChunkLexicalFeatures:
    chunk_terms: list[str]
    path_terms: list[str]
    symbol_terms: list[str]


def calculate_file_hash(file_path: Path) -> str:
    sha = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _read_ignore_lines(base_dir: Path) -> list[str]:
    """Read raw ignore-pattern lines (.gitignore then .repo_ignore) from a directory."""
    lines: list[str] = []
    for ignore_name in (".gitignore", ".repo_ignore"):
        ignore_file = base_dir / ignore_name
        if ignore_file.is_file():
            try:
                lines.extend(ignore_file.read_text(encoding="utf-8").splitlines())
            except OSError:
                continue
    return lines


def _rewrite_ignore_lines(lines: list[str], base_rel: str) -> list[str]:
    """Scope raw ignore patterns from a nested directory to be root-relative.

    Mirrors git semantics: a bare pattern matches anywhere below its defining
    directory, an anchored (/leading) pattern is relative to that directory,
    and a pattern containing a slash is directory-relative.
    """
    rewritten: list[str] = []
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        negated = line.startswith("!")
        body = line[1:] if negated else line
        if base_rel == "":
            scoped = body
        elif body.startswith("/"):
            scoped = f"{base_rel}/{body[1:]}"
        elif "/" in body:
            scoped = f"{base_rel}/{body}"
        else:
            scoped = f"{base_rel}/**/{body}"
        rewritten.append(f"!{scoped}" if negated else scoped)
    return rewritten


def _load_ignore_spec(base_dir: Path) -> pathspec.PathSpec | None:
    lines = _read_ignore_lines(base_dir)
    if not lines:
        return None
    try:
        return pathspec.PathSpec.from_lines("gitwildmatch", lines)
    except Exception:
        return None


def _matches(path: Path, rel_path: str, patterns: tuple[str, ...]) -> bool:
    name = path.name
    return any(fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(name, pattern) for pattern in patterns)


def scan_files(path: Path, options: ScanOptions) -> list[FileScanEntry]:
    root = path.parent if path.is_file() else path
    root = root.resolve()
    ignore_lines: list[str] = []
    spec_cache: tuple[int, pathspec.PathSpec | None] | None = None

    def _accumulated_ignore_spec() -> pathspec.PathSpec | None:
        nonlocal spec_cache
        if not ignore_lines:
            return None
        if spec_cache is None or spec_cache[0] != len(ignore_lines):
            try:
                built: pathspec.PathSpec | None = pathspec.PathSpec.from_lines("gitwildmatch", ignore_lines)
            except Exception:
                built = None
            spec_cache = (len(ignore_lines), built)
        return spec_cache[1]

    patterns = options.patterns
    found: dict[Path, FileScanEntry] = {}
    visited: set[Path] = set()

    scanning_single_file = path.is_file()

    def _maybe_add(file_path: Path, resolved: Path, stat: os.stat_result) -> None:
        try:
            rel = resolved.relative_to(root).as_posix()
        except ValueError:
            return
        if any(part in IGNORED_DIR_NAMES for part in Path(rel).parts):
            return
        if any(fnmatch.fnmatch(file_path.name, pat) for pat in SENSITIVE_FILE_PATTERNS):
            return
        if scanning_single_file:
            root_only = _load_ignore_spec(root)
            if root_only and root_only.match_file(rel):
                return
        else:
            accumulated_spec = _accumulated_ignore_spec()
            if accumulated_spec and accumulated_spec.match_file(rel):
                return
        if patterns and not _matches(file_path, rel, patterns):
            return
        if options.include_globs and not _matches(file_path, rel, options.include_globs):
            return
        if options.exclude_globs and _matches(file_path, rel, options.exclude_globs):
            return
        found[resolved] = FileScanEntry(path=file_path, resolved_path=resolved, rel_path=rel, size_bytes=int(stat.st_size), mtime_ns=int(stat.st_mtime_ns))

    if path.is_file():
        try:
            _maybe_add(path, path.resolve(), path.stat())
        except OSError:
            return []
        return list(found.values())

    def _walk(current: Path) -> None:
        try:
            dir_rel = current.relative_to(root).as_posix()
        except ValueError:
            dir_rel = ""
        if dir_rel == ".":
            dir_rel = ""
        try:
            ignore_lines.extend(_rewrite_ignore_lines(_read_ignore_lines(current), dir_rel))
        except OSError as err:
            if current == root:
                if isinstance(err, FileNotFoundError):
                    raise SimgrepError(f"Path not found: {current}", hint="Check the path and try again.") from err
                raise SimgrepError(f"Cannot read directory: {current}", hint="Check directory permissions.") from err
            return
        try:
            if options.follow_symlinks:
                real = current.resolve()
                if real in visited:
                    return
                visited.add(real)
        except OSError:
            return
        try:
            entries = os.scandir(current)
        except OSError as err:
            if current == root:
                if isinstance(err, FileNotFoundError):
                    raise SimgrepError(f"Path not found: {current}", hint="Check the path and try again.") from err
                raise SimgrepError(f"Cannot read directory: {current}", hint="Check directory permissions.") from err
            raise
        try:
            with entries as dir_entries:
                for entry in sorted(dir_entries, key=lambda e: e.name):
                    entry_path = Path(entry.path)
                    if entry.is_symlink() and not options.follow_symlinks:
                        continue
                    if entry.is_dir(follow_symlinks=options.follow_symlinks):
                        if entry.name in IGNORED_DIR_NAMES:
                            continue
                        try:
                            dir_resolved = entry_path.resolve()
                        except OSError:
                            continue
                        if dir_resolved in visited:
                            continue
                        try:
                            child_rel = dir_resolved.relative_to(root).as_posix()
                        except ValueError:
                            child_rel = ""
                        prune_spec = _accumulated_ignore_spec()
                        if child_rel and prune_spec and prune_spec.match_file(child_rel + "/"):
                            continue
                        _walk(entry_path)
                        continue
                    if not entry.is_file(follow_symlinks=options.follow_symlinks):
                        continue
                    try:
                        file_stat = entry.stat(follow_symlinks=options.follow_symlinks)
                        resolved = entry_path.resolve()
                        if resolved in visited:
                            continue
                        _maybe_add(entry_path, resolved, file_stat)
                    except OSError:
                        continue
        except OSError:
            return

    _walk(root)
    return sorted(found.values(), key=lambda entry: str(entry.resolved_path))


def build_file_plan(
    discovered: list[FileScanEntry],
    existing: dict[Path, FileRecord],
    *,
    options: ScanOptions,
    change_detection: ChangeDetectionMode,
) -> FilePlan:
    entries: list[FilePlanEntry] = []
    current_paths = {entry.resolved_path for entry in discovered}
    for old_path, record in sorted(existing.items(), key=lambda item: str(item[0])):
        # Stored paths were resolved at upsert time (see upsert_file); comparing
        # them directly keeps plan building free of per-row resolve() syscalls.
        if old_path not in current_paths:
            entries.append(
                FilePlanEntry(
                    path=old_path,
                    status="deleted",
                    existing_file_id=record.id,
                    old_size_bytes=record.size_bytes,
                    old_mtime_ns=record.mtime_ns,
                    old_hash=record.sha256,
                )
            )
    for entry in discovered:
        if options.max_file_size_bytes is not None and entry.size_bytes > options.max_file_size_bytes:
            entries.append(FilePlanEntry(path=entry.resolved_path, status="too_large", size_bytes=entry.size_bytes, mtime_ns=entry.mtime_ns))
            continue
        old = existing.get(entry.resolved_path)
        if old is None:
            try:
                new_hash = calculate_file_hash(entry.resolved_path) if change_detection == ChangeDetectionMode.hash else None
            except OSError:
                entries.append(
                    FilePlanEntry(
                        path=entry.resolved_path,
                        status="unreadable",
                        reason="hash_failed",
                        size_bytes=entry.size_bytes,
                        mtime_ns=entry.mtime_ns,
                    )
                )
                continue
            entries.append(FilePlanEntry(path=entry.resolved_path, status="new", size_bytes=entry.size_bytes, mtime_ns=entry.mtime_ns, new_hash=new_hash))
            continue
        if change_detection == ChangeDetectionMode.stat:
            status: FilePlanStatus = "unchanged" if old.size_bytes == entry.size_bytes and old.mtime_ns == entry.mtime_ns else "changed"
            entries.append(
                FilePlanEntry(
                    path=entry.resolved_path,
                    status=status,
                    existing_file_id=old.id,
                    old_size_bytes=old.size_bytes,
                    old_mtime_ns=old.mtime_ns,
                    old_hash=old.sha256,
                    new_hash=old.sha256,
                    size_bytes=entry.size_bytes,
                    mtime_ns=entry.mtime_ns,
                )
            )
            continue
        try:
            new_hash = calculate_file_hash(entry.resolved_path)
        except OSError:
            entries.append(
                FilePlanEntry(
                    path=entry.resolved_path,
                    status="unreadable",
                    reason="hash_failed",
                    size_bytes=entry.size_bytes,
                    mtime_ns=entry.mtime_ns,
                )
            )
            continue
        hash_status: FilePlanStatus = "unchanged" if old.sha256 is not None and old.sha256 == new_hash else "changed"
        entries.append(
            FilePlanEntry(
                path=entry.resolved_path,
                status=hash_status,
                existing_file_id=old.id,
                old_size_bytes=old.size_bytes,
                old_mtime_ns=old.mtime_ns,
                old_hash=old.sha256,
                new_hash=new_hash,
                size_bytes=entry.size_bytes,
                mtime_ns=entry.mtime_ns,
            )
        )
    return FilePlan(entries=tuple(sorted(entries, key=lambda entry: str(entry.path))))


def build_project_file_plan(
    project: ProjectConfig,
    existing: dict[Path, FileRecord],
    *,
    scan_options: ScanOptions,
    change_detection: ChangeDetectionMode,
) -> FilePlan:
    discovered: dict[Path, FileScanEntry] = {}
    for target in project.indexed_paths:
        for entry in scan_files(target, scan_options):
            discovered[entry.resolved_path] = entry
    return build_file_plan(list(discovered.values()), existing, options=scan_options, change_detection=change_detection)


_LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".java": "java",
    ".kt": "kotlin",
    ".scala": "scala",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".cpp": "c++",
    ".h": "c/c++",
    ".hpp": "c++",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "jsx",
    ".sh": "shell",
    ".bash": "bash",
    ".zsh": "zsh",
    ".md": "markdown",
    ".rst": "rst",
    ".txt": "text",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".xml": "xml",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".dockerfile": "dockerfile",
}

_TEST_PY_RE = re.compile(r"^test_.*\.py$|^.*_test\.py$")
_TEST_GO_RE = re.compile(r"^.*_test\.go$")
_TEST_JS_RE = re.compile(r"\.(spec|test)\.(ts|tsx|js|jsx)$", re.IGNORECASE)
_BUILD_FILE_RE = re.compile(r"^(pom\.xml|build\.gradle|settings\.gradle)$", re.IGNORECASE)
_DOCKERFILE_RE = re.compile(r"^Dockerfile(\..+)?$")
_DOC_SUFFIXES = frozenset({".md", ".rst", ".txt"})
_CONFIG_SUFFIXES = frozenset({".yaml", ".yml", ".json", ".xml", ".toml", ".properties", ".env"})
_DATA_SUFFIXES = frozenset({".csv", ".tsv"})


def infer_language(path: Path) -> str:
    if path.name.lower() == "dockerfile":
        return "dockerfile"
    return _LANGUAGE_BY_SUFFIX.get(path.suffix.lower(), "unknown")


def is_test_path(path: Path) -> bool:
    name = path.name
    if _TEST_PY_RE.match(name):
        return True
    if _TEST_GO_RE.match(name):
        return True
    if _TEST_JS_RE.search(name):
        return True
    path_str = str(path).lower()
    return any(f"/{seg}/" in path_str or path_str.endswith(f"/{seg}") for seg in TEST_DIR_SEGMENTS)


def is_generated_path(path: Path) -> bool:
    lower_parts = [part.lower() for part in path.parts]
    joined = "/".join(lower_parts)
    for marker in GENERATED_PATH_SEGMENTS:
        marker_lower = marker.lower()
        if "/" in marker_lower:
            if marker_lower in joined:
                return True
            continue
        if marker_lower in lower_parts:
            return True
    return False


def classify_file(path: Path) -> FileFeatures:
    is_test = is_test_path(path)
    is_generated = is_generated_path(path)
    name = path.name
    language = infer_language(path)
    if name in METADATA_FILE_NAMES or name in KNOWN_LOCK_FILES:
        role = FileRole.dependency_metadata
    elif name.lower().startswith("makefile") or name.lower() == "justfile" or _BUILD_FILE_RE.match(name) or _DOCKERFILE_RE.search(name):
        role = FileRole.build_metadata
    elif is_test:
        role = FileRole.test
    elif is_generated:
        role = FileRole.generated
    else:
        suffix = path.suffix.lower()
        if suffix in _DOC_SUFFIXES:
            role = FileRole.docs
        elif suffix in _CONFIG_SUFFIXES:
            role = FileRole.config
        elif suffix in _DATA_SUFFIXES:
            role = FileRole.data
        elif language != "unknown":
            role = FileRole.source
        else:
            role = FileRole.unknown
    return FileFeatures(file_path=path, file_role=role, language=language, is_test=is_test, is_generated=is_generated)


_ACRONYM_BOUNDARY_RE = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_BOUNDARY_RE = re.compile(r"([a-z0-9])([A-Z])")
_SNAKE_KEBAB_RE = re.compile(r"[_\-]+")
_NON_ALNUM_SPLIT_RE = re.compile(r"[^a-z0-9]+")

_CODE_LANGUAGES = frozenset({"java", "kotlin", "scala", "csharp", "typescript", "tsx", "javascript", "jsx", "go", "swift", "python"})


def tokenize_text(text: str) -> list[str]:
    if not text:
        return []
    split_acronyms = _ACRONYM_BOUNDARY_RE.sub(r"\1 \2", text)
    lowered = _CAMEL_BOUNDARY_RE.sub(r"\1 \2", split_acronyms).lower()
    snake_kebab_split = _SNAKE_KEBAB_RE.sub(" ", lowered)
    parts = _NON_ALNUM_SPLIT_RE.split(snake_kebab_split)
    return [part for part in parts if part and len(part) > 1]


def extract_symbols(text: str) -> list[str]:
    symbols = [part for part in tokenize_text(text) if len(part) >= 3 and not part.isdigit()]
    return list(dict.fromkeys(symbols))


@lru_cache(maxsize=8192)
def _cached_path_terms(path_str: str) -> tuple[tuple[str, int], ...]:
    """Top path terms for one file path string; shared by every chunk of the file."""
    return tuple(Counter(tokenize_text(path_str)).most_common(20))


def extract_chunk_terms(chunk_text: str, file_path: Path, language: str) -> list[tuple[str, str, int, float]]:
    records: list[tuple[str, str, int, float]] = []
    chunk_tokens = tokenize_text(chunk_text)
    for term, tf in Counter(chunk_tokens).most_common(80):
        records.append((term, "chunk", int(tf), 1.0))
    for term, tf in _cached_path_terms(str(file_path)):
        records.append((term, "path", int(tf), 0.35))
    if language in _CODE_LANGUAGES:
        # Mirrors Counter(extract_symbols(...)): extract_symbols dedups, so every
        # symbol count is 1 and most_common(30) keeps first-occurrence order.
        seen: set[str] = set()
        symbol_count = 0
        for token in chunk_tokens:
            if len(token) < 3 or token.isdigit() or token in seen:
                continue
            seen.add(token)
            records.append((token, "symbol", 1, 1.4))
            symbol_count += 1
            if symbol_count >= 30:
                break
    return records
