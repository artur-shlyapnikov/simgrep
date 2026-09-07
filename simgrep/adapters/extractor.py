from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


FAST_TEXT_EXTENSIONS = frozenset(
    {
        ".txt",
        ".md",
        ".rst",
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".java",
        ".go",
        ".rs",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".cs",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".scala",
        ".sh",
        ".bash",
        ".zsh",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".xml",
        ".html",
        ".css",
        ".sql",
        ".dockerfile",
    }
)
BINARY_SKIP_EXTENSIONS = frozenset(
    {".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".parquet", ".duckdb", ".sqlite", ".db", ".usearch"}
)
_CONTROL_BYTES = bytes(b for b in range(256) if (b < 32 and b not in (9, 10, 13)) or b == 127)


def _looks_binary(sample: bytes) -> bool:
    if not sample:
        return False
    if b"\x00" in sample:
        return True
    control_count = len(sample) - len(sample.translate(None, _CONTROL_BYTES))
    return control_count / len(sample) > 0.30


def _read_text_bytes(path: Path) -> str:
    raw = path.read_bytes()
    if _looks_binary(raw[:8192]):
        return ""
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


class TextExtractor:
    def extract(self, path: Path) -> str:
        if not path.is_file():
            raise FileNotFoundError(f"File not found or is not a file: {path}")
        suffix = path.suffix.lower()
        if suffix in FAST_TEXT_EXTENSIONS or path.name.lower() == "dockerfile":
            return _read_text_bytes(path)
        if suffix in BINARY_SKIP_EXTENSIONS:
            return ""
        try:
            import unstructured.partition.auto as auto_partition

            elements = auto_partition.partition(filename=str(path), strategy="fast")
            return "\n".join(str(getattr(el, "text")) for el in elements if getattr(el, "text", None))
        except Exception as exc:
            logger.warning("Failed to extract %s with unstructured: %s", path, exc)
            return ""


Extractor = TextExtractor
