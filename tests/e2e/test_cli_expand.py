"""E2E coverage for `simgrep expand` and `search --whole-unit`.

Self-contained like test_cli_related.py: deterministic token-set hashing
embedder + local runtime patching, no shared fixtures and no model downloads.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pytest
from typer.testing import CliRunner, Result

from simgrep.main import app as cli_app
from simgrep.models import Chunk

try:
    runner = CliRunner(mix_stderr=False)  # type: ignore[call-arg]
except TypeError:
    runner = CliRunner()


def run_simgrep_command(args: Sequence[str], cwd: pathlib.Path | None = None) -> Result:
    """In-process CLI invocation with a wide terminal for stable wrapping."""
    original_cwd = pathlib.Path.cwd()
    try:
        if cwd is not None:
            os.chdir(cwd)
        return runner.invoke(cli_app, list(args), env={"COLUMNS": "200"})
    finally:
        if cwd is not None:
            os.chdir(original_cwd)


class TokenSetHashingEmbedder:
    """Unit vector per token, summed and normalized: shared vocab => ~0.9."""

    ndim = 256

    def encode(self, texts: list[str], *, is_query: bool = False, batch_size: int | None = None) -> np.ndarray:
        del is_query, batch_size
        vectors = np.zeros((len(texts), self.ndim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                digest = hashlib.md5(token.encode("utf-8")).digest()
                rng = np.random.default_rng(int.from_bytes(digest[:4], "little"))
                vector = rng.standard_normal(self.ndim).astype(np.float32)
                vectors[row] += vector / float(np.linalg.norm(vector))
            norm = float(np.linalg.norm(vectors[row]))
            if norm > 0:
                vectors[row] /= norm
        return vectors


class FakeTextExtractor:
    def extract(self, path: pathlib.Path) -> str:
        sample = path.read_bytes()[:8192]
        if b"\x00" in sample:
            return ""
        return path.read_text(encoding="utf-8")


class WindowChunker:
    """Fixed-width char windows snapped to newline boundaries: multi-chunk units."""

    def __init__(self, size: int = 160) -> None:
        self.size = size

    def chunk(self, text: str) -> Sequence[Chunk]:
        chunks: list[Chunk] = []
        pos = 0
        while pos < len(text):
            end = min(pos + self.size, len(text))
            if end < len(text):
                nl = text.rfind("\n", pos + 1, end)
                if nl > pos:
                    end = nl
            piece = text[pos:end]
            if piece.strip():
                chunks.append(
                    Chunk(
                        id=-1,
                        file_id=-1,
                        text=piece,
                        start=pos,
                        end=end,
                        tokens=max(1, len(piece.split())),
                    )
                )
            pos = end
        return chunks


@dataclass(frozen=True)
class _VectorHit:
    label: int
    score: float


class RoundTripVectorIndex:
    """True cosine ranking over stored vectors."""

    def __init__(self, ndim: int = 256) -> None:
        self.ndim = ndim
        self.data: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.data)

    def add(
        self,
        labels: np.ndarray | None = None,
        vectors: np.ndarray | None = None,
        *,
        keys: np.ndarray | None = None,
        vecs: np.ndarray | None = None,
    ) -> None:
        actual_labels = labels if labels is not None else keys
        actual_vectors = vectors if vectors is not None else vecs
        assert actual_labels is not None
        assert actual_vectors is not None
        for label, vector in zip(actual_labels, actual_vectors):
            self.data[int(label)] = np.asarray(vector, dtype=np.float32)

    def remove(self, labels: np.ndarray | None = None, *, keys: np.ndarray | None = None) -> None:
        actual = labels if labels is not None else keys
        if actual is None:
            return
        for label in np.asarray(actual).tolist():
            self.data.pop(int(label), None)

    def search(self, vector: np.ndarray, k: int) -> list[_VectorHit]:
        if not self.data:
            return []
        keys = self.keys
        matrix = self.vectors(keys)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        query = np.asarray(vector, dtype=np.float32).ravel()
        qnorm = float(np.linalg.norm(query))
        if qnorm > 0:
            query = query / qnorm
        sims = (matrix / norms) @ query
        order = np.argsort(-sims)[:k]
        return [_VectorHit(label=int(keys[i]), score=float(sims[i])) for i in order]

    def save(self, path: pathlib.Path) -> None:
        payload = {str(label): vector.tolist() for label, vector in sorted(self.data.items())}
        path.write_text(json.dumps(payload), encoding="utf-8")

    def load(self, path: pathlib.Path) -> None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.data = {int(label): np.asarray(vector, dtype=np.float32) for label, vector in raw.items()}

    @property
    def keys(self) -> np.ndarray:
        return np.array(sorted(self.data), dtype=np.int64)

    def vectors(self, keys: np.ndarray | None = None) -> np.ndarray:
        actual_keys = self.keys if keys is None else np.asarray(keys, dtype=np.int64)
        rows = [self.data[int(key)] for key in actual_keys.tolist()]
        return np.stack(rows).astype(np.float32, copy=False)


class _HashingRuntime:
    def __init__(self) -> None:
        self.extractor = FakeTextExtractor()
        self.chunker = WindowChunker()
        self.embedder = TokenSetHashingEmbedder()

    def require_bulk(self) -> None:
        """No-op: the token-set hashing embedder is already bulk-friendly."""

    def new_vector_index(self, ndim: int) -> RoundTripVectorIndex:
        return RoundTripVectorIndex(ndim)


@pytest.fixture(autouse=True)
def hashing_runtime_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch RuntimeFactory on simgrep.main directly (self-sufficient, foreign-safe)."""
    runtime = _HashingRuntime()

    class _Factory:
        def for_app(self, config: object) -> _HashingRuntime:
            del config
            return runtime

        def for_project(self, config: object) -> _HashingRuntime:
            del config
            return runtime

    monkeypatch.setattr("simgrep.execution.RuntimeFactory", _Factory)


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep simgrep config/state inside tmp_path (mirrors e2e conftest isolation)."""
    home_dir = tmp_path / "simgrep_e2e_home"
    home_dir.mkdir()
    original_expanduser = os.path.expanduser

    def mock_expanduser(path_str: str) -> str:
        if path_str == "~" or path_str.startswith("~/"):
            return path_str.replace("~", str(home_dir), 1)
        return original_expanduser(path_str)

    monkeypatch.setattr(os.path, "expanduser", mock_expanduser)


_PAYMENT_FN = (
    "def charge_order(order_id: str, amount_cents: int) -> bool:\n"
    '    """Charge the gateway once per order."""\n'
    "    ledger = open_ledger()\n"
    "    if amount_cents <= 0:\n"
    "        raise ValueError('nonpositive charge')\n"
    "    receipt = ledger.charge(order_id, amount_cents)\n"
    "    audit_note(order_id, amount_cents)\n"
    "    return receipt.ok\n"
    "\n"
    "\n"
    "def unrelated_tail():\n"
    "    return 42\n"
)


@pytest.fixture
def py_file(tmp_path: pathlib.Path) -> pathlib.Path:
    target = tmp_path / "payments.py"
    target.write_text(_PAYMENT_FN, encoding="utf-8")
    return target


def _expand(path: pathlib.Path, line: int, *extra: str) -> Result:
    return run_simgrep_command(["expand", str(path), str(line), *extra])


# --- rich -------------------------------------------------------------------


def test_rich_header_and_gutter(py_file: pathlib.Path) -> None:
    result = _expand(py_file, 7)
    assert result.exit_code == 0
    # Unit is the whole function: lines 1-8 (through `return receipt.ok`).
    assert f"{py_file}:1-8 (dedent, 8 lines)" in result.stdout
    assert re.search(r"^ +7 +audit_note", result.stdout, re.MULTILINE)


def test_rich_truncation_marker_counts_hidden_lines(py_file: pathlib.Path) -> None:
    result = _expand(py_file, 2, "--max-chars", "80")
    assert result.exit_code == 0
    match = re.search(r"\[\+(\d+) more lines\]", result.stdout)
    assert match is not None
    assert int(match.group(1)) >= 1
    # Header still reports the FULL unit span.
    assert "(dedent, 8 lines)" in result.stdout


# --- json -------------------------------------------------------------------

_JSON_KEYS = {"path", "start_line", "end_line", "start_char", "end_char", "language", "family", "text", "truncated"}


def test_json_payload_has_exact_keys(py_file: pathlib.Path) -> None:
    result = _expand(py_file, 3, "--format", "json")
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert set(payload) == _JSON_KEYS


def test_json_values_span_the_whole_function(py_file: pathlib.Path) -> None:
    result = _expand(py_file, 3, "--format", "json")
    payload = json.loads(result.stdout)
    assert payload["start_line"] == 1
    assert payload["end_line"] == 8
    assert payload["start_char"] == 0
    assert payload["family"] == "dedent"
    assert payload["language"] == "python"
    assert payload["truncated"] is False
    assert payload["text"].startswith("def charge_order(")
    # Engine bounds exclude the unit's trailing newline.
    assert payload["text"].endswith("return receipt.ok")


def test_json_truncation_flags_and_marks_text(py_file: pathlib.Path) -> None:
    result = _expand(py_file, 3, "--format", "json", "--max-chars", "60")
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["truncated"] is True
    assert payload["text"].endswith("...")
    # Bounds still describe the full unit.
    assert payload["end_line"] == 8


# --- text -------------------------------------------------------------------


def test_text_format_is_raw_unit_without_decoration(py_file: pathlib.Path) -> None:
    result = _expand(py_file, 2, "--format", "text")
    assert result.exit_code == 0
    assert result.stdout == _PAYMENT_FN[: _PAYMENT_FN.index("\n\n\n") + 1]


def test_text_format_appends_ellipsis_when_capped(py_file: pathlib.Path) -> None:
    result = _expand(py_file, 2, "--format", "text", "--max-chars", "60")
    assert result.exit_code == 0
    assert result.stdout.endswith("...\n")


# --- errors -----------------------------------------------------------------


def test_line_out_of_range_exits_two_with_exact_hint(py_file: pathlib.Path) -> None:
    result = _expand(py_file, 99)
    assert result.exit_code == 2
    assert "line 99 out of range" in result.stderr
    assert f"Hint: file has {len(_PAYMENT_FN.splitlines())} lines" in result.stderr


def test_missing_file_exits_two(tmp_path: pathlib.Path) -> None:
    result = run_simgrep_command(["expand", str(tmp_path / "nope.py"), "1"])
    assert result.exit_code == 2


def test_unreadable_file_exits_one(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "locked.py"
    target.write_text("def locked():\n    pass\n", encoding="utf-8")

    def deny(self: pathlib.Path) -> bytes:
        raise PermissionError(13, "Permission denied")

    # Click's PATH validation already rejects chmod-0 files with exit 2; drive
    # the runtime read failure instead (read_text_raw maps OSError to exit 1).
    monkeypatch.setattr(pathlib.Path, "read_bytes", deny)
    result = _expand(target, 1)
    assert result.exit_code == 1


def test_invalid_language_exit_two(py_file: pathlib.Path) -> None:
    result = _expand(py_file, 1, "--language", "cobol")
    assert result.exit_code == 2


def test_invalid_format_exit_two(py_file: pathlib.Path) -> None:
    result = _expand(py_file, 1, "--format", "yaml")
    assert result.exit_code == 2


def test_line_zero_rejected_by_signature(py_file: pathlib.Path) -> None:
    result = _expand(py_file, 0)
    assert result.exit_code != 0


# --- family behavior --------------------------------------------------------


def test_language_override_switches_family(py_file: pathlib.Path) -> None:
    # Paragraph family treats the function as one blank-line-delimited block.
    result = _expand(py_file, 5, "--language", "paragraph", "--format", "json")
    payload = json.loads(result.stdout)
    assert payload["family"] == "paragraph"
    assert payload["start_line"] == 1
    assert payload["end_line"] == 8  # blank lines below bound the block


def test_brace_family_expands_to_matching_brace(tmp_path: pathlib.Path) -> None:
    c_file = tmp_path / "ledger.c"
    body = "int charge(int amount) {\n" "    if (amount <= 0) {\n" "        return -1;\n" "    }\n" "    return log_charge(amount);\n" "}\n"
    c_file.write_text(body, encoding="utf-8")
    result = _expand(c_file, 5, "--format", "json")  # line 5 sits directly in the function body
    payload = json.loads(result.stdout)
    assert payload["family"] == "brace"
    assert payload["start_line"] == 1
    assert payload["end_line"] == 6
    assert payload["text"].endswith("}")


def test_deterministic_output_across_runs(py_file: pathlib.Path) -> None:
    first = _expand(py_file, 4, "--format", "json").stdout
    second = _expand(py_file, 4, "--format", "json").stdout
    assert first == second


def test_crlf_file_expands_with_raw_newlines(tmp_path: pathlib.Path) -> None:
    crlf = tmp_path / "crlf.py"
    crlf.write_bytes(b"def win():\r\n    return 7\r\n")
    result = _expand(crlf, 2, "--format", "json")
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert (payload["start_line"], payload["end_line"]) == (1, 2)
    assert "\r\n" in payload["text"]


def test_max_chars_honored_on_single_line_file(tmp_path: pathlib.Path) -> None:
    tiny = tmp_path / "minified.json"
    tiny.write_text('{"k":"' + "v" * 496 + '"}', encoding="utf-8")
    result = _expand(tiny, 1, "--format", "json", "--max-chars", "100")
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["truncated"] is True
    assert len(payload["text"]) <= 103  # 100-char budget plus the '...' marker
    rich = _expand(tiny, 1, "--max-chars", "100")
    match = re.search(r"\[\+(\d+) more lines\]", rich.stdout)
    assert match is not None and int(match.group(1)) >= 1


def test_latin1_file_falls_back_and_expands(tmp_path: pathlib.Path) -> None:
    raw = tmp_path / "legacy.py"
    raw.write_bytes(b"caf\xe9 = brew()\nprint(caf\xe9)\n")  # invalid utf-8, valid latin-1
    result = _expand(raw, 2, "--format", "json")
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "caf\xe9" in payload["text"]


def test_lone_cr_file_counts_lines_like_offsets(tmp_path: pathlib.Path) -> None:
    # splitlines() treats lone CR as a break but the offset model does not:
    # the guard must agree with offsets, never crash with IndexError.
    lonely = tmp_path / "lonely.txt"
    lonely.write_bytes(b"a\rb\rc\n")
    result = _expand(lonely, 2, "--format", "text")
    assert result.exit_code == 2
    assert "Hint: file has 1 lines" in result.stderr


# --- search --whole-unit ----------------------------------------------------

_LINES = [f"    value_{i} = transmogrify_value_{i}()" for i in range(1, 31)]
_BIG_FN = "def big_routine():\n" + "\n".join(_LINES) + "\n"


@pytest.fixture
def unit_repo(tmp_path: pathlib.Path) -> pathlib.Path:
    repo = tmp_path / "proj"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "big.py").write_text(_BIG_FN, encoding="utf-8")
    return repo


def _search(repo: pathlib.Path, *extra: str) -> Result:
    return run_simgrep_command(["search", "transmogrify_value_25", str(repo / "src"), "--top", "1", "--format", "json", *extra])


def test_search_whole_unit_spans_multi_chunk_function(unit_repo: pathlib.Path) -> None:
    plain = json.loads(_search(unit_repo).stdout)[0]
    assert plain["line_start"] > 1  # chunk window starts mid-function
    expanded = json.loads(_search(unit_repo, "--whole-unit").stdout)[0]
    total_lines = len(_BIG_FN.splitlines())
    assert expanded["line_start"] == 1
    assert expanded["line_end"] == total_lines
    assert expanded["text"].startswith("def big_routine():")
    # Identity fields survive the transform.
    assert expanded["score"] == plain["score"]
    assert expanded["path"] == plain["path"]


def test_search_whole_unit_is_harmless_with_paths_format(unit_repo: pathlib.Path) -> None:
    paths = run_simgrep_command(["search", "transmogrify_value_25", str(unit_repo / "src"), "--format", "paths"])
    paths_wu = run_simgrep_command(["search", "transmogrify_value_25", str(unit_repo / "src"), "--format", "paths", "--whole-unit"])
    assert paths.exit_code == paths_wu.exit_code == 0
    assert "big.py" in paths.stdout
    assert paths_wu.stdout.count("big.py") == paths.stdout.count("big.py")


def test_search_whole_unit_deleted_file_degrades_stale(unit_repo: pathlib.Path) -> None:
    assert run_simgrep_command(["init"], cwd=unit_repo).exit_code == 0
    add = run_simgrep_command(["project", "add-path", str(unit_repo / "src")], cwd=unit_repo)
    assert add.exit_code == 0, add.stderr or add.stdout
    index = run_simgrep_command(["index", "--rebuild"], cwd=unit_repo)
    assert index.exit_code == 0, index.stderr or index.stdout
    (unit_repo / "src" / "big.py").unlink()

    result = run_simgrep_command(
        [
            "search",
            "transmogrify_value_25",
            str(unit_repo / "src"),
            "--freshness",
            "skip",
            "--whole-unit",
            "--context",
            "1",
            "--format",
            "json",
            "--top",
            "1",
        ],
        cwd=unit_repo,
    )
    assert result.exit_code == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    assert payload, "expected the stale hit to survive freshness skip"
    assert all(entry["stale_offsets"] is True for entry in payload)
