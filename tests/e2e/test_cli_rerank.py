"""E2E coverage for `simgrep rerank` (standalone cross-encoder candidate scoring) and `search --rerank`.

FakeReranker scores are deterministic by document length: ``len(d) % 7`` over 7.
Fixtures pick file sizes whose residues force the wanted orderings.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, cast

import pytest
from typer.testing import Result

from .conftest import assert_success, run_simgrep_command

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chunk text lengths chosen so FakeReranker scores (len % 7)/7 differ: b ranks first.
_A_TEXT = "x" * 10  # 10 % 7 = 3 -> 0.4286
_B_TEXT = "y" * 11  # 11 % 7 = 4 -> 0.5714

_RERANK_KEYS = ["query", "model", "matches", "files_seen", "chunks_scored", "truncated"]
_MATCH_KEYS = ["file_path", "line_start", "line_end", "score", "snippet"]


@pytest.fixture
def rerank_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Two one-chunk files with distinct FakeReranker length scores."""
    docs = tmp_path / "rerank_docs"
    docs.mkdir()
    (docs / "a.txt").write_text(_A_TEXT, encoding="utf-8")
    (docs / "b.txt").write_text(_B_TEXT, encoding="utf-8")
    return docs


def _run(*args: str, **kwargs: Any) -> Result:
    return run_simgrep_command(list(args), **kwargs)


def _json_payload(result: Result) -> dict[str, Any]:
    assert_success(result)
    return cast(dict[str, Any], json.loads(result.stdout))


def _pair_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    return dict(pairs)


def test_rerank_json_pinned_key_order_and_values(rerank_dir: pathlib.Path, temp_simgrep_home: pathlib.Path) -> None:
    del temp_simgrep_home
    result = _run("rerank", "error handling", str(rerank_dir / "a.txt"), str(rerank_dir / "b.txt"), "--format", "json")
    payload = json.loads(result.stdout, object_pairs_hook=_pair_hook)
    # Byte-shape: pinned top-level key order.
    assert list(payload.keys()) == _RERANK_KEYS
    assert payload["query"] == "error handling"
    assert payload["model"] == DEFAULT_MODEL
    assert payload["files_seen"] == 2
    assert payload["chunks_scored"] == 2
    assert payload["truncated"] is False
    matches = payload["matches"]
    assert len(matches) == 2  # best-per-file collapses to one match per file
    # Ranked descending: b.txt (4/7) beats a.txt (3/7).
    assert [m["file_path"].endswith(name) for m, name in zip(matches, ("b.txt", "a.txt"), strict=True)] == [True, True]
    for match in matches:
        assert list(match.keys()) == _MATCH_KEYS
        assert match["line_start"] == 1  # FakeTokenChunker carries no line metadata
        assert match["line_end"] == 1
        assert len(match["snippet"]) <= 120
    assert matches[0]["score"] > matches[1]["score"]
    assert abs(matches[0]["score"] - 4 / 7) < 1e-6
    assert abs(matches[1]["score"] - 3 / 7) < 1e-6


def test_rerank_jsonl_emits_one_match_record_per_line(rerank_dir: pathlib.Path, temp_simgrep_home: pathlib.Path) -> None:
    del temp_simgrep_home
    result = _run("rerank", "q", str(rerank_dir / "a.txt"), str(rerank_dir / "b.txt"), "--format", "jsonl")
    assert_success(result)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 2
    records = [json.loads(line, object_pairs_hook=_pair_hook) for line in lines]
    for record in records:
        assert list(record.keys()) == _MATCH_KEYS
    assert [record["score"] for record in records] == sorted((record["score"] for record in records), reverse=True)


def test_rerank_rich_lists_ranked_best_chunks(rerank_dir: pathlib.Path, temp_simgrep_home: pathlib.Path) -> None:
    del temp_simgrep_home
    result = _run("rerank", "error handling", str(rerank_dir / "a.txt"), str(rerank_dir / "b.txt"))
    assert_success(result)
    assert "error handling" in result.stdout
    assert "b.txt" in result.stdout
    assert "a.txt" in result.stdout
    # Rank markers appear in best-first order: b's line precedes a's.
    assert result.stdout.index("1.") < result.stdout.index("b.txt") < result.stdout.index("a.txt")


def test_rerank_mixed_readable_and_unreadable_warns_but_exits_zero(rerank_dir: pathlib.Path, tmp_path: pathlib.Path, temp_simgrep_home: pathlib.Path) -> None:
    del temp_simgrep_home
    missing = str(tmp_path / "nope" / "ghost.txt")
    result = _run("rerank", "q", missing, str(rerank_dir / "b.txt"), "--format", "json")
    payload = _json_payload(result)
    assert payload["files_seen"] == 1  # unreadable skipped, readable still ranked
    assert any("b.txt" in m["file_path"] for m in payload["matches"])
    combined = (result.stderr or "") + (result.output or "")
    assert "skipping unreadable" in combined or "Warning" in combined


def test_rerank_zero_readable_files_exits_one(tmp_path: pathlib.Path, temp_simgrep_home: pathlib.Path) -> None:
    del temp_simgrep_home
    result = _run("rerank", "q", str(tmp_path / "does_not_exist.txt"))
    assert result.exit_code == 1
    assert "No readable input files" in (result.stderr or "") + (result.output or "")


def test_rerank_bad_format_is_usage_error_exit_two(rerank_dir: pathlib.Path, temp_simgrep_home: pathlib.Path) -> None:
    del temp_simgrep_home
    result = _run("rerank", "q", str(rerank_dir / "a.txt"), "--format", "yaml")
    assert result.exit_code == 2
    assert "--format must be one of" in (result.stderr or "") + (result.output or "")


def test_rerank_files_from_stdin_matches_direct_paths(rerank_dir: pathlib.Path, temp_simgrep_home: pathlib.Path) -> None:
    del temp_simgrep_home
    direct = _json_payload(_run("rerank", "q", str(rerank_dir / "a.txt"), str(rerank_dir / "b.txt"), "--format", "json"))
    piped = _json_payload(
        _run(
            "rerank",
            "q",
            "--files-from",
            "-",
            "--format",
            "json",
            input_str=f"{rerank_dir / 'a.txt'}\n{rerank_dir / 'b.txt'}\n",
        )
    )
    assert [(m["file_path"], round(m["score"], 6)) for m in piped["matches"]] == [(m["file_path"], round(m["score"], 6)) for m in direct["matches"]]
    assert piped["files_seen"] == 2
    assert piped["chunks_scored"] == 2


def test_rerank_chunk_cap_exceeded_exits_one(rerank_dir: pathlib.Path, temp_simgrep_home: pathlib.Path) -> None:
    del temp_simgrep_home
    result = _run("rerank", "q", str(rerank_dir / "a.txt"), str(rerank_dir / "b.txt"), "--max-chunks", "1")
    assert result.exit_code == 1
    combined = (result.stderr or "") + (result.output or "")
    assert "cap" in combined.lower()
    assert "--max-chunks" in combined


# --- integrated surface: search --rerank -------------------------------------


def _search_payload(*args: str) -> list[dict[str, Any]]:
    result = _run(*args)
    assert_success(result)
    return cast(list[dict[str, Any]], json.loads(result.stdout))


def test_search_rerank_reorders_hits_but_keeps_json_shape(tmp_path: pathlib.Path, temp_simgrep_home: pathlib.Path) -> None:
    del temp_simgrep_home
    docs = tmp_path / "search_rerank_docs"
    docs.mkdir()
    # Scan order follows the fake index's label sort (alphabetical file scan),
    # so a_low.txt leads the plain hybrid list. The reranker reorders purely by
    # len(chunk) % 7: 13 % 7 = 6 (z_high) beats 16 % 7 = 2 (a_low).
    (docs / "a_low.txt").write_text("alpha beta gamma", encoding="utf-8")  # 16 chars -> 2/7
    (docs / "z_high.txt").write_text("delta epsilon", encoding="utf-8")  # 13 chars -> 6/7

    plain = _search_payload("search", "anything", str(docs), "--format", "json")
    assert len(plain) >= 2
    plain_keys = [sorted(item) for item in plain]
    assert pathlib.PurePath(plain[0]["path"]).name == "a_low.txt"  # hybrid order preconditions

    reranked = _search_payload("search", "anything", str(docs), "--format", "json", "--rerank")
    assert sorted(item["path"] for item in reranked) == sorted(item["path"] for item in plain)  # corpus-relative values preserved
    assert [sorted(item) for item in reranked] == plain_keys  # JSON shape unchanged

    # Reranked hits carry FakeReranker cross scores, ranked best-first.
    scores = [item["score"] for item in reranked[:2]]
    assert scores == sorted(scores, reverse=True)
    assert abs(scores[0] - 6 / 7) < 1e-6
    assert reranked[0]["path"].endswith("z_high.txt")

    # And the rerank actually inverted this corpus's plain hybrid order.
    assert [pathlib.PurePath(item["path"]).name for item in plain[:2]] != [pathlib.PurePath(item["path"]).name for item in reranked[:2]]


def test_search_plain_run_stays_default_without_rerank_flag(tmp_path: pathlib.Path, temp_simgrep_home: pathlib.Path) -> None:
    del temp_simgrep_home
    docs = tmp_path / "plain_search_docs"
    docs.mkdir()
    (docs / "only.txt").write_text("stable text body", encoding="utf-8")
    payload = _search_payload("search", "text", str(docs), "--format", "json")
    assert payload  # no --rerank flag: plain pipeline untouched, still returns hits


def test_rerank_missing_files_from_path_file_exits_one_without_traceback(tmp_path: pathlib.Path, temp_simgrep_home: pathlib.Path) -> None:
    del temp_simgrep_home
    missing = str(tmp_path / "no_such_paths.txt")
    result = _run("rerank", "q", "--files-from", missing, str(tmp_path / "a.txt"))
    assert result.exit_code == 1
    combined = (result.stderr or "") + (result.output or "")
    assert "Cannot read --files-from paths file" in combined
    assert missing in combined
    assert "Traceback" not in combined
