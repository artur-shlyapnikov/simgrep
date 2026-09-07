from __future__ import annotations

import builtins
import types
from collections.abc import Sequence
from pathlib import Path

import pytest

from simgrep.adapters.extractor import TextExtractor


def test_utf8_bom_is_removed(tmp_path: Path) -> None:
    file_path = tmp_path / "bom.txt"
    file_path.write_bytes(b"\xef\xbb\xbfHello BOM")

    text = TextExtractor().extract(file_path)

    assert text == "Hello BOM"


def test_latin1_fallback_reads_text_without_crash(tmp_path: Path) -> None:
    file_path = tmp_path / "latin1.txt"
    expected = "Seccion con simbolo: \xa9"
    file_path.write_bytes(expected.encode("latin-1"))

    text = TextExtractor().extract(file_path)

    assert text == expected


def test_binary_magic_detection_returns_empty_string(tmp_path: Path) -> None:
    file_path = tmp_path / "looks_text.txt"
    file_path.write_bytes(b"abc\x00def")

    text = TextExtractor().extract(file_path)

    assert text == ""


def test_known_binary_extension_skips_without_unstructured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    file_path = tmp_path / "image.png"
    file_path.write_bytes(b"not-an-image")

    original_import = builtins.__import__

    def fail_if_unstructured_imported(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: Sequence[str] | None = None,
        level: int = 0,
    ) -> object:
        if name == "unstructured.partition.auto":
            raise AssertionError("unstructured should not be imported for known binary extension")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_if_unstructured_imported)

    text = TextExtractor().extract(file_path)

    assert text == ""


def test_fast_text_extension_does_not_import_or_call_unstructured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    file_path = tmp_path / "main.py"
    file_path.write_text("print('ok')", encoding="utf-8")

    original_import = builtins.__import__

    def fail_if_unstructured_imported(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: Sequence[str] | None = None,
        level: int = 0,
    ) -> object:
        if name == "unstructured.partition.auto":
            raise AssertionError("unstructured should not be imported for fast text extension")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_if_unstructured_imported)

    text = TextExtractor().extract(file_path)

    assert text == "print('ok')"


def test_unstructured_fallback_is_used_for_unknown_text_extension(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    file_path = tmp_path / "notes.unknown"
    file_path.write_text("ignored direct read path", encoding="utf-8")

    auto_module = types.ModuleType("unstructured.partition.auto")
    call_state = {"called": False}

    def fake_partition(*, filename: str, strategy: str) -> list[types.SimpleNamespace]:
        call_state["called"] = True
        assert filename == str(file_path)
        assert strategy == "fast"
        return [types.SimpleNamespace(text="first"), types.SimpleNamespace(text="second")]

    auto_module.partition = fake_partition  # type: ignore[attr-defined]

    unstructured_module = types.ModuleType("unstructured")
    partition_module = types.ModuleType("unstructured.partition")
    monkeypatch.setitem(__import__("sys").modules, "unstructured", unstructured_module)
    monkeypatch.setitem(__import__("sys").modules, "unstructured.partition", partition_module)
    monkeypatch.setitem(__import__("sys").modules, "unstructured.partition.auto", auto_module)

    text = TextExtractor().extract(file_path)

    assert call_state["called"] is True
    assert text == "first\nsecond"


def test_exception_inside_unstructured_fallback_returns_empty_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    file_path = tmp_path / "notes.unknown"
    file_path.write_text("content", encoding="utf-8")

    auto_module = types.ModuleType("unstructured.partition.auto")

    def fake_partition(*, filename: str, strategy: str) -> list[types.SimpleNamespace]:
        raise RuntimeError("boom")

    auto_module.partition = fake_partition  # type: ignore[attr-defined]

    unstructured_module = types.ModuleType("unstructured")
    partition_module = types.ModuleType("unstructured.partition")
    monkeypatch.setitem(__import__("sys").modules, "unstructured", unstructured_module)
    monkeypatch.setitem(__import__("sys").modules, "unstructured.partition", partition_module)
    monkeypatch.setitem(__import__("sys").modules, "unstructured.partition.auto", auto_module)

    text = TextExtractor().extract(file_path)

    assert text == ""


def test_directory_and_missing_path_raise_clear_filenotfound(tmp_path: Path) -> None:
    extractor = TextExtractor()
    missing = tmp_path / "missing.txt"
    directory = tmp_path / "folder"
    directory.mkdir()

    with pytest.raises(FileNotFoundError, match=f"File not found or is not a file: {missing}"):
        extractor.extract(missing)

    with pytest.raises(FileNotFoundError, match=f"File not found or is not a file: {directory}"):
        extractor.extract(directory)


def test_control_character_heuristics_drive_binary_detection(tmp_path: Path) -> None:
    extractor = TextExtractor()
    (tmp_path / "empty.txt").write_bytes(b"")

    assert extractor.extract(tmp_path / "empty.txt") == ""

    ctrl_heavy = bytes([1, 2, 3, 4, 5]) * 200 + b"plain text padding plain text padding"
    (tmp_path / "ctrl.txt").write_bytes(ctrl_heavy)

    assert extractor.extract(tmp_path / "ctrl.txt") == ""


def test_looks_binary_contract_boundaries() -> None:
    from simgrep.adapters.extractor import _looks_binary

    assert _looks_binary(b"") is False
    assert _looks_binary(b"hello\x00world") is True
    assert _looks_binary(b"a\tb\nc\rd") is False
    assert _looks_binary(bytes(range(32, 127)) * 100) is False
    assert _looks_binary(b"\xe9\xe8\xe7" * 100) is False
    assert _looks_binary(b"\x01" * 30 + b"a" * 70) is False
    assert _looks_binary(b"\x01" * 31 + b"a" * 69) is True
    assert _looks_binary(b"\x7f" * 31 + b"a" * 69) is True
