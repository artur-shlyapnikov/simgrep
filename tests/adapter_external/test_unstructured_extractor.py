import codecs
from pathlib import Path

import pytest

from simgrep.adapters.extractor import TextExtractor

pytestmark = pytest.mark.external


@pytest.fixture
def temp_text_file(tmp_path: Path) -> Path:
    file = tmp_path / "test_file.txt"
    file.write_text("Hello World.\nThis is a test file.\nSimgrep is cool.", encoding="utf-8")
    return file


@pytest.fixture
def temp_empty_file(tmp_path: Path) -> Path:
    file = tmp_path / "empty_file.txt"
    file.write_text("", encoding="utf-8")
    return file


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    dir_path = tmp_path / "test_dir"
    dir_path.mkdir()
    return dir_path


@pytest.fixture
def non_utf8_file(tmp_path: Path) -> Path:
    content_latin1 = "Sección de prueba © 2024 ±5"
    file_path = tmp_path / "latin1_file.txt"
    with open(file_path, "wb") as f:
        f.write(content_latin1.encode("iso-8859-1"))
    return file_path


@pytest.fixture
def utf8_with_bom_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "bom_file.txt"
    with open(file_path, "wb") as f:
        f.write(codecs.BOM_UTF8)
        f.write("File with BOM.".encode("utf-8"))
    return file_path


@pytest.fixture
def file_with_unicode_name(tmp_path: Path) -> Path:
    file_path = tmp_path / "тестовый файл 😊 with spaces.txt"
    file_path.write_text("Content of file with unicode name.", encoding="utf-8")
    return file_path


@pytest.fixture
def text_extractor() -> TextExtractor:
    return TextExtractor()


class TestTextExtractor:
    def test_extract_from_existing_file(self, text_extractor: TextExtractor, temp_text_file: Path) -> None:
        assert text_extractor.extract(temp_text_file) == "Hello World.\nThis is a test file.\nSimgrep is cool."

    def test_extract_from_empty_file(self, text_extractor: TextExtractor, temp_empty_file: Path) -> None:
        assert text_extractor.extract(temp_empty_file) == ""

    def test_extract_from_non_existent_file(self, text_extractor: TextExtractor, tmp_path: Path) -> None:
        non_existent_file = tmp_path / "non_existent.txt"
        with pytest.raises(FileNotFoundError, match=f"File not found or is not a file: {non_existent_file}"):
            text_extractor.extract(non_existent_file)

    def test_extract_from_directory(self, text_extractor: TextExtractor, temp_dir: Path) -> None:
        with pytest.raises(FileNotFoundError, match=f"File not found or is not a file: {temp_dir}"):
            text_extractor.extract(temp_dir)

    def test_non_utf8_encoded_file(self, text_extractor: TextExtractor, non_utf8_file: Path) -> None:
        content = text_extractor.extract(non_utf8_file)
        assert "Sección de prueba" in content
        assert "2024" in content

    def test_file_with_unicode_name_handling(self, text_extractor: TextExtractor, file_with_unicode_name: Path) -> None:
        content = text_extractor.extract(file_with_unicode_name)
        assert content == "Content of file with unicode name."

    def test_extract_from_utf8_with_bom_file(self, text_extractor: TextExtractor, utf8_with_bom_file: Path) -> None:
        content = text_extractor.extract(utf8_with_bom_file)
        assert content == "File with BOM."
