import codecs
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from simgrep.adapters.unstructured_extractor import UnstructuredExtractor

pytest.mark.external


@pytest.fixture
def temp_text_file(tmp_path: Path) -> Path:
    file_content = "Hello World.\nThis is a test file.\nSimgrep is cool."
    file = tmp_path / "test_file.txt"
    file.write_text(file_content, encoding="utf-8")
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
def binary_zip_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "binary_file.zip"
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("dummy.txt", "This is a dummy file inside a zip.")
    with open(file_path, "wb") as f:
        f.write(zip_buffer.getvalue())
    return file_path


@pytest.fixture
def large_repetitive_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "large_repetitive.txt"
    pattern = "This is a highly repetitive short sentence that will be repeated many times to make the file large. "
    repetitions = 10000
    with open(file_path, "w", encoding="utf-8") as f:
        for _ in range(repetitions):
            f.write(pattern)
    return file_path


@pytest.fixture
def utf8_with_bom_file(tmp_path: Path) -> Path:
    file_content = "File with BOM."
    file_path = tmp_path / "bom_file.txt"
    with open(file_path, "wb") as f:
        f.write(codecs.BOM_UTF8)
        f.write(file_content.encode("utf-8"))
    return file_path


@pytest.fixture
def file_with_unicode_name(tmp_path: Path) -> Path:
    file_name = "тестовый файл 😊 with spaces.txt"
    file_path = tmp_path / file_name
    file_path.write_text("Content of file with unicode name.", encoding="utf-8")
    return file_path


class TestUnstructuredExtractor:
    def test_extract_from_existing_file(self, unstructured_extractor: UnstructuredExtractor, temp_text_file: Path) -> None:
        extractor = unstructured_extractor
        expected_content = "Hello World.\nThis is a test file.\nSimgrep is cool."
        assert extractor.extract(temp_text_file) == expected_content

    def test_extract_from_empty_file(self, unstructured_extractor: UnstructuredExtractor, temp_empty_file: Path) -> None:
        assert unstructured_extractor.extract(temp_empty_file) == ""

    def test_extract_from_non_existent_file(self, unstructured_extractor: UnstructuredExtractor, tmp_path: Path) -> None:
        non_existent_file = tmp_path / "non_existent.txt"
        with pytest.raises(
            FileNotFoundError,
            match=f"File not found or is not a file: {non_existent_file}",
        ):
            unstructured_extractor.extract(non_existent_file)

    def test_extract_from_directory(self, unstructured_extractor: UnstructuredExtractor, temp_dir: Path) -> None:
        with pytest.raises(FileNotFoundError, match=f"File not found or is not a file: {temp_dir}"):
            unstructured_extractor.extract(temp_dir)

    def test_non_utf8_encoded_file(self, unstructured_extractor: UnstructuredExtractor, non_utf8_file: Path) -> None:
        # Unstructured handles various encodings automatically
        content = unstructured_extractor.extract(non_utf8_file)
        assert "Sección de prueba" in content
        assert "2024" in content

    @pytest.mark.timeout(10)
    def test_pathological_binary_file_zip(self, unstructured_extractor: UnstructuredExtractor, binary_zip_file: Path) -> None:
        # Unstructured might extract metadata or nothing from a zip. We expect an empty string.
        content = unstructured_extractor.extract(binary_zip_file)
        assert content == ""
        assert isinstance(content, str)

    @pytest.mark.timeout(20)
    def test_very_large_repetitive_file(self, unstructured_extractor: UnstructuredExtractor, large_repetitive_file: Path) -> None:
        content = unstructured_extractor.extract(large_repetitive_file)
        assert isinstance(content, str)
        expected_pattern_start = "This is a highly repetitive short sentence"
        assert content.startswith(expected_pattern_start)
        assert len(content) > 900000

    def test_file_with_unicode_name_handling(self, unstructured_extractor: UnstructuredExtractor, file_with_unicode_name: Path) -> None:
        content = unstructured_extractor.extract(file_with_unicode_name)
        assert content == "Content of file with unicode name."

    def test_extract_from_utf8_with_bom_file(self, unstructured_extractor: UnstructuredExtractor, utf8_with_bom_file: Path) -> None:
        expected_content = "File with BOM."
        content = unstructured_extractor.extract(utf8_with_bom_file)
        assert content == expected_content
