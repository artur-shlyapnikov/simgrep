import codecs
import zipfile
from io import BytesIO
from pathlib import Path
from typing import List

import pytest

from simgrep.processor import extract_text_from_file


# Fixtures for creating temporary files and directories
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
    # ISO-8859-1 (Latin-1) content with special characters
    # § (section sign), © (copyright), ± (plus-minus)
    # These are 0xA7, 0xA9, 0xB1 in Latin-1
    content_latin1 = "Sección de prueba © 2024 ±5"
    file_path = tmp_path / "latin1_file.txt"
    with open(file_path, "wb") as f:
        f.write(content_latin1.encode("iso-8859-1"))
    return file_path


@pytest.fixture
def binary_zip_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "binary_file.zip"
    # Create a simple valid zip file in memory
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("dummy.txt", "This is a dummy file inside a zip.")
    # Write the BytesIO buffer to the actual file
    with open(file_path, "wb") as f:
        f.write(zip_buffer.getvalue())
    return file_path


@pytest.fixture
def large_repetitive_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "large_repetitive.txt"
    # Create a 1MB file (approx) with repetitive content
    # 100 chars * 10000 reps = 1,000,000 chars = ~1MB
    pattern = "This is a highly repetitive short sentence that will be repeated many times to make the file large. "
    repetitions = 10000  # (len(pattern) is 100)
    with open(file_path, "w", encoding="utf-8") as f:
        for _ in range(repetitions):
            f.write(pattern)
    return file_path


@pytest.fixture
def utf8_with_bom_file(tmp_path: Path) -> Path:
    file_content = "File with BOM."
    file_path = tmp_path / "bom_file.txt"
    # Write content with UTF-8 BOM
    with open(file_path, "wb") as f:
        f.write(codecs.BOM_UTF8)
        f.write(file_content.encode("utf-8"))
    return file_path


@pytest.fixture
def file_with_unicode_name(tmp_path: Path) -> Path:
    # File name with spaces and unicode characters (e.g., Cyrillic, emoji)
    # Most modern OS and pathlib handle these fine.
    file_name = "тестовый файл 😊 with spaces.txt"
    file_path = tmp_path / file_name
    file_path.write_text("Content of file with unicode name.", encoding="utf-8")
    return file_path


# Tests for extract_text_from_file
class TestExtractTextFromFile:
    def test_extract_from_existing_file(self, temp_text_file: Path):
        expected_content = "Hello World.\nThis is a test file.\nSimgrep is cool."
        assert extract_text_from_file(temp_text_file) == expected_content

    def test_extract_from_empty_file(self, temp_empty_file: Path):
        assert extract_text_from_file(temp_empty_file) == ""

    def test_extract_from_non_existent_file(self, tmp_path: Path):
        non_existent_file = tmp_path / "non_existent.txt"
        with pytest.raises(
            FileNotFoundError,
            match=f"File not found or is not a file: {non_existent_file}",
        ):
            extract_text_from_file(non_existent_file)

    def test_extract_from_directory(self, temp_dir: Path):
        with pytest.raises(
            FileNotFoundError, match=f"File not found or is not a file: {temp_dir}"
        ):
            extract_text_from_file(temp_dir)

    def test_non_utf8_encoded_file(self, non_utf8_file: Path):
        # Expected: unstructured either detects encoding or uses UTF-8 with replacement chars.
        # The important part is that it doesn't crash.
        # "Sección de prueba © 2024 ±5"
        # If read as UTF-8, '§' (0xA7 in Latin-1) might become � or be misinterpreted.
        # If unstructured uses chardet or similar, it might get it right.
        # For this test, we check it doesn't crash and returns *something*.
        # A more precise check would require knowing unstructured's exact behavior.
        try:
            content = extract_text_from_file(non_utf8_file)
            # Check if the core text is somewhat present, possibly with replacement characters
            assert "Secci" in content  # Start of "Sección"
            assert "prueba" in content
            assert "2024" in content
            # If specific characters are replaced, it's acceptable for this test.
            # e.g., content might be "Secci�n de prueba � 2024 �5"
            # print(f"Content from non-UTF-8: '{content}'")
        except RuntimeError as e:
            pytest.fail(f"extract_text_from_file crashed on non-UTF-8 file: {e}")
        except UnicodeDecodeError as e:
            pytest.fail(f"Unhandled UnicodeDecodeError for non-UTF-8 file: {e}")

    @pytest.mark.timeout(10)  # Protect against hangs
    def test_pathological_binary_file_zip(self, binary_zip_file: Path):
        # unstructured might try to parse zip files if it has the capability.
        # We expect it to either extract text (if any parsable parts) or return empty/minimal,
        # or raise RuntimeError, but not hang or crash.
        try:
            content = extract_text_from_file(binary_zip_file)
            # If unstructured has a zip partitioner, it might extract "This is a dummy file inside a zip."
            # If not, it might return empty string or some binary garbage if interpreted as text.
            # The key is no crash and no excessive processing.
            assert isinstance(content, str)
            # print(f"Content from zip: '{content}'")
            # If it extracts the content of dummy.txt:
            # assert "This is a dummy file inside a zip." in content
            # For now, just ensure it runs and returns a string.
        except RuntimeError:
            # This is an acceptable outcome if unstructured explicitly fails on this file type.
            pass
        except Exception as e:
            pytest.fail(
                f"extract_text_from_file failed unexpectedly on binary (zip) file: {e}"
            )

    @pytest.mark.timeout(20)  # Allow more time for larger file
    def test_very_large_repetitive_file(self, large_repetitive_file: Path):
        try:
            content = extract_text_from_file(large_repetitive_file)
            assert isinstance(content, str)
            # Check if the beginning of the pattern is present
            expected_pattern_start = "This is a highly repetitive short sentence"
            assert content.startswith(expected_pattern_start)
            # Check if the file content is roughly the expected size,
            # implying most of it was read.
            # len(pattern) = 100, repetitions = 10000. Total ~1,000,000.
            # unstructured might add newlines between elements, slightly changing length.
            assert len(content) > 900000  # Check it's mostly there
        except RuntimeError as e:
            pytest.fail(f"extract_text_from_file crashed on large repetitive file: {e}")
        except Exception as e:
            pytest.fail(
                f"extract_text_from_file failed unexpectedly on large repetitive file: {e}"
            )

    def test_file_with_unicode_name_handling(self, file_with_unicode_name: Path):
        # This test primarily ensures that `str(file_path)` and its use in
        # `unstructured.partition.auto.partition(filename=str(file_path))`
        # works correctly with unicode filenames.
        try:
            content = extract_text_from_file(file_with_unicode_name)
            assert content == "Content of file with unicode name."
        except Exception as e:
            pytest.fail(f"extract_text_from_file failed on file with unicode name: {e}")

    def test_extract_from_utf8_with_bom_file(self, utf8_with_bom_file: Path):
        # unstructured should handle UTF-8 BOM transparently.
        expected_content = "File with BOM."
        try:
            content = extract_text_from_file(utf8_with_bom_file)
            assert content == expected_content
        except Exception as e:
            pytest.fail(f"extract_text_from_file failed on UTF-8 file with BOM: {e}")

# NOTE: TestChunkTextSimple has been removed as the function `chunk_text_simple`
# no longer exists in `simgrep.processor`. It was replaced by token-based chunking
# (`chunk_text_by_tokens`), which requires different testing strategies.
# New tests should be written for `load_tokenizer` and `chunk_text_by_tokens`.