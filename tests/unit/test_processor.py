from pathlib import Path
from typing import List

import pytest

from simgrep.processor import chunk_text_simple, extract_text_from_file


# Fixtures for creating temporary files and directories
@pytest.fixture
def temp_text_file(tmp_path: Path) -> Path:
    file_content = "Hello World.\nThis is a test file.\nSimgrep is cool."
    file = tmp_path / "test_file.txt"
    file.write_text(file_content)
    return file


@pytest.fixture
def temp_empty_file(tmp_path: Path) -> Path:
    file = tmp_path / "empty_file.txt"
    file.write_text("")
    return file


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    dir_path = tmp_path / "test_dir"
    dir_path.mkdir()
    return dir_path


# Tests for extract_text_from_file
class TestExtractTextFromFile:
    def test_extract_from_existing_file(self, temp_text_file: Path):
        expected_content = "Hello World.\nThis is a test file.\nSimgrep is cool."
        # Unstructured might add an extra newline between elements if they are seen as separate blocks.
        # For simple text, it usually preserves it or joins paragraphs.
        # Let's assume it preserves simple newlines from typical .txt.
        # If elements are [Paragraph("Hello World."), Paragraph("This is a test file."), Paragraph("Simgrep is cool.")]
        # and they are joined by "\n", the result is as expected.
        assert extract_text_from_file(temp_text_file) == expected_content

    def test_extract_from_empty_file(self, temp_empty_file: Path):
        # If the file is empty, unstructured partition might return an empty list of elements,
        # or elements with empty text. Joining them should result in an empty string.
        assert extract_text_from_file(temp_empty_file) == ""

    def test_extract_from_non_existent_file(self, tmp_path: Path):
        non_existent_file = tmp_path / "non_existent.txt"
        with pytest.raises(
            FileNotFoundError,
            match=f"File not found or is not a file: {non_existent_file}",
        ):
            extract_text_from_file(non_existent_file)

    def test_extract_from_directory(self, temp_dir: Path):
        with pytest.raises(FileNotFoundError, match=f"File not found or is not a file: {temp_dir}"):
            extract_text_from_file(temp_dir)

    # Consider mocking unstructured.partition.auto.partition for error case if needed,
    # but for now, focusing on file system interactions.
    # def test_unstructured_processing_error(self, temp_text_file: Path, mocker):
    #     mocker.patch("unstructured.partition.auto.partition", side_effect=Exception("Test Unstructured Error"))
    #     with pytest.raises(RuntimeError, match=f"Failed to extract text from {temp_text_file}"):
    #         extract_text_from_file(temp_text_file)


# Tests for chunk_text_simple
class TestChunkTextSimple:
    @pytest.mark.parametrize(
        "text, chunk_size, overlap, expected_chunks",
        [
            # Basic case
            (
                "abcdefghijklmnopqrstuvwxyz",
                10,
                3,
                [
                    "abcdefghij",
                    "hijklmnopq",
                    "opqrstuvwx",
                    "vwxyz",
                ],  # Corrected last chunk
            ),
            # Text shorter than chunk size
            ("abc", 10, 3, ["abc"]),
            # Exact multiple of (chunk_size - overlap) after first chunk
            (
                "abcdefghijklmno",
                10,
                5,
                ["abcdefghij", "fghijklmno"],
            ),  # 15 chars, step 5. 0-9, 5-14.
            # Last chunk smaller
            (
                "abcdefghijkl",
                10,
                5,
                ["abcdefghij", "fghijkl"],
            ),  # 12 chars, step 5. 0-9, 5-11
            # Zero overlap
            (
                "abcdefghijklm",
                5,
                0,
                ["abcde", "fghij", "klm"],
            ),  # 13 chars, step 5. 0-4, 5-9, 10-12
            # Empty text
            ("", 10, 3, []),
            # Overlap makes next chunk start beyond text
            ("abcdefghij", 10, 0, ["abcdefghij"]),
            # Overlap makes next chunk start exactly at end (no more chunks)
            (
                "abcdefghij",
                5,
                2,
                ["abcde", "defgh", "ghij"],  # Corrected last chunk
            ),  # text len 10, chunk 5, overlap 2, step 3
            ("12345", 5, 0, ["12345"]),
            (
                "1234567890",
                5,
                1,
                ["12345", "56789", "90"],
            ),  # Chunks: 0-4, 4-8, 8-9 (Corrected: 8-12 -> "90")
            (
                "1234567890",
                5,
                4,
                ["12345", "23456", "34567", "45678", "56789", "67890"],
            ),  # Step 1
        ],
    )
    def test_chunking_logic(self, text: str, chunk_size: int, overlap: int, expected_chunks: List[str]):
        assert chunk_text_simple(text, chunk_size, overlap) == expected_chunks

    @pytest.mark.parametrize(
        "text, chunk_size, overlap, expected_chunks_manual_check",
        [
            (
                "Hello world. This is a test.",
                20,
                5,
                ["Hello world. This is", "is is a test."],
            ),
        ],
    )
    def test_chunking_logic_manual(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
        expected_chunks_manual_check: List[str],
    ):
        assert chunk_text_simple(text, chunk_size, overlap) == expected_chunks_manual_check

    @pytest.mark.parametrize(
        "chunk_size, overlap, error_message_match",
        [
            (0, 0, "chunk_size_chars must be a positive integer."),
            (-1, 0, "chunk_size_chars must be a positive integer."),
            (10, -1, "overlap_chars must be a non-negative integer."),
            (10, 10, "overlap_chars must be less than chunk_size_chars."),
            (10, 11, "overlap_chars must be less than chunk_size_chars."),
        ],
    )
    def test_invalid_parameters(self, chunk_size: int, overlap: int, error_message_match: str):
        with pytest.raises(ValueError, match=error_message_match):
            chunk_text_simple("some text", chunk_size, overlap)

    def test_chunk_size_one_no_overlap(self):
        text = "abc"
        chunk_size = 1
        overlap = 0
        expected = ["a", "b", "c"]
        assert chunk_text_simple(text, chunk_size, overlap) == expected

    def test_chunk_size_one_with_overlap_error(self):
        text = "abc"
        chunk_size = 1
        overlap = 1  # overlap must be less than chunk_size
        with pytest.raises(ValueError, match="overlap_chars must be less than chunk_size_chars."):
            chunk_text_simple(text, chunk_size, overlap)

    def test_long_text_consistency(self):
        text = "a" * 1000
        chunk_size = 100
        overlap = 10
        chunks = chunk_text_simple(text, chunk_size, overlap)

        # Expected number of chunks:
        # First chunk covers 100. Remaining text 900.
        # Each step is chunk_size - overlap = 90.
        # Number of additional steps = ceil(900 / 90) = 10.
        # Total chunks = 1 (initial) + 10 = 11.
        # Or, using a formula: 1 + ceil((L-C)/(C-O)) if L > C else 1 (if L>0) else 0
        # L=1000, C=100, O=10. C-O = 90.
        # 1 + ceil((1000-100)/90) = 1 + ceil(900/90) = 1 + 10 = 11.
        assert len(chunks) == 11

        # Check content
        assert chunks[0] == "a" * 100
        assert chunks[1] == "a" * 100  # starts at index 90
        assert chunks[1][0:10] == "a" * 10  # Overlap part
        assert chunks[1][10:] == "a" * 90  # New part

        # Last chunk
        # Last chunk starts at index: (11-1) * 90 = 10 * 90 = 900
        # It should be text[900 : 900+100] = text[900:1000]
        assert chunks[-1] == "a" * 100
        assert len(chunks[-1]) == 100

    def test_text_length_equals_chunk_size(self):
        text = "1234567890"
        chunk_size = 10
        overlap = 2
        expected = ["1234567890"]
        assert chunk_text_simple(text, chunk_size, overlap) == expected
