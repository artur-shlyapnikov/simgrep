import codecs
import zipfile
from io import BytesIO
from pathlib import Path
from typing import List

import pytest

from simgrep.processor import chunk_text_simple, extract_text_from_file


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
    repetitions = 10000 # (len(pattern) is 100)
    with open(file_path, "w", encoding="utf-8") as f:
        for _ in range(repetitions):
            f.write(pattern)
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
        with pytest.raises(FileNotFoundError, match=f"File not found or is not a file: {temp_dir}"):
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

    @pytest.mark.timeout(10) # Protect against hangs
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
            pytest.fail(f"extract_text_from_file failed unexpectedly on binary (zip) file: {e}")

    @pytest.mark.timeout(20) # Allow more time for larger file
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
            assert len(content) > 900000 # Check it's mostly there
        except RuntimeError as e:
            pytest.fail(f"extract_text_from_file crashed on large repetitive file: {e}")
        except Exception as e:
            pytest.fail(f"extract_text_from_file failed unexpectedly on large repetitive file: {e}")

    def test_file_with_unicode_name_handling(self, file_with_unicode_name: Path):
        # This test primarily ensures that `str(file_path)` and its use in
        # `unstructured.partition.auto.partition(filename=str(file_path))`
        # works correctly with unicode filenames.
        try:
            content = extract_text_from_file(file_with_unicode_name)
            assert content == "Content of file with unicode name."
        except Exception as e:
            pytest.fail(f"extract_text_from_file failed on file with unicode name: {e}")


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
                    "abcdefghij", # 0-9
                    "hijklmnopq", # 7-16
                    "opqrstuvwx", # 14-23
                    "vwxyz",      # 21-25
                ],
            ),
            # Text shorter than chunk size
            ("abc", 10, 3, ["abc"]),
            # Exact multiple of (chunk_size - overlap) after first chunk
            (
                "abcdefghijklmno", # len 15
                10, # C
                5,  # O
                # Step = C-O = 5
                ["abcdefghij", # 0-9
                 "fghijklmno", # 5-14
                 "klmno"],      # 10-14
            ),
            # Last chunk smaller
            (
                "abcdefghijkl", # len 12
                10, # C
                5,  # O
                # Step = C-O = 5
                ["abcdefghij", # 0-9
                 "fghijkl"],   # 5-11
            ),
            # Zero overlap
            (
                "abcdefghijklm", # len 13
                5,  # C
                0,  # O
                # Step = C-O = 5
                ["abcde", # 0-4
                 "fghij", # 5-9
                 "klm"],  # 10-12
            ),
            # Empty text
            ("", 10, 3, []),
            # Overlap makes next chunk start beyond text
            ("abcdefghij", 10, 0, ["abcdefghij"]),
            # Overlap makes next chunk start exactly at end (no more chunks)
            (
                "abcdefghij", # len 10
                5, # C
                2, # O
                # Step = C-O = 3
                ["abcde", # 0-4
                 "defgh", # 3-7
                 "ghij"], # 6-9
            ),
            ("12345", 5, 0, ["12345"]),
            (
                "1234567890", # len 10
                5, # C
                1, # O
                # Step = C-O = 4
                ["12345", # 0-4
                 "56789", # 4-8
                 "90"],   # 8-9 (text[8:8+5] -> text[8:10])
            ),
            (
                "1234567890", # len 10
                5, # C
                4, # O
                # Step = C-O = 1
                ["12345", # 0-4
                 "23456", # 1-5
                 "34567", # 2-6
                 "45678", # 3-7
                 "56789", # 4-8
                 "67890", # 5-9
                 "7890"], # 6-9 (len 4, shorter, so last)
            ),
            # Test Case: Text Containing Only Whitespace or Newlines
            # text = '     ', C=3, O=1, S=2. L=5
            # idx=0, chunk='   ' (0-2). idx=2.
            # idx=2, chunk='   ' (2-4). idx=4.
            # idx=4, chunk=' '   (4-4). len=1 < C. break.
            # text = '     ', C=3, O=1, S=2. L=5
            # idx=0, chunk='   ' (0-2). idx=2.
            # idx=2, chunk='   ' (2-4). idx=4.
            # idx=4, chunk=' '   (4-4). len=1 < C. break.
            ("     ", 3, 1, ["   ", "   ", " "]),
            ("\n\n\n\n", 2, 0, ["\n\n", "\n\n"]), # 4 newlines, C=2, O=0, S=2
            # text = '  \t  ', C=5, O=2, S=3. L=5
            # idx=0, chunk='  \t  ' (0-4). idx=3.
            # idx=3, chunk='  '    (3-4). len=2 < C. break.
            ("  \t  ", 5, 2, ["  \t  ", "  "]),
            # text = '      ', C=2, O=1, S=1. L=6
            # idx=0, chunk='  '. idx=1
            # idx=1, chunk='  '. idx=2
            # idx=2, chunk='  '. idx=3
            # idx=3, chunk='  '. idx=4
            # idx=4, chunk='  '. idx=5
            # idx=5, chunk=' '. len=1 < C. break.
            ("      ", 2, 1, ["  ", "  ", "  ", "  ", "  ", " "]),
            # Test Case: Text Length Causes Last Chunk to be Exactly `overlap_chars` Long
            # C=10, O=7 (implies S=3). Last chunk should be of length 7.
            # text = "abcdefghijklm" (length 13)
            # Chunk 1: "abcdefghij" (text[0:10])
            # Next start: idx = 3.
            # Chunk 2: "defghijklm" (text[3:13]) -> this is the last chunk. Length 10.
            # The overlap from Chunk 1 with Chunk 2 is "defghij" (7 chars).
            # This example doesn't make the *last chunk itself* equal to overlap_chars.
            # Let's re-craft:
            # text = "abcdefghijxxx" (len 13), C=10, O=3 (S=7)
            # Chunk 1: "abcdefghij" (text[0:10])
            # Next start: idx = 7
            # Chunk 2: "hijxxx" (text[7:13]) -> length 6. Overlap was "hij" (3 chars).
            # This still doesn't fit "last chunk is exactly overlap_chars long".

            # Let text_len = C + S - O where S is step C-O
            # No, let text_len = C + (N-1)*S such that the last chunk is text[ (N-1)*S : (N-1)*S + C ]
            # and we want the part of this last chunk that is new (not overlapping with previous) to be small.
            # The question is: "Last chunk to be exactly overlap_chars long"
            # This means text[ (N-1)*S : L ] where L - (N-1)*S == O.
            # Example: C=5, O=3 (S=2).
            # T = "abcdefg" (len 7)
            # C1: "abcde" (0-4)
            # idx=2. C2: "cdefg" (2-6)
            # idx=4. C3: "efg" (4-6). Length 3. This is O.
            ("abcdefg", 5, 3, ["abcde", "cdefg", "efg"]),
            # Example: C=10, O=7 (S=3).
            # T = "abcdefghijklm" (len 13)
            # C1: "abcdefghij" (0-9)
            # idx=3. C2: "defghijklm" (3-12)
            # idx=6. C3: "ghijklm" (6-12). Length 7. This is O.
            ("abcdefghijklm", 10, 7, ["abcdefghij", "defghijklm", "ghijklm"]),
        ],
    )
    def test_chunking_logic(self, text: str, chunk_size: int, overlap: int, expected_chunks: List[str]):
        assert chunk_text_simple(text, chunk_size, overlap) == expected_chunks

    @pytest.mark.parametrize(
        "text, chunk_size, overlap, expected_chunks_manual_check",
        [
            (
                "Hello world. This is a test.", # len 29
                20, # C
                5,  # O
                # S = 15
                ["Hello world. This is", # 0-19
                 "is is a test."],      # 15-28 (text[15:15+20])
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
        # This covers: Test Case: chunk_size_chars is 1 and overlap_chars is 0
        text = "abc"
        chunk_size = 1
        overlap = 0
        expected = ["a", "b", "c"]
        assert chunk_text_simple(text, chunk_size, overlap) == expected

        # Performance aspect for very long string (conceptual, not a true perf test here)
        long_text = "a" * 200 # Small scale for unit test speed
        expected_long = ["a"] * 200
        assert chunk_text_simple(long_text, chunk_size, overlap) == expected_long


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

        # L=1000, C=100, O=10, S=90
        # Chunks start at 0, 90, 180, ..., 810 (10 chunks, text[810:910])
        # Next start 900. chunk text[900:1000] (11th chunk)
        # Next start 990. chunk text[990:1000] (12th chunk, len 10, shorter than C)
        assert len(chunks) == 12
        assert chunks[0] == "a" * 100
        assert chunks[1] == "a" * 100 # text[90:190]
        assert chunks[1][0:10] == "a" * 10 # Overlap part
        assert chunks[1][10:] == "a" * 90 # New part

        assert chunks[10] == "a" * 100 # This is the chunk text[900:1000]
        assert len(chunks[10]) == 100

        assert chunks[-1] == "a" * 10 # Last chunk text[990:1000]
        assert len(chunks[-1]) == 10

    def test_text_length_equals_chunk_size(self):
        text = "1234567890"
        chunk_size = 10
        overlap = 2
        # L=10, C=10, O=2, S=8
        # idx=0, chunk="1234567890". len=10. idx=8.
        # idx=8, chunk="90". len=2 < C. break.
        expected = ["1234567890", "90"]
        assert chunk_text_simple(text, chunk_size, overlap) == expected