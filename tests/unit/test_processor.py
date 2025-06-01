import codecs
import zipfile
from io import BytesIO
from pathlib import Path
from typing import List

import pytest
from transformers import PreTrainedTokenizerBase

from simgrep.processor import extract_text_from_file


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


# tests for extract_text_from_file
class TestExtractTextFromFile:
    def test_extract_from_existing_file(self, temp_text_file: Path) -> None:
        expected_content = "Hello World.\nThis is a test file.\nSimgrep is cool."
        assert extract_text_from_file(temp_text_file) == expected_content

    def test_extract_from_empty_file(self, temp_empty_file: Path) -> None:
        assert extract_text_from_file(temp_empty_file) == ""

    def test_extract_from_non_existent_file(self, tmp_path: Path) -> None:
        non_existent_file = tmp_path / "non_existent.txt"
        with pytest.raises(
            FileNotFoundError,
            match=f"File not found or is not a file: {non_existent_file}",
        ):
            extract_text_from_file(non_existent_file)

    def test_extract_from_directory(self, temp_dir: Path) -> None:
        with pytest.raises(
            FileNotFoundError, match=f"File not found or is not a file: {temp_dir}"
        ):
            extract_text_from_file(temp_dir)

    def test_non_utf8_encoded_file(self, non_utf8_file: Path) -> None:
        try:
            content = extract_text_from_file(non_utf8_file)
            assert "Secci" in content
            assert "prueba" in content
            assert "2024" in content
        except RuntimeError as e:
            pytest.fail(f"extract_text_from_file crashed on non-UTF-8 file: {e}")
        except UnicodeDecodeError as e:
            pytest.fail(f"Unhandled UnicodeDecodeError for non-UTF-8 file: {e}")

    @pytest.mark.timeout(10)
    def test_pathological_binary_file_zip(self, binary_zip_file: Path) -> None:
        try:
            content = extract_text_from_file(binary_zip_file)
            assert isinstance(content, str)
        except RuntimeError:
            pass
        except Exception as e:
            pytest.fail(
                f"extract_text_from_file failed unexpectedly on binary (zip) file: {e}"
            )

    @pytest.mark.timeout(20)  # allow more time for larger file
    def test_very_large_repetitive_file(self, large_repetitive_file: Path) -> None:
        try:
            content = extract_text_from_file(large_repetitive_file)
            assert isinstance(content, str)
            expected_pattern_start = "This is a highly repetitive short sentence"
            assert content.startswith(expected_pattern_start)
            assert len(content) > 900000
        except RuntimeError as e:
            pytest.fail(f"extract_text_from_file crashed on large repetitive file: {e}")
        except Exception as e:
            pytest.fail(
                f"extract_text_from_file failed unexpectedly on large repetitive file: {e}"
            )

    def test_file_with_unicode_name_handling(
        self, file_with_unicode_name: Path
    ) -> None:
        try:
            content = extract_text_from_file(file_with_unicode_name)
            assert content == "Content of file with unicode name."
        except Exception as e:
            pytest.fail(f"extract_text_from_file failed on file with unicode name: {e}")

    def test_extract_from_utf8_with_bom_file(self, utf8_with_bom_file: Path) -> None:
        expected_content = "File with BOM."
        try:
            content = extract_text_from_file(utf8_with_bom_file)
            assert content == expected_content
        except Exception as e:
            pytest.fail(f"extract_text_from_file failed on UTF-8 file with BOM: {e}")


# tests for load_tokenizer
class TestLoadTokenizer:
    VALID_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # a common small model
    INVALID_MODEL_NAME = "this-model-does-not-exist-ever-12345"

    def test_load_valid_tokenizer(self) -> None:
        from transformers import PreTrainedTokenizerBase

        from simgrep.processor import load_tokenizer

        try:
            tokenizer = load_tokenizer(self.VALID_MODEL_NAME)
            assert isinstance(tokenizer, PreTrainedTokenizerBase)
            # simple check if tokenizer works
            assert tokenizer.encode("hello world") is not None
        except RuntimeError as e:
            pytest.fail(f"Failed to load a valid tokenizer: {e}")

    def test_load_invalid_tokenizer(self) -> None:
        from simgrep.processor import load_tokenizer

        with pytest.raises(
            RuntimeError,
            match=f"Failed to load tokenizer for model '{self.INVALID_MODEL_NAME}'",
        ):
            load_tokenizer(self.INVALID_MODEL_NAME)


# tests for chunk_text_by_tokens
class TestChunkTextByTokens:
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # for consistency

    @pytest.fixture(scope="class")
    def tokenizer(self) -> PreTrainedTokenizerBase:
        from simgrep.processor import load_tokenizer

        return load_tokenizer(self.MODEL_NAME)

    def test_empty_text(self, tokenizer: PreTrainedTokenizerBase) -> None:
        from simgrep.processor import chunk_text_by_tokens

        chunks = chunk_text_by_tokens("", tokenizer, 10, 2)
        assert chunks == []

    def test_text_shorter_than_chunk_size(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        from simgrep.processor import chunk_text_by_tokens

        text = "Short text."  # this text will likely be 3-4 tokens
        chunks = chunk_text_by_tokens(text, tokenizer, 10, 2)
        assert len(chunks) == 1
        # all-minilm-l6-v2 is uncased, so output text will be lowercased.
        assert chunks[0]["text"] == text.strip().lower()
        assert chunks[0]["start_char_offset"] == 0
        assert chunks[0]["end_char_offset"] == len(text)
        assert chunks[0]["token_count"] == len(
            tokenizer.encode(text, add_special_tokens=False)
        )

    def test_text_equals_chunk_size(self, tokenizer: PreTrainedTokenizerBase) -> None:
        from simgrep.processor import chunk_text_by_tokens

        # craft text that is exactly chunk_size_tokens tokens
        # this is model specific, so we'll approximate.
        # let's aim for 5 tokens with chunk_size 5.
        text_tokens = ["This", "is", "five", "tokens", "exactly"]
        text = tokenizer.convert_tokens_to_string(text_tokens)

        # verify token count with the tokenizer
        actual_token_ids = tokenizer.encode(text, add_special_tokens=False)
        # this assertion might be fragile if the model tokenizes differently than expected.
        # for this test, we assume it tokenizes as one token per word here.

        chunks = chunk_text_by_tokens(
            text, tokenizer, len(actual_token_ids), 0
        )  # no overlap
        assert len(chunks) == 1
        # all-minilm-l6-v2 is uncased, so output text will be lowercased.
        assert chunks[0]["text"].strip() == text.strip().lower()
        assert chunks[0]["token_count"] == len(actual_token_ids)

    def test_multiple_chunks_no_overlap(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        from simgrep.processor import chunk_text_by_tokens

        # approx 10 tokens. "this is a slightly longer sentence for testing purposes."
        text = "This is a test sentence. Here is another one for good measure."
        # tokenize to get actual token count
        token_ids = tokenizer.encode(text, add_special_tokens=False)

        chunk_size = 5
        overlap = 0
        chunks = chunk_text_by_tokens(text, tokenizer, chunk_size, overlap)

        expected_num_chunks = (
            len(token_ids) + chunk_size - 1
        ) // chunk_size  # ceiling division
        assert len(chunks) == expected_num_chunks

        reconstructed_token_ids = []
        for chunk in chunks:
            chunk_token_ids = tokenizer.encode(chunk["text"], add_special_tokens=False)
            reconstructed_token_ids.extend(chunk_token_ids)

        # we can't directly compare token_ids list due to how chunking might split words/subwords
        # but we can check character offsets
        assert chunks[0]["start_char_offset"] == 0
        if len(chunks) > 1:
            # the start of the second chunk should be the end of the first token sequence of the first chunk
            # this is also tricky. let's check total length.
            assert chunks[-1]["end_char_offset"] == len(text)

    def test_multiple_chunks_with_overlap(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        from simgrep.processor import chunk_text_by_tokens

        text = "This is a test sentence with quite a few words to ensure multiple overlapping chunks are created."
        chunk_size = 10
        overlap = 3
        chunks = chunk_text_by_tokens(text, tokenizer, chunk_size, overlap)

        assert len(chunks) > 1
        # check overlap: end of first chunk's tokens should overlap with start of second chunk's tokens
        if len(chunks) > 1:
            # the last `overlap` tokens of the first chunk's *token source* should match
            # the first `overlap` tokens of the second chunk's *token source*.
            # this is hard to verify perfectly without knowing the exact token boundaries from original.
            # a simpler check: the start_char_offset of chunk2 should be less than end_char_offset of chunk1
            assert chunks[1]["start_char_offset"] < chunks[0]["end_char_offset"]

    def test_invalid_chunk_size(self, tokenizer: PreTrainedTokenizerBase) -> None:
        from simgrep.processor import chunk_text_by_tokens

        with pytest.raises(
            ValueError, match="chunk_size_tokens must be a positive integer"
        ):
            chunk_text_by_tokens("text", tokenizer, 0, 0)
        with pytest.raises(
            ValueError, match="chunk_size_tokens must be a positive integer"
        ):
            chunk_text_by_tokens("text", tokenizer, -1, 0)

    def test_invalid_overlap_size(self, tokenizer: PreTrainedTokenizerBase) -> None:
        from simgrep.processor import chunk_text_by_tokens

        with pytest.raises(
            ValueError, match="overlap_tokens must be a non-negative integer"
        ):
            chunk_text_by_tokens("text", tokenizer, 10, -1)
        with pytest.raises(
            ValueError, match="overlap_tokens must be less than chunk_size_tokens"
        ):
            chunk_text_by_tokens("text", tokenizer, 10, 10)
        with pytest.raises(
            ValueError, match="overlap_tokens must be less than chunk_size_tokens"
        ):
            chunk_text_by_tokens("text", tokenizer, 10, 11)

    def test_text_results_in_no_tokens(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        from simgrep.processor import chunk_text_by_tokens

        # some tokenizers might return empty for only special characters or whitespace.
        # however, hf tokenizers usually handle this. let's use empty string after stripping.
        text = "      "  # only whitespace
        chunks = chunk_text_by_tokens(text, tokenizer, 10, 2)
        assert chunks == []  # because full_text.strip() will be empty


# tests for generate_embeddings
class TestGenerateEmbeddings:
    VALID_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    INVALID_MODEL_NAME = "this-model-does-not-exist-ever-12345"

    @pytest.fixture(scope="class")
    def sentence_transformer_model(self) -> "SentenceTransformer": # type: ignore # noqa: F821
        from sentence_transformers import SentenceTransformer
        # This relies on the model being pre-cached by CI/Makefile scripts
        # or downloaded during the first run of this fixture.
        return SentenceTransformer(self.VALID_MODEL_NAME)

    def test_generate_valid_embeddings(self, sentence_transformer_model: "SentenceTransformer") -> None: # type: ignore # noqa: F821
        import numpy as np
        from simgrep.processor import generate_embeddings

        texts = ["Hello world", "Simgrep is amazing"]
        try:
            # Use the pre-loaded model
            embeddings = generate_embeddings(texts, model=sentence_transformer_model)
            assert isinstance(embeddings, np.ndarray)
            assert embeddings.shape[0] == len(texts)
            assert embeddings.shape[1] > 0  # embedding dimension
        except RuntimeError as e:
            pytest.fail(f"Failed to generate embeddings with a valid model: {e}")

    def test_generate_embeddings_empty_list(self, sentence_transformer_model: "SentenceTransformer") -> None: # type: ignore # noqa: F821
        import numpy as np
        from simgrep.processor import generate_embeddings

        texts: List[str] = []
        try:
            # Use the pre-loaded model
            embeddings = generate_embeddings(texts, model=sentence_transformer_model)
            assert isinstance(embeddings, np.ndarray)
            assert embeddings.shape[0] == 0
            if embeddings.ndim == 1:
                assert embeddings.shape == (0,)
            elif embeddings.ndim == 2:
                assert embeddings.shape[1] > 0
            else:
                pytest.fail(
                    f"Unexpected ndim {embeddings.ndim} for empty list embedding, shape: {embeddings.shape}"
                )
        except RuntimeError as e:
            pytest.fail(f"Failed to generate embeddings for an empty list: {e}")

    def test_generate_embeddings_invalid_model(self) -> None:
        from simgrep.processor import generate_embeddings

        texts = ["This will fail"]
        # This test specifically checks the model loading by name path with an invalid name
        with pytest.raises(
            RuntimeError,
            match=f"Failed to generate embeddings using model '{self.INVALID_MODEL_NAME}'",
        ):
            generate_embeddings(texts, model_name=self.INVALID_MODEL_NAME)