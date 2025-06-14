import codecs
import hashlib
import zipfile
from io import BytesIO
from pathlib import Path
from typing import List

import pytest
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizerBase

from simgrep.adapters.hf_chunker import HFChunker, load_tokenizer
from simgrep.adapters.sentence_embedder import SentenceEmbedder
from simgrep.adapters.unstructured_extractor import UnstructuredExtractor
from simgrep.utils import calculate_file_hash


pytest.importorskip("unstructured")


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
    @pytest.fixture
    def extractor(self) -> UnstructuredExtractor:
        return UnstructuredExtractor()

    def test_extract_from_existing_file(self, extractor: UnstructuredExtractor, temp_text_file: Path) -> None:
        expected_content = "Hello World.\nThis is a test file.\nSimgrep is cool."
        assert extractor.extract(temp_text_file) == expected_content

    def test_extract_from_empty_file(self, extractor: UnstructuredExtractor, temp_empty_file: Path) -> None:
        assert extractor.extract(temp_empty_file) == ""

    def test_extract_from_non_existent_file(self, extractor: UnstructuredExtractor, tmp_path: Path) -> None:
        non_existent_file = tmp_path / "non_existent.txt"
        with pytest.raises(
            FileNotFoundError,
            match=f"File not found or is not a file: {non_existent_file}",
        ):
            extractor.extract(non_existent_file)

    def test_extract_from_directory(self, extractor: UnstructuredExtractor, temp_dir: Path) -> None:
        with pytest.raises(FileNotFoundError, match=f"File not found or is not a file: {temp_dir}"):
            extractor.extract(temp_dir)

    def test_non_utf8_encoded_file(self, extractor: UnstructuredExtractor, non_utf8_file: Path) -> None:
        # Unstructured handles various encodings automatically
        content = extractor.extract(non_utf8_file)
        assert "Sección de prueba" in content
        assert "2024" in content

    @pytest.mark.timeout(10)
    def test_pathological_binary_file_zip(self, extractor: UnstructuredExtractor, binary_zip_file: Path) -> None:
        # Unstructured might extract metadata or nothing from a zip. We expect an empty string.
        content = extractor.extract(binary_zip_file)
        assert content == ""
        assert isinstance(content, str)

    @pytest.mark.timeout(20)
    def test_very_large_repetitive_file(self, extractor: UnstructuredExtractor, large_repetitive_file: Path) -> None:
        content = extractor.extract(large_repetitive_file)
        assert isinstance(content, str)
        expected_pattern_start = "This is a highly repetitive short sentence"
        assert content.startswith(expected_pattern_start)
        assert len(content) > 900000

    def test_file_with_unicode_name_handling(self, extractor: UnstructuredExtractor, file_with_unicode_name: Path) -> None:
        content = extractor.extract(file_with_unicode_name)
        assert content == "Content of file with unicode name."

    def test_extract_from_utf8_with_bom_file(self, extractor: UnstructuredExtractor, utf8_with_bom_file: Path) -> None:
        expected_content = "File with BOM."
        content = extractor.extract(utf8_with_bom_file)
        assert content == expected_content


class TestLoadTokenizer:
    VALID_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    INVALID_MODEL_NAME = "this-model-does-not-exist-ever-12345"

    def test_load_valid_tokenizer(self) -> None:
        try:
            tokenizer = load_tokenizer(self.VALID_MODEL_NAME)
            assert isinstance(tokenizer, PreTrainedTokenizerBase)
            assert tokenizer.encode("hello world") is not None
        except Exception as e:
            pytest.fail(f"Failed to load a valid tokenizer: {e}")

    def test_load_invalid_tokenizer(self) -> None:
        with pytest.raises(
            Exception,  # Can be OSError or custom SimgrepError
        ):
            load_tokenizer(self.INVALID_MODEL_NAME)


class TestHFTokenChunker:
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    @pytest.fixture
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return load_tokenizer(self.MODEL_NAME)

    @pytest.fixture
    def chunker(self) -> HFChunker:
        return HFChunker(
            model_name=self.MODEL_NAME,
            chunk_size=10,
            overlap=2,
        )

    def test_empty_text(self, chunker: HFChunker) -> None:
        assert chunker.chunk("") == []

    def test_text_shorter_than_chunk_size(self, tokenizer: PreTrainedTokenizerBase) -> None:
        chunker = HFChunker(
            model_name=self.MODEL_NAME,
            chunk_size=128,
            overlap=20,
        )
        text = "This is a short text."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0].text.lower() == text.lower()


class TestSentenceEmbedder:
    VALID_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    INVALID_MODEL_NAME = "this-model-does-not-exist-ever-12345"

    @pytest.fixture(scope="class")
    def embedder(self) -> SentenceEmbedder:
        return SentenceEmbedder(self.VALID_MODEL_NAME)

    def test_generate_valid_embeddings(self, embedder: SentenceEmbedder) -> None:
        import numpy as np

        texts = ["Hello world", "Simgrep is amazing"]
        embeddings = embedder.encode(texts)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == embedder.ndim

    def test_generate_embeddings_empty_list(self, embedder: SentenceEmbedder) -> None:
        import numpy as np

        texts: List[str] = []
        embeddings = embedder.encode(texts)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 0

    def test_generate_embeddings_invalid_model(self) -> None:
        with pytest.raises(Exception):
            SentenceEmbedder(self.INVALID_MODEL_NAME)


class TestCalculateFileHash:
    def test_calculate_file_hash_valid_file(self, temp_text_file: Path) -> None:
        expected = hashlib.sha256(temp_text_file.read_bytes()).hexdigest()
        assert calculate_file_hash(temp_text_file) == expected

    def test_calculate_file_hash_missing_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.txt"
        with pytest.raises(FileNotFoundError):
            calculate_file_hash(missing)