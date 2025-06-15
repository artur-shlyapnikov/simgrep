from pathlib import Path
import pytest
from simgrep.core.abstractions import TextExtractor


@pytest.mark.contract
class TextExtractorContract:
    def test_extract_returns_string(self, text_extractor: TextExtractor, text_file: Path):
        content = text_extractor.extract(text_file)
        assert isinstance(content, str)

    def test_extract_from_non_existent_file_raises(self, text_extractor: TextExtractor, tmp_path: Path):
        non_existent_file = tmp_path / "non_existent.txt"
        with pytest.raises(FileNotFoundError):
            text_extractor.extract(non_existent_file)
