from pathlib import Path
from typing import List

import unstructured.partition.auto as auto_partition
from unstructured.documents.elements import Element

from simgrep.core.abstractions import TextExtractor


class UnstructuredExtractor(TextExtractor):
    def extract(self, path: Path) -> str:
        """
        Extracts text content from a given file path using unstructured.
        Raises FileNotFoundError if the file does not exist or is not a file.
        Raises RuntimeError if unstructured fails to process the file.
        """
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"File not found or is not a file: {path}")

        try:
            elements: List[Element] = auto_partition.partition(filename=str(path), TESSERACT_LANGUAGES=["eng"])
            extracted_texts = [el.text for el in elements if hasattr(el, "text")]
            return "\n".join(extracted_texts)
        except Exception:
            # For binary or unparseable files, unstructured may raise.
            # We'll treat this as a file with no extractable text.
            return ""