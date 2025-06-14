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
            elements: List[Element] = auto_partition.partition(filename=str(path))
            extracted_texts = [el.text for el in elements if hasattr(el, "text")]
            return "\n".join(extracted_texts)
        except Exception as e:
            # Consider logging this instead of printing for production code
            print(f"Error processing file {path} with unstructured: {e}")
            raise RuntimeError(f"Failed to extract text from {path}") from e
