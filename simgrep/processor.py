from pathlib import Path
from typing import List
import unstructured.partition.auto as auto_partition
from unstructured.documents.elements import Element

def extract_text_from_file(file_path: Path) -> str:
    """
    Extracts text content from a given file path using unstructured.
    Raises FileNotFoundError if the file does not exist or is not a file.
    Raises RuntimeError if unstructured fails to process the file.
    """
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"File not found or is not a file: {file_path}")

    try:
        # Use the general auto partition function.
        # This function can handle various file types, including text.
        elements: List[Element] = auto_partition.partition(filename=str(file_path))
        
        # Concatenate text from all elements
        extracted_texts = [el.text for el in elements if hasattr(el, 'text')]
        return "\n".join(extracted_texts)
    except Exception as e:
        # Catch potential errors from unstructured partitioning
        # and re-raise as a more specific error or handle as needed.
        print(f"Error processing file {file_path} with unstructured: {e}")
        # Depending on desired behavior, either re-raise or return empty string/raise custom error
        raise RuntimeError(f"Failed to extract text from {file_path}") from e
