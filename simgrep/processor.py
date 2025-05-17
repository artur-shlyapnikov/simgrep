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


def chunk_text_simple(text: str, chunk_size_chars: int, overlap_chars: int) -> List[str]:
    """
    Splits a given text into a list of overlapping character-based chunks.
    """
    if chunk_size_chars <= 0:
        raise ValueError("chunk_size_chars must be a positive integer.")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be a non-negative integer.")
    if overlap_chars >= chunk_size_chars:
        raise ValueError("overlap_chars must be less than chunk_size_chars.")

    if not text:
        return []

    chunks: List[str] = []
    current_idx = 0
    
    # Effective step must be positive due to validation overlap_chars < chunk_size_chars
    effective_step = chunk_size_chars - overlap_chars

    while current_idx < len(text):
        chunk = text[current_idx : current_idx + chunk_size_chars]
        chunks.append(chunk)
        
        # If the end of the current chunk is at or beyond the end of the text,
        # it's the last chunk.
        if current_idx + chunk_size_chars >= len(text):
            break
        
        current_idx += effective_step
        # If current_idx itself becomes >= len(text) after stepping,
        # the 'while' condition will handle termination before next iteration if step is large.
        # However, the current logic ensures at least one chunk is added if text is not empty.

    return chunks