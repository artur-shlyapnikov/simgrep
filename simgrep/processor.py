from pathlib import Path
from typing import List

import numpy as np
import unstructured.partition.auto as auto_partition
from sentence_transformers import SentenceTransformer
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
        extracted_texts = [el.text for el in elements if hasattr(el, "text")]
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

        # If the chunk is empty, it means current_idx was already at or past len(text).
        # This check is mostly a safeguard, as `while current_idx < len(text)` should prevent it.
        if not chunk:
            break

        # Add chunk if its length is >= overlap_chars,
        # OR if overlap_chars is 0 (any non-empty chunk is fine),
        # OR if it's the very first chunk (even if short).
        if len(chunk) >= overlap_chars or overlap_chars == 0 or not chunks:
            chunks.append(chunk)
        else:
            # This chunk is shorter than overlap_chars, overlap_chars > 0,
            # and it's not the first chunk.
            # This implies it's a small trailing piece. Stop processing.
            break

        current_idx += effective_step

    return chunks


def generate_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Generates vector embeddings for a list of input texts using a specified
    sentence-transformer model.

    The underlying SentenceTransformer library handles caching of models,
    typically in `~/.cache/torch/sentence_transformers/` or a similar
    directory, so subsequent calls with the same model_name are faster
    after the initial download.

    Args:
        texts: A list of strings for which to generate embeddings.
               If the list is empty, an empty numpy array is returned.
        model_name: The identifier of the sentence-transformer model to use
                    (e.g., 'all-MiniLM-L6-v2').

    Returns:
        A 2D numpy.ndarray where each row corresponds to the embedding
        of the text at the same index in the input list. Returns an
        empty np.ndarray with shape (0, embedding_dim) if texts is empty.
        The embedding dimension is model-specific.

    Raises:
        RuntimeError: If there's an issue with model loading or the
                      embedding process itself (e.g., network error during
                      download, incompatible model format).
    """
    # SentenceTransformer().encode([]) returns an empty float32 numpy array of shape (0, hidden_dimension)
    # So, we can rely on that behavior for empty lists.
    # No special handling needed for `if not texts:` before model loading if we let `model.encode` handle it.

    try:
        model = SentenceTransformer(model_name)
        # show_progress_bar=False is suitable for now as lists are expected to be small.
        embeddings = model.encode(texts, show_progress_bar=False)
        return embeddings
    except Exception as e:
        # Catching a broad Exception as SentenceTransformer can raise various errors.
        error_message = (
            f"Failed to generate embeddings using model '{model_name}'. "
            f"Ensure the model name is correct and an internet connection "
            f"is available for the first download. Original error: {e}"
        )
        raise RuntimeError(error_message) from e
