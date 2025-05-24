import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import usearch.index

try:
    from .exceptions import VectorStoreError
except ImportError:
    from simgrep.exceptions import VectorStoreError  # type: ignore

logger = logging.getLogger(__name__)


def create_inmemory_index(
    embeddings: np.ndarray,
    labels_for_usearch: np.ndarray,
    metric: str = "cos",
    dtype: str = "f32",
) -> usearch.index.Index:
    """
    Creates and populates an in-memory USearch index.

    Args:
        embeddings: A 2D NumPy array of shape (num_chunks, embedding_dimension)
                    containing the vector embeddings for text chunks.
        labels_for_usearch: A 1D NumPy array of dtype np.int64, containing the unique labels
                            (e.g., ChunkData.usearch_label) for each embedding.
                            Shape must match embeddings.shape[0].
        metric: The distance metric for USearch (e.g., "cos", "ip", "l2sq").
        dtype: The data type of vectors in the index (e.g., "f32", "f16").

    Returns:
        A populated usearch.index.Index object.

    Raises:
        ValueError: If inputs are invalid.
        VectorStoreError: If USearch index creation fails.
    """
    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D NumPy array.")
    if not isinstance(labels_for_usearch, np.ndarray) or labels_for_usearch.ndim != 1:
        raise ValueError("labels_for_usearch must be a 1D NumPy array.")
    if embeddings.shape[0] == 0:
        # It's valid to have an empty index if no embeddings are provided.
        # USearch index can be initialized without data.
        logger.info("Embeddings array is empty. Creating an empty USearch index.")
    elif embeddings.shape[0] != labels_for_usearch.shape[0]:
        raise ValueError(
            f"Number of embeddings ({embeddings.shape[0]}) must match number of labels ({labels_for_usearch.shape[0]})."
        )

    # Ensure labels are np.int64 as USearch expects this for keys
    processed_labels = labels_for_usearch
    if labels_for_usearch.size > 0 and labels_for_usearch.dtype != np.int64:
        processed_labels = labels_for_usearch.astype(np.int64)

    num_dimensions: int
    if embeddings.shape[0] > 0:
        num_dimensions = embeddings.shape[1]
    else:
        # Cannot infer dimensions from empty embeddings.
        # This case should be handled by the caller providing ndim if creating an empty index
        # or by USearch having a default. For now, let's raise if empty and no explicit ndim.
        # However, the function signature doesn't take ndim.
        # Let's assume if embeddings are empty, we create an index that can be added to later.
        # USearch requires ndim at initialization.
        # This function is for *populating* an index. If embeddings are empty,
        # the caller should perhaps create an empty index differently.
        # For now, let's allow creation but log a warning if it's empty.
        # If we try to create an index with ndim=0 (from embeddings.shape[1]), USearch will fail.
        # A common pattern is to create an empty index with known ndim, then add.
        # This function combines creation and adding.
        # If embeddings are empty, we can't get num_dimensions.
        # Let's assume the caller provides non-empty embeddings or handles empty index creation.
        # For this function, if embeddings are empty, we will create an index that is also empty.
        # The `ndim` parameter for `usearch.index.Index` is mandatory.
        # A better approach for "empty" is that the caller must provide `ndim`.
        # For now, if `embeddings` is empty, we'll assume the caller knows `ndim` isn't derived.
        # This function is primarily for when embeddings *exist*.
        # Let's stick to the original logic: if embeddings are empty, raise ValueError.
        if embeddings.shape[0] == 0:
            raise ValueError(
                "Embeddings array cannot be empty when creating an index that infers ndim."
            )
        num_dimensions = embeddings.shape[1]

    logger.info(
        f"Creating USearch index with ndim={num_dimensions}, metric='{metric}', dtype='{dtype}'."
    )
    try:
        index = usearch.index.Index(
            ndim=num_dimensions,
            metric=metric,
            dtype=dtype,
        )
        if embeddings.shape[0] > 0:  # Only add if there are embeddings
            index.add(keys=processed_labels, vectors=embeddings)
            logger.info(f"Added {embeddings.shape[0]} embeddings to the USearch index.")
        else:
            logger.info(
                "USearch index created empty as no embeddings were provided to add."
            )

    except Exception as e:  # Catch broad USearch errors
        logger.error(f"Failed to create or populate USearch index: {e}")
        raise VectorStoreError("Failed to create or populate USearch index") from e

    return index


def search_inmemory_index(
    index: usearch.index.Index, query_embedding: np.ndarray, k: int = 5
) -> List[Tuple[int, float]]:
    """
    Searches an in-memory USearch index for the top-k most similar vectors.

    Args:
        index: The populated usearch.index.Index object to search.
        query_embedding: A 1D or 2D NumPy array for the query embedding.
                         If 1D (single query), shape: (embedding_dimension,).
                         If 2D (batch of 1 query), shape: (1, embedding_dimension).
        k: The number of top similar items to retrieve.

    Returns:
        A list of tuples, where each tuple is (chunk_original_index, similarity_score).
        Returns an empty list if no matches are found or if the index is empty.
        Similarity score is normalized (e.g., 0.0 to 1.0 for cosine similarity).

    Raises:
        ValueError: If inputs are invalid (e.g., k <= 0, dimension mismatch).
        VectorStoreError: If USearch search operation fails.
    """
    if not isinstance(index, usearch.index.Index):
        raise ValueError("Provided index is not a valid usearch.index.Index object.")
    if not isinstance(query_embedding, np.ndarray):
        raise ValueError("Query embedding must be a NumPy array.")
    if k <= 0:
        raise ValueError("k (number of results) must be a positive integer.")

    if len(index) == 0:
        logger.info("Search attempted on an empty USearch index. Returning no results.")
        return []

    processed_query_embedding = query_embedding
    if processed_query_embedding.ndim == 1:
        processed_query_embedding = np.expand_dims(processed_query_embedding, axis=0)

    if processed_query_embedding.shape[0] != 1:
        raise ValueError(
            f"Expected a single query embedding, but got batch of {processed_query_embedding.shape[0]}."
        )
    if processed_query_embedding.shape[1] != index.ndim:
        raise ValueError(
            f"Query embedding dimension ({processed_query_embedding.shape[1]}) "
            f"does not match index dimension ({index.ndim})."
        )

    logger.info(f"Searching USearch index for top {k} results.")
    try:
        search_result: Union[usearch.index.Matches, usearch.index.BatchMatches] = (
            index.search(vectors=processed_query_embedding, count=k)
        )
    except Exception as e:
        logger.error(f"USearch search operation failed: {e}")
        raise VectorStoreError("USearch search operation failed") from e

    results: List[Tuple[int, float]] = []
    actual_keys: Optional[np.ndarray] = None
    actual_distances: Optional[np.ndarray] = None
    num_found_for_query: int = 0

    if isinstance(search_result, usearch.index.BatchMatches):
        if search_result.counts is not None and len(search_result.counts) > 0:
            num_found_for_query = search_result.counts[0]
            if num_found_for_query > 0:
                actual_keys = search_result.keys[0]
                actual_distances = search_result.distances[0]
    elif isinstance(search_result, usearch.index.Matches):  # Single query result
        num_found_for_query = search_result.count  # type: ignore[attr-defined]
        if num_found_for_query > 0:
            actual_keys = search_result.keys
            actual_distances = search_result.distances

    if (
        num_found_for_query > 0
        and actual_keys is not None
        and actual_distances is not None
    ):
        logger.debug(f"Found {num_found_for_query} matches in USearch.")
        for i in range(num_found_for_query):
            key: int = int(actual_keys[i])
            distance: float = float(actual_distances[i])
            similarity: float
            metric_str = str(index.metric).lower()

            if "cos" in metric_str:
                similarity = 1.0 - distance
            elif (
                "ip" in metric_str
            ):  # Inner product; higher is better. USearch returns negative IP for similarity.
                similarity = -distance  # So, negate to get positive similarity.
            elif "l2" in metric_str:  # L2 squared distance; lower is better.
                similarity = 1.0 / (1.0 + distance)  # Simple inverse, can be refined.
            else:
                logger.warning(
                    f"Unknown metric '{index.metric}' for similarity conversion. Returning raw negative distance."
                )
                similarity = -distance  # Default to negative distance if metric unknown
            results.append((key, similarity))
    else:
        logger.info("No matches found by USearch for the query.")

    return results


def load_persistent_index(index_path: Path) -> Optional[usearch.index.Index]:
    """
    Loads a USearch index from a file.

    Args:
        index_path: Path to the USearch index file.

    Returns:
        A loaded usearch.index.Index object, or None if the file does not exist.

    Raises:
        VectorStoreError: If loading fails for an existing file.
    """
    if index_path.exists() and index_path.is_file():
        logger.info(f"Attempting to load USearch index from {index_path}")
        try:
            # For usearch >= 2.9.0, load is an instance method.
            # Create a temporary instance; load() will reconfigure it.
            # The Index constructor defaults to ndim=0, which is fine for this.
            index = usearch.index.Index()
            index.load(str(index_path))
            logger.info(
                f"Successfully loaded USearch index from {index_path} with {len(index)} items."
            )
            return index
        except Exception as e:
            logger.error(f"Failed to load USearch index from {index_path}: {e}")
            raise VectorStoreError(f"Failed to load index from {index_path}") from e
    else:
        logger.info(f"USearch index file not found at {index_path}. No index loaded.")
        return None


def save_persistent_index(index: usearch.index.Index, index_path: Path) -> None:
    """
    Saves a USearch index to a file atomically.

    Args:
        index: The usearch.index.Index object to save.
        index_path: The target path to save the index file.

    Raises:
        VectorStoreError: If saving fails.
    """
    logger.info(
        f"Attempting to save USearch index with {len(index)} items to {index_path}"
    )
    try:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists for USearch index: {index_path.parent}")
    except OSError as e:
        logger.error(
            f"Failed to create directory for USearch index at {index_path.parent}: {e}"
        )
        raise VectorStoreError(
            f"Could not create directory for USearch index at {index_path.parent}"
        ) from e

    temp_index_path = index_path.with_suffix(index_path.suffix + ".tmp")

    try:
        logger.info(f"Saving USearch index to temporary file {temp_index_path}")
        index.save(str(temp_index_path))
        logger.info(
            f"Successfully saved USearch index to temporary file {temp_index_path}"
        )

        try:
            os.replace(temp_index_path, index_path)
            logger.info(
                f"Atomically moved temporary index from {temp_index_path} to {index_path}"
            )
        except OSError as e_move:
            logger.error(
                f"Failed to move temporary index {temp_index_path} to {index_path}: {e_move}"
            )
            if temp_index_path.exists():
                try:
                    temp_index_path.unlink()
                    logger.debug(
                        f"Cleaned up temporary index file {temp_index_path} after move failure."
                    )
                except OSError as e_unlink:
                    logger.error(
                        f"Failed to clean up temporary index file {temp_index_path} after move failure: {e_unlink}"
                    )
            raise VectorStoreError(
                f"Failed to finalize saving index to {index_path}"
            ) from e_move

    except Exception as e:  # Covers index.save() errors
        logger.error(
            f"Failed to save USearch index to temporary file {temp_index_path}: {e}"
        )
        # Ensure temp file is cleaned up if save failed partway and left a file
        if temp_index_path.exists():
            try:
                temp_index_path.unlink()
                logger.debug(
                    f"Cleaned up temporary index file {temp_index_path} after save failure."
                )
            except OSError as e_unlink:
                logger.error(
                    f"Failed to clean up temporary index file {temp_index_path} after save failure: {e_unlink}"
                )
        raise VectorStoreError(f"Failed to save index to {temp_index_path}") from e
    finally:
        # Final check for temp file, in case an error occurred before os.replace but after save
        if temp_index_path.exists():
            try:
                temp_index_path.unlink()
                logger.warning(
                    f"Lingering temporary index file {temp_index_path} found and removed in finally block."
                )
            except OSError as e_unlink:
                logger.error(
                    f"Failed to clean up lingering temporary index file {temp_index_path} in finally block: {e_unlink}"
                )
