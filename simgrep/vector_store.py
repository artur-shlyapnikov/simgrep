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


class VectorStore:
    """Simple wrapper around ``usearch.index.Index`` supporting persistence."""

    def __init__(
        self,
        index_path: Optional[Path] | None = None,
        *,
        ndim: Optional[int] = None,
        metric: str = "cos",
        dtype: str = "f32",
    ) -> None:
        """Load an existing index or create a new one."""

        self.path: Optional[Path] = index_path
        self.metric = metric
        self.dtype = dtype

        loaded: Optional[usearch.index.Index] = None
        if index_path is not None and index_path.exists():
            loaded = load_persistent_index(index_path)

        if loaded is not None:
            self.index = loaded
        else:
            if ndim is None:
                raise ValueError("ndim must be provided when creating a new index")
            self.index = usearch.index.Index(ndim=ndim, metric=metric, dtype=dtype)

    def add_vectors(self, vectors: np.ndarray, labels: np.ndarray) -> None:
        """Add vectors and their integer labels to the index."""
        if not isinstance(vectors, np.ndarray) or vectors.ndim != 2:
            raise ValueError("Embeddings must be a 2D NumPy array.")
        if not isinstance(labels, np.ndarray) or labels.ndim != 1:
            raise ValueError("labels must be a 1D NumPy array.")
        if vectors.shape[0] == 0:
            raise ValueError("Embeddings array cannot be empty")
        if vectors.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Number of embeddings ({vectors.shape[0]}) must match number of labels ({labels.shape[0]})"
            )
        if vectors.shape[1] != self.index.ndim:
            raise ValueError(
                f"Embedding dimension ({vectors.shape[1]}) does not match index dimension ({self.index.ndim})"
            )

        self.index.add(keys=labels.astype(np.int64), vectors=vectors.astype(np.float32))

    def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        """Search for the ``k`` most similar vectors."""

        return search_inmemory_index(self.index, query, k)

    def save(self, path: Optional[Path] | None = None) -> None:
        """Persist the index to ``path`` or ``self.path``."""

        target = path or self.path
        if target is None:
            raise ValueError("No path specified to save index")
        save_persistent_index(self.index, target)
        self.path = target

    def load(self, path: Path) -> None:
        """Load a persisted index into this instance."""

        idx = load_persistent_index(path)
        if idx is None:
            raise VectorStoreError(f"Index file not found at {path}")
        self.index = idx
        self.path = path


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
        # it's valid to have an empty index if no embeddings are provided.
        # usearch index can be initialized without data.
        logger.info("Embeddings array is empty. Creating an empty USearch index.")
    elif embeddings.shape[0] != labels_for_usearch.shape[0]:
        raise ValueError(
            f"Number of embeddings ({embeddings.shape[0]}) must match number of labels ({labels_for_usearch.shape[0]})."
        )

    # ensure labels are np.int64 as usearch expects this for keys
    processed_labels = labels_for_usearch
    if labels_for_usearch.size > 0 and labels_for_usearch.dtype != np.int64:
        processed_labels = labels_for_usearch.astype(np.int64)

    num_dimensions: int
    if embeddings.shape[0] > 0:
        num_dimensions = embeddings.shape[1]
    else:
        # cannot infer dimensions from empty embeddings.
        # this case should be handled by the caller providing ndim if creating an empty index
        # or by usearch having a default. for now, let's raise if empty and no explicit ndim.
        # however, the function signature doesn't take ndim.
        # let's assume if embeddings are empty, we create an index that can be added to later.
        # usearch requires ndim at initialization.
        # this function is for *populating* an index. if embeddings are empty,
        # the caller should perhaps create an empty index differently.
        # for now, if `embeddings` is empty, we'll assume the caller knows `ndim` isn't derived.
        # this function is primarily for when embeddings *exist*.
        # let's stick to the original logic: if embeddings are empty, raise valueerror.
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
        if embeddings.shape[0] > 0:  # only add if there are embeddings
            index.add(keys=processed_labels, vectors=embeddings)
            logger.info(f"Added {embeddings.shape[0]} embeddings to the USearch index.")
        else:
            logger.info(
                "USearch index created empty as no embeddings were provided to add."
            )

    except Exception as e:  # catch broad usearch errors
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
                         if 1d (single query), shape: (embedding_dimension,).
                         if 2d (batch of 1 query), shape: (1, embedding_dimension).
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
    elif isinstance(search_result, usearch.index.Matches):  # single query result
        # .count attribute is not available on usearch.index.matches in recent versions.
        # use len(search_result.keys) instead.
        if hasattr(search_result, 'keys') and search_result.keys is not None:
            num_found_for_query = len(search_result.keys)
            if num_found_for_query > 0:
                actual_keys = search_result.keys
                actual_distances = search_result.distances
        else: # should not happen if keys is a mandatory attribute of matches
            num_found_for_query = 0
            actual_keys = None
            actual_distances = None


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
            ):  # inner product; higher is better. usearch returns negative ip for similarity.
                similarity = -distance  # so, negate to get positive similarity.
            elif "l2" in metric_str:  # l2 squared distance; lower is better.
                similarity = 1.0 / (1.0 + distance)  # simple inverse, can be refined.
            else:
                logger.warning(
                    f"Unknown metric '{index.metric}' for similarity conversion. Returning raw negative distance."
                )
                similarity = -distance  # default to negative distance if metric unknown
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
            # for usearch >= 2.9.0, load is an instance method.
            # create a temporary instance; load() will reconfigure it.
            # the index constructor defaults to ndim=0, which is fine for this.
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

    # step 1: save to temporary file
    try:
        logger.info(f"Saving USearch index to temporary file {temp_index_path}")
        index.save(str(temp_index_path))
        logger.info(
            f"Successfully saved USearch index to temporary file {temp_index_path}"
        )
    except Exception as e_save:  # catch errors specifically from index.save()
        logger.error(
            f"Failed to save USearch index to temporary file {temp_index_path}: {e_save}"
        )
        if temp_index_path.exists():
            try:
                temp_index_path.unlink()
                logger.debug(
                    f"Cleaned up temporary index file {temp_index_path} after save failure."
                )
            except OSError as e_unlink_save_fail:
                logger.error(
                    f"Failed to clean up temporary index file {temp_index_path} "
                    f"after save failure: {e_unlink_save_fail}"
                )
        raise VectorStoreError(
            f"Failed to save index to temporary file {temp_index_path}"
        ) from e_save

    # step 2: atomic move (only if save to temp succeeded)
    try:
        logger.info(
            f"Attempting to atomically move temporary index from {temp_index_path} to {index_path}"
        )
        os.replace(temp_index_path, index_path)
        logger.info(
            f"Atomically moved temporary index from {temp_index_path} to {index_path}"
        )
    except OSError as e_move:
        logger.error(
            f"Failed to move temporary index {temp_index_path} to {index_path}: {e_move}"
        )
        # temp file still exists here, cleanup is important
        if temp_index_path.exists():
            try:
                temp_index_path.unlink()
                logger.debug(
                    f"Cleaned up temporary index file {temp_index_path} after move failure."
                )
            except OSError as e_unlink_move_fail:
                logger.error(
                    f"Failed to clean up temporary index file {temp_index_path} "
                    f"after move failure: {e_unlink_move_fail}"
                )
        raise VectorStoreError(
            f"Failed to finalize saving index to {index_path}"
        ) from e_move
    finally:
        # final check for temp file, in case an error occurred before os.replace but after save,
        # or if an unexpected error (not oserror from replace, not exception from save) occurred.
        if temp_index_path.exists():
            try:
                temp_index_path.unlink()
                logger.warning(
                    f"Lingering temporary index file {temp_index_path} found and removed in finally block."
                )
            except OSError as e_unlink_final:
                logger.error(
                    f"Failed to clean up lingering temporary index file {temp_index_path} in finally block: {e_unlink_final}"
                )