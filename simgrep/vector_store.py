from typing import List, Optional, Tuple, Union

import numpy as np
import usearch.index  # For usearch.index.Index, usearch.index.Matches, usearch.index.BatchMatches

# Consider defining constants for default USearch parameters if they are used in multiple places
# For D1.4, defaults in function signatures are sufficient.
# DEFAULT_USEARCH_METRIC = "cos"
# DEFAULT_USEARCH_DTYPE = "f32"


def create_inmemory_index(
    embeddings: np.ndarray,
    metric: str = "cos",
    dtype: str = "f32",
    # accuracy: Optional[float] = None, # Future: for tuning
    # connectivity: Optional[int] = None, # Future: for HNSW M parameter
    # expansion_add: Optional[int] = None, # Future: for HNSW efConstruction
    # expansion_search: Optional[int] = None # Future: for HNSW ef
) -> usearch.index.Index:
    """
    Creates and populates an in-memory USearch index.

    Args:
        embeddings: A 2D NumPy array of shape (num_chunks, embedding_dimension)
                    containing the vector embeddings for text chunks.
                    This function assumes `embeddings` is not empty and is correctly shaped.
                    The caller (e.g., main.py) should handle empty/invalid `embeddings`
                    before calling this function.
        metric: The distance metric for USearch (e.g., "cos", "ip", "l2sq").
        dtype: The data type of vectors in the index (e.g., "f32", "f16").

    Returns:
        A populated usearch.index.Index object.

    Raises:
        ValueError: If `embeddings` is not a 2D NumPy array or if `ndim` cannot be determined.
                    (Though pre-checks by caller should prevent this).
    """
    # Basic sanity check, though primary validation is expected from the caller
    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D NumPy array.")
    if embeddings.shape[0] == 0:
        # This case should ideally be handled by the caller to avoid creating an index
        # with an unknown ndim if embeddings.shape[1] can't be accessed.
        # If it must be handled here, ndim would need to be passed or defaulted.
        # For D1.4, caller handles this.
        raise ValueError("Embeddings array cannot be empty when creating an index.")

    num_vectors: int = embeddings.shape[0]
    num_dimensions: int = embeddings.shape[1]

    index = usearch.index.Index(
        ndim=num_dimensions,
        metric=metric,
        dtype=dtype,
        # accuracy=accuracy, # For future tuning
        # connectivity=connectivity, # M for HNSW
        # expansion_add=expansion_add, # efConstruction for HNSW
        # expansion_search=expansion_search # ef for HNSW
    )

    # Prepare Labels: Use the original 0-based index of each chunk embedding as its label in USearch.
    # USearch expects np.int64 for labels (referred to as keys in API).
    labels = np.arange(num_vectors, dtype=np.int64)

    # Add Embeddings to Index:
    index.add(keys=labels, vectors=embeddings)  # `keys` is the parameter name in usearch for labels

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
    """
    if not isinstance(index, usearch.index.Index):
        raise ValueError("Provided index is not a valid usearch.index.Index object.")
    if not isinstance(query_embedding, np.ndarray):
        raise ValueError("Query embedding must be a NumPy array.")
    if k <= 0:
        raise ValueError("k (number of results) must be a positive integer.")

    # If index is empty, no search can be performed
    if len(index) == 0:  # `len(index)` gives the number of items in the USearch index
        return []

    processed_query_embedding = query_embedding
    if processed_query_embedding.ndim == 1:
        processed_query_embedding = np.expand_dims(processed_query_embedding, axis=0)  # Reshape (D,) to (1, D)

    if processed_query_embedding.shape[0] != 1:  # Ensure it's a single query for this simple case
        raise ValueError(f"Expected a single query embedding, but got batch of {processed_query_embedding.shape[0]}.")
    if processed_query_embedding.shape[1] != index.ndim:
        raise ValueError(
            f"Query embedding dimension ({processed_query_embedding.shape[1]}) "
            f"does not match index dimension ({index.ndim})."
        )

    # `count` is the parameter name for k in usearch's search method
    # `index.search` can return `Matches` (for 1D input) or `BatchMatches` (for 2D input).
    # Our `processed_query_embedding` is always 2D (1, ndim).
    search_result: Union[usearch.index.Matches, usearch.index.BatchMatches] = index.search(
        vectors=processed_query_embedding, count=k
    )

    results: List[Tuple[int, float]] = []

    actual_keys: Optional[np.ndarray] = None
    actual_distances: Optional[np.ndarray] = None
    num_found_for_query: int = 0

    if isinstance(search_result, usearch.index.BatchMatches):
        # This is the expected path for (1, ndim) input.
        # search_result.counts is a np.ndarray of shape (batch_size,)
        if search_result.counts is not None and len(search_result.counts) > 0:
            num_found_for_query = search_result.counts[0]
            if num_found_for_query > 0:
                # search_result.keys is np.ndarray of objects (arrays), shape (batch_size,)
                actual_keys = search_result.keys[0]
                actual_distances = search_result.distances[0]
    elif isinstance(search_result, usearch.index.Matches):
        # This path would be taken if input was 1D (ndim,).
        # search_result.counts is an int for Matches.
        # search_result.keys and .distances are 1D np.ndarrays.
        if search_result.counts is not None: # Check for None although it's typed as int
             num_found_for_query = search_result.counts
        if num_found_for_query > 0: # Check num_found_for_query directly
            actual_keys = search_result.keys
            actual_distances = search_result.distances
    
    if num_found_for_query > 0 and actual_keys is not None and actual_distances is not None:
        for i in range(num_found_for_query):
            key: int = int(actual_keys[i])
            distance: float = float(actual_distances[i])

            similarity: float
            metric_str = str(index.metric).lower()

            if "cos" in metric_str:
                similarity = 1.0 - distance
            elif "ip" in metric_str:
                similarity = -distance
            elif "l2" in metric_str:
                similarity = 1.0 / (1.0 + distance)
            else:
                print(
                    f"Warning: Unknown metric '{index.metric}' for similarity conversion. Returning raw negative distance."
                )
                similarity = -distance
            results.append((key, similarity))

    return results