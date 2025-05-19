from pathlib import Path


def format_show_basic(file_path: Path, chunk_text: str, score: float) -> str:
    """
    Formats a single search result for basic "show" output.

    Args:
        file_path: The path to the file containing the chunk.
        chunk_text: The text of the relevant chunk.
        score: The similarity score of the chunk.

    Returns:
        A formatted string representing the search result.
    """
    formatted_score = f"{score:.4f}"  # Format score to 4 decimal places
    return f"File: {str(file_path)}\nScore: {formatted_score}\nChunk: {chunk_text}"
