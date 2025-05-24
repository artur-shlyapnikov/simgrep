import sys
from pathlib import Path
from typing import List, Optional


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


def format_paths(
    file_paths: List[Path],
    use_relative: bool,
    base_path: Optional[Path]
) -> str:
    """
    Formats a list of file paths for display. Paths are unique and sorted.
    Can display absolute paths or paths relative to a base_path.

    Args:
        file_paths: A list of Path objects.
        use_relative: If True, attempts to make paths relative to base_path.
        base_path: The Path object to make paths relative to. Required if use_relative is True.

    Returns:
        A string containing newline-separated file paths, or a message if no paths are found.
    """
    if not file_paths:
        return "No matching files found."

    # Ensure all paths are absolute, then unique and sorted.
    # Paths from retrieve_chunk_for_display should already be absolute and resolved.
    unique_absolute_paths = sorted(list(set(p.resolve() for p in file_paths)))
    
    output_paths_str_list: List[str] = []

    if use_relative:
        if base_path is None:
            print(
                "Warning (simgrep internal): base_path was not provided to format_paths "
                "when use_relative was True. Defaulting to absolute paths.",
                file=sys.stderr
            )
            # Fallback to absolute paths for this call
            for p_abs in unique_absolute_paths:
                output_paths_str_list.append(str(p_abs))
        else:
            abs_base_path = base_path.resolve()
            for p_abs in unique_absolute_paths:
                try:
                    # Ensure p_abs is also resolved before making it relative
                    output_paths_str_list.append(str(p_abs.resolve().relative_to(abs_base_path)))
                except ValueError:
                    # Fallback to absolute path if it cannot be made relative
                    # (e.g., different drive on Windows, or not a subpath)
                    output_paths_str_list.append(str(p_abs.resolve()))
    else: # use_relative is False
        for p_abs in unique_absolute_paths:
            output_paths_str_list.append(str(p_abs.resolve()))
    
    return "\n".join(output_paths_str_list)