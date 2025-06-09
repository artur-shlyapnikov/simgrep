import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console


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
    formatted_score = f"{score:.4f}"  # format score to 4 decimal places
    return f"File: {str(file_path)}\nScore: {formatted_score}\nChunk: {chunk_text}"


def format_paths(
    file_paths: List[Path],
    use_relative: bool,
    base_path: Optional[Path],
    *,
    console: Optional[Console] = None,
) -> str:
    """
    Formats a list of file paths for display. Paths are unique and sorted.
    Can display absolute paths or paths relative to a base_path.
    Optionally accepts a Rich ``Console`` instance for emitting warnings.

    Args:
        file_paths: A list of Path objects.
        use_relative: If True, attempts to make paths relative to base_path.
        base_path: The Path object to make paths relative to. Required if ``use_relative`` is ``True``.
        console: Optional ``Console`` used for warning messages. If ``None`` a new ``Console`` is created.

    Returns:
        A string containing newline-separated file paths, or a message if no paths are found.
    """
    if not file_paths:
        return "No matching files found."

    # ensure all paths are absolute, then unique and sorted.
    # paths from retrieve_chunk_for_display should already be absolute and resolved.
    unique_absolute_paths = sorted(list(set(p.resolve() for p in file_paths)))

    output_paths_str_list: List[str] = []

    if console is None:
        console = Console()

    if use_relative:
        if base_path is None:
            console.print(
                "Warning (simgrep internal): base_path was not provided to format_paths " "when use_relative was True. Defaulting to absolute paths.",
                style="yellow",
            )
            # fallback to absolute paths for this call
            for p_abs in unique_absolute_paths:
                output_paths_str_list.append(str(p_abs))
        else:
            abs_base_path = base_path.resolve()
            for p_abs in unique_absolute_paths:
                try:
                    # ensure p_abs is also resolved before making it relative
                    output_paths_str_list.append(str(p_abs.resolve().relative_to(abs_base_path)))
                except ValueError:
                    # fallback to absolute path if it cannot be made relative
                    # (e.g., different drive on windows, or not a subpath)
                    output_paths_str_list.append(str(p_abs.resolve()))
    else:  # use_relative is false
        for p_abs in unique_absolute_paths:
            output_paths_str_list.append(str(p_abs.resolve()))

    return "\n".join(output_paths_str_list)


def format_json(results: List[Dict[str, Any]]) -> str:
    """
    Formats a list of result dictionaries as a JSON string.
    Converts Path objects to strings for serialization.

    Args:
        results: A list of dictionaries, where each dictionary represents a search result.

    Returns:
        A JSON formatted string.
    """
    if not results:
        return "[]"

    serializable_results = []
    for res in results:
        serializable_res = res.copy()
        for key, value in serializable_res.items():
            if isinstance(value, Path):
                serializable_res[key] = str(value)
        serializable_results.append(serializable_res)

    return json.dumps(serializable_results, indent=2)


def format_count(results: List[Dict[str, Any]]) -> str:
    """
    Formats the search results into a count summary.

    Args:
        results: A list of dictionaries, where each dictionary represents a search result.

    Returns:
        A string summarizing the number of matching chunks and files.
    """
    num_chunks = len(results)
    if num_chunks == 0:
        return "0 matching chunks in 0 files."

    # Assumes 'file_path' is a key in each result dictionary.
    # The searcher logic ensures this.
    unique_files = {res["file_path"] for res in results if "file_path" in res}
    num_files = len(unique_files)

    chunk_str = "chunk" if num_chunks == 1 else "chunks"
    file_str = "file" if num_files == 1 else "files"

    return f"{num_chunks} matching {chunk_str} in {num_files} {file_str}."
