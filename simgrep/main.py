import typer
from pathlib import Path
from rich.console import Console
import warnings

# Filter the specific UserWarning from unstructured regarding libmagic
# This warning is advisory and unstructured can often proceed without libmagic
# for common file types.
warnings.filterwarnings(
    "ignore",
    message="libmagic is unavailable but assists in filetype detection. Please consider installing libmagic for better results.",
    # category=UserWarning # category can be added if we are sure it's always UserWarning
)

# Use relative import for processor module
from .processor import extract_text_from_file, chunk_text_simple, generate_embeddings
import numpy as np # For type hinting (np.ndarray)

__version__ = "0.1.0"  # Placeholder version, can be updated or managed elsewhere

app = typer.Typer()
console = Console()


def version_callback(value: bool):
    if value:
        console.print(f"simgrep version: {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: bool = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
):
    """
    simgrep CLI application.
    """
    pass


@app.command()
def search(
    query_text: str = typer.Argument(..., help="The text or concept to search for."),
    path_to_search: Path = typer.Argument(
        ...,
        exists=False,  # Using False to allow custom error handling below, as per existing code.
        file_okay=True,  # Path must be a file.
        dir_okay=False,  # Path must not be a directory.
        # readable=True, # Typer would check this. If custom error needed, handle manually.
        resolve_path=True,
        help="The path to the text file to search within.",
    ),
):
    """
    Searches for a query within a specified text file.
    (Placeholder for D1.1 - focuses on text extraction)
    """
    console.print(
        f"Searching for: '[bold blue]{query_text}[/bold blue]' in file: '[green]{path_to_search}[/green]'"
    )

    try:
        # Manual checks because exists=False was chosen for path_to_search Argument.
        if not path_to_search.exists():
            console.print(
                f"[bold red]Error: File not found: {path_to_search}[/bold red]"
            )
            raise typer.Exit(code=1)
        if not path_to_search.is_file():
            console.print(
                f"[bold red]Error: Path provided is not a file: {path_to_search}[/bold red]"
            )
            raise typer.Exit(code=1)

        # Optional: Check for readability if Typer's readable=True is not used.
        # import os
        # if not os.access(path_to_search, os.R_OK):
        #     console.print(f"[bold red]Error: File is not readable: {path_to_search}[/bold red]")
        #     raise typer.Exit(code=1)

        extracted_content = extract_text_from_file(path_to_search)
        console.print(f"\n[bold]Extracted Content (first 500 chars):[/bold]")
        console.print(
            extracted_content[:500] + "..."
            if len(extracted_content) > 500
            else extracted_content
        )

        # --- BEGIN NEW CODE FOR D1.2 ---
        # Define hardcoded chunking parameters for now
        chunk_size_chars: int = 200  # Example value
        overlap_chars: int = 50  # Example value

        console.print(
            f"\n[bold]Chunking parameters:[/bold] Size={chunk_size_chars} chars, Overlap={overlap_chars} chars"
        )

        try:
            # Need to import List for type hint if not already done, but it's usually available with Typer/Pydantic
            from typing import (
                List,
            )  # Add if not available, though often implicitly available

            text_chunks: List[str] = chunk_text_simple(
                text=extracted_content,
                chunk_size_chars=chunk_size_chars,
                overlap_chars=overlap_chars,
            )

            console.print(f"\n[bold]Generated Chunks ({len(text_chunks)}):[/bold]")
            if not text_chunks:
                console.print(
                    "No chunks generated (text might be empty or too short for parameters)."
                )
            else:
                for i, chunk_item in enumerate(text_chunks):
                    # Displaying chunk with a bit of context, limited length for readability
                    display_chunk = (
                        chunk_item[:100] + "..."
                        if len(chunk_item) > 100
                        else chunk_item
                    )
                    console.print(
                        f'  [cyan]Chunk {i + 1}:[/cyan] "{display_chunk}" (Length: {len(chunk_item)})'
                    )

        except ValueError as ve:
            console.print(f"[bold red]Error during chunking: {ve}[/bold red]")
            raise typer.Exit(code=1)
        # --- END NEW CODE FOR D1.2 ---

        # --- BEGIN NEW CODE FOR D1.3 ---
        query_embedding: np.ndarray
        chunk_embeddings: np.ndarray = np.array([]) # Initialize to an empty array

        console.print(f"\n[bold]Step 3: Generating Embeddings[/bold]")
        # Model name is currently hardcoded in generate_embeddings default.
        # Future phases will make this configurable.
        # Inform user about potential first-time model download.
        console.print(
            "  (This may take a moment on first run if the embedding model needs to be downloaded...)"
        )

        try:
            # 1. Embed the query
            console.print(f"  Embedding query: '[italic blue]{query_text}[/italic blue]'")
            query_embedding = generate_embeddings(texts=[query_text]) # Query text must be in a list
            console.print(f"    Query embedding shape: {query_embedding.shape}")

            # 2. Embed the text chunks (if any)
            if text_chunks:
                console.print(f"  Embedding {len(text_chunks)} text chunk(s)...")
                chunk_embeddings = generate_embeddings(texts=text_chunks)
                console.print(f"    Chunk embeddings shape: {chunk_embeddings.shape}")

                # Verification: For D1.3, ensure both query and chunks have same embedding dimension
                # This check is only valid if chunk_embeddings were actually produced.
                if chunk_embeddings.size > 0 and query_embedding.shape[1] != chunk_embeddings.shape[1]:
                    console.print(
                        f"[bold red]Error: Query embedding dimension ({query_embedding.shape[1]}) "
                        f"does not match chunk embedding dimension ({chunk_embeddings.shape[1]}). "
                        f"This should not happen with the same model.[/bold red]"
                    )
                    raise typer.Exit(code=1)
            else:
                console.print("  No text chunks found in the file to embed.")
                # chunk_embeddings remains an empty np.array as initialized

            # At this point, query_embedding and chunk_embeddings are available.
            # For D1.3, we've printed their shapes. Subsequent deliverables will use them.

        except RuntimeError as e: # Catch the specific RuntimeError from generate_embeddings
            console.print(f"[bold red]Embedding Generation Failed:[/bold red] {e}")
            raise typer.Exit(code=1)
        except Exception as e: # Catch any other unexpected errors
            console.print(f"[bold red]An unexpected error occurred during embedding: {e}[/bold red]")
            # Consider logging traceback here for debugging if --verbose is implemented later
            raise typer.Exit(code=1)
        # --- END NEW CODE FOR D1.3 ---

        # Later phases will embed and search these chunks.
    except (
        FileNotFoundError
    ) as e:  # This might be redundant if using path_to_search.exists() check above
        console.print(
            f"[bold red]Error: {e}[/bold red]"
        )  # processor.py also raises FileNotFoundError
        raise typer.Exit(code=1)
    except RuntimeError as e:  # Catching the re-raised error from processor
        console.print(f"[bold red]Error during text extraction: {e}[/bold red]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()