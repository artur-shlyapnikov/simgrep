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

# Conditional import based on how the script is run
if __package__ is None or __package__ == "":
    # Executed as a script (e.g., python simgrep/main.py)
    # Assumes processor.py is in the same directory.
    from processor import extract_text_from_file
else:
    # Imported as part of a package (e.g., python -m simgrep.main or when installed)
    from .processor import extract_text_from_file

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
        help="Show version and exit."
    )
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
        exists=False, # Using False to allow custom error handling below, as per existing code.
        file_okay=True, # Path must be a file.
        dir_okay=False, # Path must not be a directory.
        # readable=True, # Typer would check this. If custom error needed, handle manually.
        resolve_path=True,
        help="The path to the text file to search within."
    )
):
    """
    Searches for a query within a specified text file.
    (Placeholder for D1.1 - focuses on text extraction)
    """
    console.print(f"Searching for: '[bold blue]{query_text}[/bold blue]' in file: '[green]{path_to_search}[/green]'")

    try:
        # Manual checks because exists=False was chosen for path_to_search Argument.
        if not path_to_search.exists():
            console.print(f"[bold red]Error: File not found: {path_to_search}[/bold red]")
            raise typer.Exit(code=1)
        if not path_to_search.is_file():
            console.print(f"[bold red]Error: Path provided is not a file: {path_to_search}[/bold red]")
            raise typer.Exit(code=1)
        
        # Optional: Check for readability if Typer's readable=True is not used.
        # import os
        # if not os.access(path_to_search, os.R_OK):
        #     console.print(f"[bold red]Error: File is not readable: {path_to_search}[/bold red]")
        #     raise typer.Exit(code=1)

        extracted_content = extract_text_from_file(path_to_search)
        console.print(f"\n[bold]Extracted Content (first 500 chars):[/bold]")
        console.print(extracted_content[:500] + "..." if len(extracted_content) > 500 else extracted_content)
        # Later phases will chunk, embed, and search this content.
    except FileNotFoundError as e: # This might be redundant if using path_to_search.exists() check above
        console.print(f"[bold red]Error: {e}[/bold red]") # processor.py also raises FileNotFoundError
        raise typer.Exit(code=1)
    except RuntimeError as e: # Catching the re-raised error from processor
        console.print(f"[bold red]Error during text extraction: {e}[/bold red]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()