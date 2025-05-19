from pathlib import Path
from typing import List, TypedDict, cast

import numpy as np
import unstructured.partition.auto as auto_partition
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from unstructured.documents.elements import Element


class ProcessedChunkInfo(TypedDict):
    text: str
    start_char_offset: int
    end_char_offset: int
    token_count: int


def extract_text_from_file(file_path: Path) -> str:
    """
    Extracts text content from a given file path using unstructured.
    Raises FileNotFoundError if the file does not exist or is not a file.
    Raises RuntimeError if unstructured fails to process the file.
    """
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"File not found or is not a file: {file_path}")

    try:
        elements: List[Element] = auto_partition.partition(filename=str(file_path))
        extracted_texts = [el.text for el in elements if hasattr(el, "text")]
        return "\n".join(extracted_texts)
    except Exception as e:
        print(f"Error processing file {file_path} with unstructured: {e}")
        raise RuntimeError(f"Failed to extract text from {file_path}") from e


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """
    Loads a Hugging Face tokenizer.
    """
    try:
        # AutoTokenizer.from_pretrained is typed to return Union[PreTrainedTokenizerFast, PreTrainedTokenizer]
        # both of which are subtypes of PreTrainedTokenizerBase.
        # We cast to satisfy mypy strict checks when it might infer Any.
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return cast(PreTrainedTokenizerBase, tokenizer)
    except OSError as e:
        raise RuntimeError(
            f"Failed to load tokenizer for model '{model_name}'. "
            "Ensure the model name is correct and an internet connection "
            "is available for the first download. Original error: {e}"
        ) from e


def chunk_text_by_tokens(
    full_text: str,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> List[ProcessedChunkInfo]:
    """
    Splits a given text into a list of overlapping token-based chunks.
    """
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be a positive integer.")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be a non-negative integer.")
    if overlap_tokens >= chunk_size_tokens:
        raise ValueError("overlap_tokens must be less than chunk_size_tokens.")

    if not full_text.strip():
        return []

    # Tokenize the entire text, getting input IDs and offset mappings.
    # add_special_tokens=False is important as we are chunking raw content.
    # The sentence-embedding model will later add its own special tokens (CLS, SEP)
    # when processing the final chunk text for embedding.
    # truncation=False ensures we process the entire text.
    encoding = tokenizer(
        full_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False
    )

    all_token_ids = encoding.input_ids
    all_offsets = encoding.offset_mapping

    if not all_token_ids: # e.g., text contained only characters the tokenizer ignores
        return []

    chunks: List[ProcessedChunkInfo] = []
    step = chunk_size_tokens - overlap_tokens # Must be > 0 due to prior validation

    current_token_idx = 0
    while current_token_idx < len(all_token_ids):
        token_slice_end = current_token_idx + chunk_size_tokens
        
        chunk_token_ids_batch = all_token_ids[current_token_idx : token_slice_end]
        chunk_offsets_batch = all_offsets[current_token_idx : token_slice_end]

        if not chunk_token_ids_batch: # Reached the end, slice is empty
            break

        # Determine character offsets for the current chunk
        # The first token's start offset is the chunk's start
        start_char = chunk_offsets_batch[0][0]
        # The last token's end offset is the chunk's end
        end_char = chunk_offsets_batch[-1][1]
        
        # Decode the token IDs to get the chunk text
        # skip_special_tokens=True is good practice here, though we used add_special_tokens=False
        chunk_text = tokenizer.decode(chunk_token_ids_batch, skip_special_tokens=True)
        
        # Sanity check (optional, can be commented out for performance)
        # if chunk_text != full_text[start_char:end_char]:
        #     print(f"Warning: Decoded text mismatch for chunk. "
        #           f"Tokenizer decoded: '{chunk_text}', "
        #           f"Direct slice: '{full_text[start_char:end_char]}'")

        num_tokens_in_this_chunk = len(chunk_token_ids_batch)

        chunks.append(ProcessedChunkInfo(
            text=chunk_text,
            start_char_offset=start_char,
            end_char_offset=end_char,
            token_count=num_tokens_in_this_chunk
        ))
        
        current_token_idx += step
        
    return chunks


def generate_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Generates vector embeddings for a list of input texts using a specified
    sentence-transformer model.
    """
    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, show_progress_bar=False)
        return embeddings
    except Exception as e:
        error_message = (
            f"Failed to generate embeddings using model '{model_name}'. "
            f"Ensure the model name is correct and an internet connection "
            f"is available for the first download. Original error: {e}"
        )
        raise RuntimeError(error_message) from e