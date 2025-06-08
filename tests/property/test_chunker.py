import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st
from transformers import PreTrainedTokenizerBase

pytest.importorskip("transformers")
pytest.importorskip("sentence_transformers")

from simgrep.processor import load_tokenizer, chunk_text_by_tokens

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizerBase:
    return load_tokenizer(MODEL_NAME)

@settings(max_examples=50, suppress_health_check=[HealthCheck.filter_too_much])
@given(
    text=st.text(),
    chunk_size_tokens=st.integers(min_value=1, max_value=20),
    overlap_tokens=st.integers(min_value=0, max_value=19),
)
def test_chunk_text_roundtrip(tokenizer: PreTrainedTokenizerBase, text: str, chunk_size_tokens: int, overlap_tokens: int) -> None:
    assume(overlap_tokens < chunk_size_tokens)
    assume(text == text.strip())

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    expected = tokenizer.decode(token_ids, skip_special_tokens=True)
    assume(expected == text.lower())

    # ensure the text fits entirely in one chunk so overlap does not
    # introduce duplicate text when concatenating chunk contents
    # ensure only one chunk is produced to avoid duplicate text when
    # concatenating overlapping chunks
    chunk_size_tokens = max(chunk_size_tokens, len(token_ids) + overlap_tokens)

    chunks = chunk_text_by_tokens(text, tokenizer, chunk_size_tokens, overlap_tokens)

    if expected == "":
        assert chunks == []
        return

    concatenated = "".join(chunk["text"] for chunk in chunks)
    assert concatenated == expected
    assert chunks[0]["start_char_offset"] == 0
    assert chunks[-1]["end_char_offset"] == len(text)
    for chunk in chunks:
        assert chunk["start_char_offset"] <= chunk["end_char_offset"]
