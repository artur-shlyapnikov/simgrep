import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from transformers import PreTrainedTokenizerBase

from simgrep.adapters.hf_token_chunker import HFTokenChunker, load_tokenizer

pytest.importorskip("transformers")
pytest.importorskip("sentence_transformers")


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizerBase:
    return load_tokenizer(MODEL_NAME)


@settings(max_examples=200, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
@given(
    text=st.text(alphabet=st.characters(max_codepoint=0x10FFFF, blacklist_categories=("Cc", "Cs"))),
    chunk_size_tokens=st.integers(min_value=1, max_value=20),
    overlap_tokens=st.integers(min_value=0, max_value=19),
)
def test_chunk_text_roundtrip(tokenizer: PreTrainedTokenizerBase, text: str, chunk_size_tokens: int, overlap_tokens: int) -> None:
    assume(overlap_tokens < chunk_size_tokens)
    assume(text == text.strip())

    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = encoding.input_ids
    all_offsets = encoding.offset_mapping
    expected_decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)

    # To test the roundtrip logic, we force the chunker to produce only one chunk
    # that covers all the tokens. This simplifies checking offsets and content.
    chunk_size_for_one_chunk = max(chunk_size_tokens, len(token_ids) + 1)

    chunker = HFTokenChunker(
        model_name=MODEL_NAME,
        chunk_size=chunk_size_for_one_chunk,
        overlap=overlap_tokens,
    )
    chunks = chunker.chunk(text)

    if not token_ids:
        assert chunks == []
        return

    # It's possible for text to produce tokens which then decode to an empty string.
    # In this case, we might get a single chunk with empty text.
    if not chunks:
        assert expected_decoded_text == ""
        return

    assert len(chunks) == 1
    chunk = chunks[0]

    # The decoded text from the chunk should match the decoded text from all original tokens
    assert chunk.text == expected_decoded_text

    if all_offsets:
        assert chunk.start_char_offset == all_offsets[0][0]
        assert chunk.end_char_offset == all_offsets[-1][1]
        assert chunk.start_char_offset <= chunk.end_char_offset
