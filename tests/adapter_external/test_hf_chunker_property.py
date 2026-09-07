import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from simgrep.adapters.chunker import HFChunker

pytestmark = pytest.mark.external


@settings(max_examples=25, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
@given(
    text=st.text(alphabet=st.characters(max_codepoint=0x10FFFF, blacklist_categories=("Cc", "Cs"))),
    chunk_size_tokens=st.integers(min_value=1, max_value=20),
    overlap_tokens=st.integers(min_value=0, max_value=19),
)
def test_chunk_text_roundtrip(hf_chunker: HFChunker, text: str, chunk_size_tokens: int, overlap_tokens: int) -> None:
    assume(overlap_tokens < chunk_size_tokens)
    assume(text == text.strip())

    tokenizer = hf_chunker._tokenizer
    assert tokenizer is not None
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = encoding.input_ids
    expected_decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    assert isinstance(expected_decoded_text, str)
    if not token_ids:
        chunker = HFChunker(model_name=hf_chunker.model_name, chunk_size=128, overlap=20)
        assert chunker.chunk(text) == []
        return

    chunk_size_for_one_chunk = len(token_ids) + overlap_tokens
    chunker_for_test = HFChunker(
        model_name=hf_chunker.model_name,
        chunk_size=chunk_size_for_one_chunk,
        overlap=overlap_tokens,
    )
    chunks = chunker_for_test.chunk(text)

    if not chunks:
        cleaned = expected_decoded_text
        for ch in ("\u200b", "\u200c", "\u200d", "\ufeff"):
            cleaned = cleaned.replace(ch, "")
        assert cleaned.strip() == ""
        return

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.start <= chunk.end
    assert chunk.start >= 0
    assert chunk.end <= len(text)
    assert chunk.text == text[chunk.start : chunk.end]
