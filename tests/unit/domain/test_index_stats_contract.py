"""Contract pin: IndexStats timing fields (scan_seconds, hash_seconds,
extract_chunk_seconds, embedding_seconds, store_seconds, index_save_seconds)
were dead weight and are removed. This pins the surviving field set so the
fields cannot silently return and consumers cannot silently lose fields."""

from __future__ import annotations

import dataclasses

from simgrep.models import IndexStats


def test_index_stats_field_set_pins_timing_removal() -> None:
    assert tuple(f.name for f in dataclasses.fields(IndexStats)) == (
        "files_seen",
        "files_processed",
        "files_indexed",
        "files_skipped_unchanged",
        "files_skipped_too_large",
        "files_pruned_deleted",
        "ignored_count",
        "unreadable_count",
        "chunks_indexed",
        "vectors_added",
        "vectors_removed",
        "index_mutated",
        "errors",
        "plan_seconds",
        "total_seconds",
    )
