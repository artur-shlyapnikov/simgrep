"""Round-13 edge coverage for ClustersEngine.run_batch (fakes only, no CLI)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simgrep.clusters_engine import ClustersEngine
from simgrep.corpus import ChunkBatch, StoredChunk
from simgrep.errors import ClustersError
from simgrep.models import ClustersOptions, FileRole
from tests.conftest import FakeRuntime


def _batch(tmp_path: Path, labels: list[int], vectors: np.ndarray, file_names: list[str]) -> ChunkBatch:
    """Aligned batch preserving the caller's label order; labels need not be 1..N."""
    chunks = tuple(
        StoredChunk(
            label=label,
            file_id=i + 1,
            file_path=tmp_path / name,
            text=f"t{label}",
            start_char=0,
            end_char=2,
            token_count=1,
            line_start=i * 10 + 1,
            line_end=i * 10 + 5,
            role=FileRole.source,
            language="python",
        )
        for i, (label, name) in enumerate(zip(labels, file_names, strict=True))
    )
    return ChunkBatch(chunks=chunks, vectors=vectors.astype(np.float32), indexed_count=len(chunks))


def test_gapped_labels_survive_row_remap(tmp_path: Path) -> None:
    """Vector labels need not be 1..N nor inserted in sorted order; the engine's
    label->row remap must pair each label with its own vector."""
    vectors = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
    labels = [50, 7, 23, 91]
    files = ["z.py", "a.py", "c.py", "d.py"]
    batch = _batch(tmp_path, labels, vectors, files)

    outcome = ClustersEngine(FakeRuntime()).run_batch(batch, ClustersOptions())

    assert outcome.chunks_scanned == 4
    # Two independent duplicate pairs: {50, 7} along e1 and {23, 91} along e2.
    assert outcome.total_found == 2
    cluster = next(c for c in outcome.clusters if {m.label for m in c.members} == {50, 7})
    assert cluster.score == pytest.approx(1.0)
    # Members stay ordered by (file_path, line_start, label).
    assert [(Path(m.file_path).name, m.line_start) for m in cluster.members] == [
        ("a.py", 11),
        ("z.py", 1),
    ]


@pytest.mark.xfail(
    strict=True,
    reason="Product bug (probed at f24c943): run_batch returns the empty outcome "
    "before cluster_components runs, so invalid options (min_size<2, threshold "
    "outside (0,1]) are silently accepted on an empty corpus; validation must be "
    "input-independent.",
)
def test_empty_corpus_still_validates_options(tmp_path: Path) -> None:
    batch = ChunkBatch(chunks=(), vectors=np.zeros((0, 4), dtype=np.float32), indexed_count=0)
    with pytest.raises(ClustersError):
        ClustersEngine(FakeRuntime()).run_batch(batch, ClustersOptions(min_size=1))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Product bug (probed at 0fc43df): run_batch applies the top cap as a bare "
        "slice (found[: options.top]) with no validation, so top=-1 silently drops "
        "the last cluster (and top<=0 empties the outcome) instead of raising "
        "ClustersError; min_size and threshold ARE validated."
    ),
)
def test_negative_top_raises_clusters_error(tmp_path: Path) -> None:
    batch = _batch(
        tmp_path,
        labels=[1, 2],
        vectors=np.array([[1, 0], [1, 0]], dtype=np.float32),
        file_names=["a.py", "b.py"],
    )
    with pytest.raises(ClustersError):
        ClustersEngine(FakeRuntime()).run_batch(batch, ClustersOptions(top=-1))
