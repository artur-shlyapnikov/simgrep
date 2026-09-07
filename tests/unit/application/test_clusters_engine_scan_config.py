"""Regression: ClustersEngine.clusters_path must honor app_config scan settings.

The ephemeral scan for `simgrep clusters PATH` previously used default
ScanOptions, silently ignoring the user's configured file_patterns
(and max_file_size_bytes / follow_symlinks).
"""

from __future__ import annotations

from pathlib import Path

from simgrep.clusters_engine import ClustersEngine
from simgrep.models import AppConfig
from tests.conftest import FakeEmbedder, FakeTextExtractor, FakeTokenChunker, FakeVectorIndex


class _FakeRuntime:
    def __init__(self) -> None:
        self.extractor = FakeTextExtractor()
        self.chunker = FakeTokenChunker()
        self.embedder = FakeEmbedder()

    def new_vector_index(self, ndim: int) -> FakeVectorIndex:
        return FakeVectorIndex(ndim)


_BODY = "alpha bravo charlie delta\n"


def test_clusters_path_honors_app_config_file_patterns(tmp_path: Path) -> None:
    target = tmp_path / "tree"
    target.mkdir()
    # Two identical markdown files form a duplicate pair; an identical txt file
    # matches the DEFAULT patterns but not the configured ones.
    (target / "one.md").write_text(_BODY, encoding="utf-8")
    (target / "two.md").write_text(_BODY, encoding="utf-8")
    (target / "three.txt").write_text(_BODY, encoding="utf-8")

    app_config = AppConfig(file_patterns=("*.md",))
    outcome = ClustersEngine(_FakeRuntime()).clusters_path(target, app_config)

    assert outcome.chunks_scanned == 2
    clustered_paths = {Path(member.file_path).suffix for cluster in outcome.clusters for member in cluster.members}
    assert clustered_paths == {".md"}
