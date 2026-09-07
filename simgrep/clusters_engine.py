from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from simgrep.clustering import cluster_components
from simgrep.corpus import ChunkBatch, CorpusAccess, StoredChunk
from simgrep.errors import ClustersError
from simgrep.models import (
    AppConfig,
    ClusterMember,
    ClustersOptions,
    ClustersOutcome,
    FreshnessMode,
    ProjectConfig,
)


class ClustersEngine:
    """Semantic duplicate-cluster detection over stored chunk vectors (batch-only)."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def clusters_project(
        self,
        project: ProjectConfig,
        app_config: AppConfig,
        options: ClustersOptions,
        freshness: FreshnessMode = FreshnessMode.auto,
    ) -> ClustersOutcome:
        """Compatibility shim; transports go through simgrep.execution.execute_clusters."""
        with CorpusAccess(self.runtime).open_project(project, app_config, freshness=freshness) as corpus:
            return self.run_batch(corpus.snapshot(), options)

    def clusters_path(self, path: Path, app_config: AppConfig, options: ClustersOptions | None = None) -> ClustersOutcome:
        """Compatibility shim; transports go through simgrep.execution.execute_clusters."""
        if not path.exists():
            raise ClustersError(f"Path not found: {path}", hint="Check the path and try again.")
        with CorpusAccess(self.runtime).open_ephemeral([path], app_config) as corpus:
            return self.run_batch(corpus.snapshot(), options if options is not None else ClustersOptions())

    def run_batch(self, batch: ChunkBatch, options: ClustersOptions) -> ClustersOutcome:
        if not 0 < options.threshold <= 1:
            raise ClustersError(
                f"Threshold must satisfy 0 < threshold <= 1, got {options.threshold}.",
                hint="Pass a similarity threshold between 0 and 1, e.g. 0.85.",
            )
        chunks_scanned = batch.indexed_count
        if chunks_scanned > options.max_chunks:
            raise ClustersError(
                f"Too many chunks to cluster ({chunks_scanned} > {options.max_chunks}).",
                hint="Narrow the scope (e.g. a subdirectory) or raise --max-chunks.",
            )
        members = _gather_members(batch.chunks)
        if not members:
            return ClustersOutcome(clusters=(), total_found=0, chunks_scanned=chunks_scanned)
        kept = sorted(members)
        row_of_label = {label: idx for idx, label in enumerate(batch.labels)}
        ordered = batch.vectors[[row_of_label[label] for label in kept]]
        found, total_found = cluster_components(ordered, {label: members[label] for label in kept}, options)
        return ClustersOutcome(
            clusters=tuple(found[: options.top]),
            total_found=total_found,
            chunks_scanned=chunks_scanned,
        )


def _gather_members(chunks: Iterable[StoredChunk]) -> dict[int, ClusterMember]:
    members: dict[int, ClusterMember] = {}
    for chunk in chunks:
        if chunk.line_start is None or chunk.line_end is None:
            continue
        members[chunk.label] = ClusterMember(
            label=chunk.label,
            file_path=str(chunk.file_path),
            line_start=chunk.line_start,
            line_end=chunk.line_end,
        )
    return members
