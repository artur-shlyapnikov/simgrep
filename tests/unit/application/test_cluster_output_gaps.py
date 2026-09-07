"""Round-13 rendering-contract gaps for render_cluster_outcome (simgrep/output.py)."""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from simgrep.models import ClusterMember, ClustersOutcome, SemanticCluster
from simgrep.output import render_cluster_outcome


def _outcome(total_found: int) -> ClustersOutcome:
    cluster = SemanticCluster(
        members=(
            ClusterMember(label=1, file_path="/r/a.py", line_start=1, line_end=3),
            ClusterMember(label=2, file_path="/r/b.py", line_start=1, line_end=3),
        ),
        score=0.9,
        duplicated_lines=6,
    )
    return ClustersOutcome(clusters=(cluster,), total_found=total_found, chunks_scanned=2)


def test_rich_header_notes_hidden_clusters_when_cap_bites() -> None:
    buf = io.StringIO()
    render_cluster_outcome(_outcome(total_found=3), format="rich", console=Console(file=buf, width=200))
    text = buf.getvalue()
    assert "3 found" in text
    assert "1 shown" in text


def test_rich_header_has_no_shown_note_without_cap() -> None:
    buf = io.StringIO()
    render_cluster_outcome(_outcome(total_found=1), format="rich", console=Console(file=buf, width=200))
    text = buf.getvalue()
    assert "1 found" in text
    assert "shown" not in text


def test_count_prints_precap_total_even_when_capped(capsys: pytest.CaptureFixture[str]) -> None:
    render_cluster_outcome(_outcome(total_found=7), format="count")
    assert capsys.readouterr().out.strip() == "7"
