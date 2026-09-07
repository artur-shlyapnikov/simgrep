from pathlib import Path

from simgrep.models import SCHEMA_VERSION, AppConfig, FilePlan, FilePlanEntry, FreshnessMode, ProjectConfig


def test_app_config_defaults() -> None:
    cfg = AppConfig()
    assert cfg.model == "ibm-granite/granite-embedding-30m-english"
    assert cfg.chunk_size == 128
    assert cfg.chunk_overlap == 20
    assert cfg.batch_size == 128
    assert cfg.freshness == FreshnessMode.auto


def test_project_config_artifact_paths() -> None:
    root = Path("/tmp/demo-root")
    project = ProjectConfig(SCHEMA_VERSION, "demo", root, (root,), "fake", 128, 20)
    assert project.metadata_db_path == root / ".simgrep" / "metadata.duckdb"
    assert project.vector_index_path == root / ".simgrep" / "vectors.usearch"
    assert project.index_lock_path == root / ".simgrep" / "index.lock"


def test_file_plan_mutation_detection() -> None:
    plan = FilePlan(entries=(FilePlanEntry(path=Path("a.py"), status="new"), FilePlanEntry(path=Path("b.py"), status="unchanged")))
    assert plan.has_mutations
    deleted_only = FilePlan(entries=(FilePlanEntry(path=Path("c.py"), status="deleted"),))
    assert deleted_only.has_mutations
    assert not deleted_only.has_indexable_work
    assert not FilePlan(entries=()).has_indexable_work
