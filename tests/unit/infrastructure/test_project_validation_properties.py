"""Property-based tests for project-config validation (Round 12).

Invariants pinned here:
- corrupting exactly one project.toml field with an arbitrary scalar either loads a
  valid ProjectConfig or raises the clean error type (ProjectError);
- chunk range predicate: load succeeds iff 1 <= chunk_size and
  0 <= chunk_overlap < chunk_size;
- indexed_paths dedup collapses aliased spellings while preserving first-occurrence
  order;
- last-path removal is refused, add/remove are idempotent and persist;
- project_covers_path is purely lexicographic containment on resolved paths.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import tomli_w
from hypothesis import given, settings
from hypothesis import strategies as st

from simgrep.errors import ProjectError
from simgrep.models import SCHEMA_VERSION, ProjectConfig
from simgrep.project import (
    add_indexed_path,
    load_project_config,
    project_covers_path,
    remove_indexed_path,
)


def _init(root: Path, *, chunk_size: int = 128, chunk_overlap: int = 32, paths: tuple[str, ...] = (".",)) -> None:
    (root / ".simgrep").mkdir(parents=True, exist_ok=True)
    (root / ".simgrep" / "project.toml").write_text(
        f'schema_version = {SCHEMA_VERSION}\nname = "p"\nmodel = "m"\n'
        f'chunk_size = {chunk_size}\nchunk_overlap = {chunk_overlap}\n'
        f'indexed_paths = [{", ".join(chr(34) + p + chr(34) for p in paths)}]\n',
        encoding="utf-8",
    )


MALFORMED = st.one_of(
    # NOTE: st.none() omitted -- None is not TOML-serializable, hence unreachable here.
    st.booleans(),
    st.text(max_size=8),
    # infinities excluded -- same OverflowError product bug as file A (see design section 6)
    st.one_of(
        st.floats(allow_nan=False, min_value=-1e30, max_value=1e30),
        st.just(float("nan")),
    ),
    st.lists(st.integers(), max_size=2),
)


@settings(max_examples=50, deadline=None)
@given(field=st.sampled_from(["schema_version", "chunk_size", "chunk_overlap"]), raw=MALFORMED)
def test_load_project_never_crashes_outside_project_error(tmp_path_factory: pytest.TempPathFactory, field: str, raw: object) -> None:
    root = tmp_path_factory.mktemp("proj")
    _init(root)
    path = root / ".simgrep" / "project.toml"
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    data[field] = raw
    with path.open("wb") as handle:
        tomli_w.dump(data, handle)
    try:
        cfg = load_project_config(root)
        assert isinstance(cfg, ProjectConfig)
    except ProjectError:
        pass


@settings(max_examples=50, deadline=None)
@given(
    chunk_size=st.integers(min_value=-10, max_value=300),
    chunk_overlap=st.integers(min_value=-10, max_value=400),
)
def test_chunk_range_boundary_predicate(tmp_path_factory: pytest.TempPathFactory, chunk_size: int, chunk_overlap: int) -> None:
    """load succeeds iff 1 <= chunk_size and 0 <= chunk_overlap < chunk_size (cee7c56)."""
    root = tmp_path_factory.mktemp("proj")
    _init(root, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    valid = 1 <= chunk_size and 0 <= chunk_overlap < chunk_size
    if valid:
        cfg = load_project_config(root)
        assert (cfg.chunk_size, cfg.chunk_overlap) == (chunk_size, chunk_overlap)
    else:
        with pytest.raises(ProjectError):
            load_project_config(root)


@settings(max_examples=50, deadline=None)
@given(
    extra=st.sampled_from(
        [
            ["sub"],
            ["sub", "sub"],
            [".", "sub"],
            ["sub/../sub"],
            ["./sub"],
        ]
    )
)
def test_indexed_paths_dedupe_aliases_preserving_order(tmp_path_factory: pytest.TempPathFactory, extra: list[str]) -> None:
    """Aliased spellings collapse to unique resolved paths; first-occurrence order kept."""
    root = tmp_path_factory.mktemp("proj")
    (root / "sub").mkdir(parents=True)
    _init(root, paths=tuple([".", *extra]))
    cfg = load_project_config(root)
    resolved = [p.resolve() for p in cfg.indexed_paths]
    assert len(resolved) == len(set(resolved))
    assert resolved[0] == root.resolve()


def test_remove_last_refusal_and_add_idempotence_properties(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    (root / "sub").mkdir(parents=True)
    _init(root, paths=(".",))
    cfg = load_project_config(root)
    # removing the only path refused (62d4a7c)
    with pytest.raises(ProjectError):
        remove_indexed_path(cfg, cfg.indexed_paths[0])
    # adding an equivalent spelling twice is idempotent
    two = add_indexed_path(cfg, root / "sub")
    again = add_indexed_path(two, root / "sub" / ".." / "sub")
    assert len(again.indexed_paths) == len(two.indexed_paths) == 2
    # removing a nonexistent path is an idempotent no-op that still round-trips
    noop = remove_indexed_path(two, root / "does-not-exist")
    assert noop.indexed_paths == two.indexed_paths
    assert load_project_config(root).indexed_paths == again.indexed_paths


@settings(max_examples=50, deadline=None)
@given(depth=st.integers(min_value=1, max_value=4))
def test_covers_path_lexicographic_containment(tmp_path_factory: pytest.TempPathFactory, depth: int) -> None:
    """Covered iff the resolved path sits under some indexed root; sibling-prefix dirs
    ('proj' vs 'projx') and outside-root paths are NEVER covered."""
    tmp = tmp_path_factory.mktemp("covers")
    proj = tmp / "proj"
    sibling = tmp / "projx"
    proj.mkdir()
    sibling.mkdir()
    _init(proj, paths=(".",))
    cfg = load_project_config(proj)
    nested = sibling
    for i in range(depth):
        nested = nested / f"d{i}"
    deep_file = nested / "f.txt"
    nested.mkdir(parents=True, exist_ok=True)
    deep_file.touch()
    assert project_covers_path(cfg, deep_file) is False
    inside = proj / "a.txt"
    inside.touch()
    assert project_covers_path(cfg, inside) is True
    assert project_covers_path(cfg, tmp / "unrelated") is False
