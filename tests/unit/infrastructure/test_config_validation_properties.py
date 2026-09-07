"""Property-based tests for app-config validation (Round 12).

Invariants pinned here:
- corrupting exactly one config key with an arbitrary scalar either loads a valid
  AppConfig or raises the clean error type (ConfigError) -- never a raw
  TypeError/KeyError/AttributeError;
- float/bool truncation semantics for int fields;
- lexical_weight boundary predicate (nan always rejected);
- schema-version mismatch falls back to defaults silently (app side);
- save -> load round-trip identity for randomized valid configs;
- known product bug: raw OverflowError escapes the clean-error invariant
  (pinned as strict xfail until config.py widens its except clause).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from simgrep.config import _coerce_config, load_app_config, save_app_config
from simgrep.errors import ConfigError
from simgrep.models import AppConfig

CLEAN_ERRORS = (ConfigError,)

# Malformed-scalar pool targeting every _coerce/_validate branch.
MALFORMED_SCALARS = st.one_of(
    # NOTE: infinities/huge floats EXCLUDED here -- see product-bug xfail test below
    # (_coerce_field wraps only TypeError/ValueError; int(inf) -> raw OverflowError).
    st.one_of(
        st.floats(allow_nan=False, min_value=-1e30, max_value=1e30),
        st.just(float("nan")),
    ),
    st.text(max_size=8),
    st.integers(),
    st.lists(st.integers(), max_size=3),
)
CONFIG_KEYS = [
    "schema_version",
    "model",
    "chunk_size",
    "chunk_overlap",
    "batch_size",
    "max_file_size_bytes",
    "follow_symlinks",
    "file_patterns",
    "lexical_top",
    "lexical_weight",
    "freshness",
    "context_lines",
    "max_chars",
]


@settings(max_examples=50, deadline=None)
@given(key=st.sampled_from(CONFIG_KEYS), raw=MALFORMED_SCALARS)
def test_load_never_crashes_outside_clean_error_types(key: str, raw: object) -> None:
    """Invariant: corrupting exactly one key with an arbitrary scalar either loads a valid
    AppConfig or raises ConfigError -- never TypeError/KeyError/AttributeError/ValueError."""
    try:
        cfg = _coerce_config({key: raw})
        assert isinstance(cfg, AppConfig)
    except CLEAN_ERRORS:
        pass


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (12.9, 12),  # float truncation is silent
        (True, 1),  # bool-as-int
        (False, 0),
        ("12", 12),  # digit strings coerce
        (0.999999, 0),  # truncates to invalid -> rejected below
    ],
)
def test_int_coercion_truncates_floats_and_bools(tmp_path: Path, raw: object, expected: int) -> None:
    if expected < 1:
        with pytest.raises(ConfigError):
            _coerce_config({"chunk_size": raw})
        return
    # default chunk_overlap=20 would collide with small sizes; pin it valid
    cfg = _coerce_config({"chunk_size": raw, "chunk_overlap": 0})
    assert cfg.chunk_size == expected


@settings(max_examples=50, deadline=None)
@given(weight=st.one_of(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False), st.just(float("nan"))))
def test_lexical_weight_boundary_property(weight: float) -> None:
    """Valid iff 0.0 <= weight <= 1.0 after float(); nan always rejected."""
    try:
        cfg = _coerce_config({"lexical_weight": weight})
        assert 0.0 <= cfg.lexical_weight <= 1.0
    except ConfigError:
        assert not (0.0 <= weight <= 1.0)


@pytest.mark.parametrize(
    ("raw", "expect_defaults"),
    [("1", False), (1.0, False), (99, True), (True, False)],
)
def test_schema_version_semantics(raw: object, expect_defaults: bool) -> None:
    """App config: coercible current-version values load; mismatched versions fall back to
    defaults silently (contrast: project.toml raises -- covered in file B)."""
    cfg = _coerce_config({"schema_version": raw, "model": "custom-model"})
    assert cfg.model == ("custom-model" if not expect_defaults else AppConfig().model)


@settings(max_examples=50, deadline=None)
@given(
    chunk_size=st.integers(min_value=1, max_value=4096),
    chunk_overlap=st.integers(min_value=0, max_value=4095),
    batch_size=st.integers(min_value=1, max_value=512),
    lexical_top=st.integers(min_value=0, max_value=2000),
    max_chars=st.integers(min_value=1, max_value=10**6),
    weight=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_save_load_roundtrip_identity(
    tmp_path_factory: pytest.TempPathFactory,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    lexical_top: int,
    max_chars: int,
    weight: float,
) -> None:
    """save -> load preserves every field for boundary-valid randomized configs."""
    overlap = min(chunk_overlap, chunk_size - 1)
    path = tmp_path_factory.mktemp("cfg") / "config.toml"
    cfg = AppConfig(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        batch_size=batch_size,
        lexical_top=lexical_top,
        max_chars=max_chars,
        lexical_weight=weight,
    )
    save_app_config(cfg, config_file=path)
    assert load_app_config(config_file=path) == cfg


@pytest.mark.xfail(
    strict=True,
    reason="PRODUCT BUG: _coerce_field wraps only (TypeError, ValueError); "
    "int(float('inf')) raises raw OverflowError instead of ConfigError. "
    "Repro: _coerce_config({'chunk_size': float('inf')}) or load_app_config on a toml "
    "with chunk_size = inf. Flip this xfail when config.py widens the except clause.",
)
def test_int_overflow_wrapped_as_config_error() -> None:
    with pytest.raises(ConfigError):
        _coerce_config({"chunk_size": float("inf")})


@pytest.mark.parametrize("patterns", [[], ["", "   "], (), "not-a-list", 5])
def test_file_patterns_rejects_empty_and_wrong_typed(patterns: object) -> None:
    """Empty, whitespace-only, and wrong-typed pattern values are all rejected with
    ConfigError (strict contract since fe7d2bd; previously non-list values silently
    fell back to defaults)."""
    with pytest.raises(ConfigError):
        _coerce_config({"file_patterns": patterns})
