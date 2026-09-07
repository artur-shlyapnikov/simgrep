"""Contract pins for removed API surface: the custom IndexError shadowing the builtin
(removed in 4c921de) and the dead SearchOutcome.notices field (removed in 2683f98)."""

from __future__ import annotations

import dataclasses

from simgrep.errors import ConfigError, MetadataError, ProjectError, SearchError, SimgrepError
from simgrep.models import SearchOutcome


def test_error_taxonomy_pins_builtin_indexerror_restored() -> None:
    assert issubclass(ConfigError, SimgrepError)
    assert issubclass(ProjectError, SimgrepError)
    assert issubclass(MetadataError, SimgrepError)
    assert issubclass(SearchError, SimgrepError)
    import simgrep.errors as errors_module

    assert not hasattr(errors_module, "IndexError")
    assert IndexError.__module__ == "builtins"
    assert not issubclass(IndexError, SimgrepError)


def test_search_outcome_field_set_pins_notices_removal() -> None:
    assert tuple(f.name for f in dataclasses.fields(SearchOutcome)) == (
        "results",
        "base_path",
        "files_seen",
        "chunks_searched",
        "semantic_candidates",
    )
