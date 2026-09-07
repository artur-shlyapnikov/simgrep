"""Lazy bulk embedder: search flows must never pay the torch import cost."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from simgrep.adapters import vector as vector_mod
from simgrep.models import AppConfig
from simgrep.runtime import LazyBulkEmbedder, RuntimeFactory

PY = sys.executable


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([PY, "-c", code], capture_output=True, text=True, cwd=Path(__file__).parents[3], check=True)


def test_lazy_ndim_does_not_import_torch() -> None:
    result = _run(
        "import sys; sys.path.insert(0, '.');\n"
        "import simgrep.runtime as rt\n"
        "rt._ndim_without_torch = lambda model: 384\n"
        "e = rt.LazyBulkEmbedder('some/model')\n"
        "assert e.ndim == 384\n"
        "assert 'torch' not in sys.modules, 'torch imported by lazy ndim'\n"
    )
    assert result.returncode == 0, result.stderr


def test_usearch_guard_imports_torch_only_when_pending() -> None:
    pending = _run(
        "import sys; sys.path.insert(0, '.');\n"
        "from simgrep.adapters import vector\n"
        "vector.mark_torch_pending()\n"
        "vector._usearch()\n"
        "assert 'torch' in sys.modules, 'guard ignored pending flag'\n"
    )
    assert pending.returncode == 0, pending.stderr

    clean = _run(
        "import sys; sys.path.insert(0, '.');\n"
        "from simgrep.adapters import vector\n"
        "vector._usearch()\n"
        "assert 'torch' not in sys.modules, 'guard imported torch without pending'\n"
    )
    assert clean.returncode == 0, clean.stderr


def test_encode_materializes_inner_embedder_once(monkeypatch: Any) -> None:
    constructed: list[str] = []

    class FakeInner:
        def __init__(self, model_name: str, normalize_embeddings: bool) -> None:
            constructed.append(model_name)
            self.ndim = 7

        def encode(self, texts: list[str], *, is_query: bool = False, batch_size: int | None = None) -> str:
            return "vectors"

    monkeypatch.setattr("simgrep.runtime.SentenceEmbedder", FakeInner)
    lazy = LazyBulkEmbedder("m")
    assert lazy.encode(["x"]) == "vectors"
    assert lazy.encode(["y"]) == "vectors"
    assert constructed == ["m"], "inner embedder constructed more than once"
    assert lazy.ndim == 7


def test_require_bulk_sets_and_clears_pending_flag(monkeypatch: Any) -> None:
    class FakeInner:
        def __init__(self, model_name: str, normalize_embeddings: bool) -> None:
            self.ndim = 3

    monkeypatch.setattr("simgrep.runtime.SentenceEmbedder", FakeInner)
    monkeypatch.setattr(vector_mod, "_TORCH_PENDING", False)
    lazy = LazyBulkEmbedder("m")
    monkeypatch.setattr(lazy, "_mark_pending", lambda: None)
    lazy.require()
    assert vector_mod._TORCH_PENDING is False


def test_factory_search_flow_never_constructs_torch_embedder(monkeypatch: Any) -> None:
    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("SentenceEmbedder constructed in search flow")

    class FakeOnnx:
        ndim = 4

    monkeypatch.setattr("simgrep.runtime.SentenceEmbedder", explode)
    monkeypatch.setattr("simgrep.runtime.has_onnx_cache", lambda model: True)
    monkeypatch.setattr("simgrep.runtime.OnnxEmbedder", lambda model: FakeOnnx())
    monkeypatch.setattr("simgrep.runtime._ndim_without_torch", lambda model: 4)

    factory = RuntimeFactory()
    runtime = factory.for_app(AppConfig(model="fake/model"))
    from simgrep import runtime as rt

    assert isinstance(runtime.query_embedder, rt._LazyQueryEmbedder)
    assert isinstance(runtime.embedder, LazyBulkEmbedder)
    assert runtime.embedder.ndim == 4
    assert runtime.query_embedder.ndim == 4, "query session builds on first use"
    assert isinstance(runtime.query_embedder._inner, FakeOnnx)


def test_lazy_ndim_prefers_onnx_meta_over_hub(monkeypatch: Any) -> None:
    """Query-only flows read ndim from the local ONNX meta; the hub config
    path must not be consulted when the meta file exists."""
    from simgrep import runtime as rt

    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("hub fallback must not run when ONNX meta exists")

    monkeypatch.setattr(rt, "_ndim_without_torch", explode)
    monkeypatch.setattr(rt, "onnx_cache_dir", lambda model, root=None: _write_fake_meta({"ndim": 384}))
    assert rt.LazyBulkEmbedder("some/model").ndim == 384


def test_lazy_ndim_falls_through_without_meta(monkeypatch: Any) -> None:
    from simgrep import runtime as rt

    monkeypatch.setattr(rt, "_ndim_without_torch", lambda model: 512)
    monkeypatch.setattr(rt, "onnx_cache_dir", lambda model, root=None: Path("/nonexistent"))
    assert rt.LazyBulkEmbedder("some/model").ndim == 512


def test_lazy_ndim_never_imports_hub_subprocess() -> None:
    """Module-absence is process-global; assert it in a pristine subprocess."""
    result = _run(
        "import sys; sys.path.insert(0, '.');\n"
        "import json, tempfile\n"
        "from pathlib import Path\n"
        "import simgrep.runtime as rt\n"
        "cache = Path(tempfile.mkdtemp())\n"
        "(cache / 'meta.json').write_text(json.dumps({'ndim': 384}))\n"
        "rt.onnx_cache_dir = lambda model, root=None: cache\n"
        "e = rt.LazyBulkEmbedder('some/model')\n"
        "assert e.ndim == 384\n"
        "assert 'huggingface_hub' not in sys.modules, 'hub imported for meta ndim'\n"
        "assert 'torch' not in sys.modules, 'torch imported for meta ndim'\n"
    )
    assert result.returncode == 0, result.stderr


def _write_fake_meta(payload: dict[str, int]) -> Path:
    import tempfile

    cache = Path(tempfile.mkdtemp(prefix="simgrep-onnx-meta-"))
    (cache / "meta.json").write_text(json.dumps(payload), encoding="utf-8")
    return cache


def test_ndim_without_torch_returns_none_for_unknown_model() -> None:
    from simgrep.runtime import _ndim_without_torch

    assert _ndim_without_torch("definitely/not-a-real-model-xyz") is None
