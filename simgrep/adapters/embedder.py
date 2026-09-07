from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np

from simgrep.errors import SimgrepError

_PROMPT_CONFIG = "config_sentence_transformers.json"


def _default_device(torch_mod: Any) -> str:
    """Mirror SentenceTransformer's auto selection: cuda, then mps, then cpu."""
    if torch_mod.cuda.is_available():
        return "cuda"
    mps = getattr(torch_mod.backends, "mps", None)
    if mps is not None and torch_mod.backends.mps.is_available():
        return "mps"
    return "cpu"


def _import_transformers_light() -> tuple[Any, Any]:
    """Import transformers without its eager sklearn/scipy chain.

    ``transformers.generation.candidate_generator`` imports ``sklearn.metrics``
    at module scope whenever sklearn is installed (~1s of imports per CLI
    invocation). Embeddings never generate, so stub the two modules for the
    duration of the import and evict them afterwards; the stubbed attribute
    stays bound inside transformers but is never called by simgrep.
    """
    import importlib.machinery
    import sys
    import types

    stub_metrics = types.ModuleType("sklearn.metrics")
    stub_metrics.roc_curve = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    stub_sklearn = types.ModuleType("sklearn")
    stub_sklearn.metrics = stub_metrics  # type: ignore[attr-defined]
    # find_spec() reads module.__spec__; without a spec, transformers'
    # availability probes raise ValueError instead of reporting unavailable.
    stub_sklearn.__spec__ = importlib.machinery.ModuleSpec("sklearn", None)  # type: ignore[attr-defined]
    stub_metrics.__spec__ = importlib.machinery.ModuleSpec("sklearn.metrics", None)  # type: ignore[attr-defined]
    stubs = {"sklearn": stub_sklearn, "sklearn.metrics": stub_metrics}
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        from transformers import AutoModel, AutoTokenizer

        return AutoModel, AutoTokenizer
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class SentenceEmbedder:
    """Embedding adapter on raw ``transformers``.

    Replicates the sentence-transformers pipeline (Transformer -> Pooling ->
    Normalize) without importing the ``sentence_transformers`` package, whose
    import chain alone costs about a second per CLI invocation. Pooling mode is
    read from the model's ``modules.json`` (cls and mean are supported; other
    modes fail loudly instead of producing silently wrong vectors).
    """

    def __init__(self, model_name: str, normalize_embeddings: bool = True):
        self._model_name = model_name
        self._normalize_embeddings = normalize_embeddings
        self._query_prompt: str | None = None
        try:
            import torch

            AutoModel, AutoTokenizer = _import_transformers_light()

            # Raw per-tensor tqdm (transformers >=5 core_model_loading) is noise in a
            # CLI whose progress surfaces are the indexing sink and HF download bars.
            from transformers.utils import logging as _hf_logging

            _hf_logging.disable_progress_bar()  # type: ignore[no-untyped-call]

            self._torch = torch
            self._device = torch.device(os.environ.get("SIMGREP_DEVICE") or _default_device(torch))
            self._tokenizer = self._load(
                lambda: AutoTokenizer.from_pretrained(model_name, local_files_only=True),  # type: ignore[no-untyped-call]
                lambda: AutoTokenizer.from_pretrained(model_name),  # type: ignore[no-untyped-call]
            )
            self._model = self._load(
                lambda: AutoModel.from_pretrained(model_name, local_files_only=True),  # type: ignore[no-untyped-call]
                lambda: AutoModel.from_pretrained(model_name),  # type: ignore[no-untyped-call]
            )
            self._model.to(self._device)
            self._model.eval()
            self._use_fp16 = self._device.type == "mps" and os.environ.get("SIMGREP_ENCODE_DTYPE", "fp16").lower() != "fp32"
            if self._use_fp16:
                # fp16 halves weight bandwidth; measured +18% encode throughput on
                # M1 Pro with max cosine delta 5e-7 against fp32 (ranking-identical).
                self._model.half()
            hidden = int(getattr(self._model.config, "hidden_size", 0) or 0)
            if hidden <= 0:
                raise SimgrepError(f"Could not determine embedding dimension for model '{model_name}'.")
            self._ndim = hidden
            self._pooling = self._resolve_pooling()
        except SimgrepError:
            raise
        except Exception as exc:
            raise SimgrepError(f"Failed to load embedding model '{model_name}'.") from exc

    @staticmethod
    def _load(primary: Callable[[], Any], fallback: Callable[[], Any]) -> Any:
        # Offline-first: a cached model must not cost a network round-trip.
        try:
            return primary()
        except Exception:
            return fallback()

    def _cached_repo_file(self, filename: str) -> Any:
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(self._model_name, filename, local_files_only=True)
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            return payload
        except Exception:
            return None

    def _resolve_pooling(self) -> str:
        modules = self._cached_repo_file("modules.json")
        if modules is None:
            # SentenceTransformer's default for plain transformer models is mean pooling.
            return "mean"
        pooling_entry = next((m for m in modules if str(m.get("type", "")).endswith("models.Pooling")), None)
        if pooling_entry is None:
            return "cls"
        pool_path = str(pooling_entry.get("path") or "").strip("/")
        config = self._cached_repo_file(f"{pool_path}/config.json") if pool_path else None
        if config is None:
            return "cls"
        if config.get("pooling_mode_cls_token"):
            return "cls"
        if config.get("pooling_mode_mean_tokens"):
            return "mean"
        raise SimgrepError(
            f"Unsupported pooling mode for model '{self._model_name}': {config}.",
            hint="simgrep supports pooling_mode_cls_token and pooling_mode_mean_tokens models.",
        )

    def _query_prefix(self) -> str:
        if self._query_prompt is None:
            prompt = ""
            if "qwen" in self._model_name.lower():
                prompts = (self._cached_repo_file(_PROMPT_CONFIG) or {}).get("prompts", {})
                prompt = str(prompts.get("query", ""))
            self._query_prompt = prompt
        return self._query_prompt

    @property
    def ndim(self) -> int:
        return self._ndim

    def encode(self, texts: list[str], *, is_query: bool = False, batch_size: int | None = None) -> np.ndarray:
        try:
            torch = self._torch
            if not texts:
                return np.zeros((0, self._ndim), dtype=np.float32)
            if is_query:
                prefix = self._query_prefix()
                if prefix:
                    texts = [prefix + text for text in texts]
            size = batch_size if batch_size is not None and batch_size > 0 else 32
            # Length-sorted batches minimize padding, like SentenceTransformer.encode.
            order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
            vectors = torch.empty((len(texts), self._ndim), dtype=torch.float32)
            with torch.inference_mode():
                for start in range(0, len(order), size):
                    batch_idx = order[start : start + size]
                    features = self._tokenizer(
                        [texts[i] for i in batch_idx],
                        padding=True,
                        truncation="longest_first",
                        return_tensors="pt",
                    )
                    features = {key: value.to(self._device) for key, value in features.items()}
                    hidden_states = self._model(**features).last_hidden_state
                    if self._pooling == "cls":
                        pooled = hidden_states[:, 0].float()
                    else:
                        # Pool in fp16 via broadcast (no expanded fp32 mask copy);
                        # sum(dtype=float32) accumulates in fp32: measured +12%
                        # encode throughput, max cosine delta 0.0 vs fp32 pooling.
                        mask = features["attention_mask"].unsqueeze(-1)
                        pooled = (hidden_states * mask).sum(dim=1, dtype=torch.float32) / torch.clamp(mask.sum(dim=1, dtype=torch.float32), min=1e-9)
                    if self._normalize_embeddings:
                        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                    vectors[batch_idx] = pooled.float().to("cpu")
            return cast(np.ndarray, vectors.numpy())
        except SimgrepError:
            raise
        except Exception as exc:
            raise SimgrepError(f"Failed to generate embeddings using model '{self._model_name}'.") from exc

    def export_onnx(self, destination: Path) -> None:
        """Export the loaded model to ONNX for the light query-path runtime.

        Always exports fp32 weights (the fp16 fast path is MPS-only; ONNX
        Runtime CPU executes fp16 graphs far slower). Dynamic batch and
        sequence axes so every query shape is served by one graph.
        """
        try:
            torch = self._torch
            sample = self._tokenizer(["simgrep onnx export sample"], padding=True, truncation="longest_first", return_tensors="pt")
            features = {key: value.to(self._device) for key, value in sample.items()}
            destination.mkdir(parents=True, exist_ok=True)
            tmp_path = destination / "model.onnx.tmp"
            was_fp16 = getattr(self, "_use_fp16", False)
            if was_fp16:
                self._model.float()
            try:
                with torch.inference_mode():
                    torch.onnx.export(
                        self._model,
                        (features["input_ids"], features["attention_mask"]),
                        str(tmp_path),
                        input_names=["input_ids", "attention_mask"],
                        output_names=["last_hidden_state"],
                        dynamic_axes={
                            "input_ids": {0: "batch", 1: "seq"},
                            "attention_mask": {0: "batch", 1: "seq"},
                            "last_hidden_state": {0: "batch", 1: "seq"},
                        },
                        opset_version=17,
                        dynamo=False,
                    )
            finally:
                if was_fp16:
                    self._model.half()
            tmp_path.replace(destination / "model.onnx")
            meta = {"model": self._model_name, "ndim": self._ndim, "pooling": self._pooling}
            (destination / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        except SimgrepError:
            raise
        except Exception as exc:
            raise SimgrepError(f"Failed to export ONNX model '{self._model_name}'.") from exc
