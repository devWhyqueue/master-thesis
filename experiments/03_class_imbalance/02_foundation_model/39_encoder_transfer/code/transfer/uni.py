"""Frozen UNI2-h loading, transform, and per-patch embedding (exp-39, phase 03).

Experiment-local and opt-in: the shared benchmark's Virchow2-only feature
pipeline (``imbalance_benchmark.datasets.features``) is never imported or
modified here. Every constant below
is read from the frozen ``configs/encoder_lock.json`` rather than duplicated.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import timm
import torch
from PIL import Image
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers.mlp import SwiGLUPacked
from torch.utils.data import DataLoader, Dataset

from imbalance_benchmark.common import compute_sha256, find_repo_root, write_json

__all__ = [
    "ENCODER_LOCK",
    "FEATURE_DIM",
    "load_uni_model",
    "extract_slide_features",
    "write_or_verify_provenance",
]

_LOCK_PATH = (
    find_repo_root()
    / "experiments/03_class_imbalance/02_foundation_model/39_encoder_transfer"
    / "configs/encoder_lock.json"
)
ENCODER_LOCK: dict[str, Any] = json.loads(_LOCK_PATH.read_text(encoding="utf-8"))
FEATURE_DIM: int = int(ENCODER_LOCK["architecture"]["output_dim"])


def _checkpoint_snapshot() -> Path:
    return find_repo_root() / ENCODER_LOCK["encoder"]["local_pin"]


def load_uni_model(
    device: torch.device,
) -> tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor]]:
    """Load the pinned UNI2-h checkpoint and its resolved evaluation transform."""
    snapshot = _checkpoint_snapshot()
    checkpoint = snapshot / ENCODER_LOCK["encoder"]["checkpoint_file"]
    if compute_sha256(checkpoint) != ENCODER_LOCK["encoder"]["checkpoint_sha256"]:
        raise ValueError("Pinned UNI2-h checkpoint no longer matches its locked hash")
    kwargs = dict(ENCODER_LOCK["architecture"]["timm_kwargs"])
    kwargs["mlp_layer"] = SwiGLUPacked
    kwargs["act_layer"] = torch.nn.SiLU
    commit = ENCODER_LOCK["encoder"]["commit_sha"]
    model = timm.create_model(
        f"hf-hub:{ENCODER_LOCK['encoder']['hf_repo']}@{commit}",
        pretrained=True,
        cache_dir=str(snapshot.parents[2]),
        **kwargs,
    )
    model = model.eval().to(device)
    transforms = cast(
        Callable[[Image.Image], torch.Tensor],
        create_transform(**resolve_data_config(model.pretrained_cfg, model=model)),
    )
    return model, transforms


class _ImagePathDataset(Dataset):  # type: ignore[type-arg]
    def __init__(
        self, paths: list[str], transforms: Callable[[Image.Image], torch.Tensor]
    ) -> None:
        self._paths = paths
        self._transforms = transforms

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._transforms(Image.open(self._paths[index]).convert("RGB"))


def _loader_worker_count() -> int:
    """Parallel image-decode workers, sized to the job's allocated CPUs."""
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    total = int(slurm_cpus) if slurm_cpus else os.cpu_count() or 1
    return max(0, min(total - 1, 7))


def extract_slide_features(
    image_paths: list[str],
    dtype: str,
    device: torch.device,
    model_cache: dict[str, Any],
    batch_size: int = 8,
) -> torch.Tensor:
    """Embed a slide's ordered patch images into a stacked (n_patches, 1536) tensor.

    UNI2-h pools to a single token per image (``global_pool='token'``); unlike
    Virchow2 there is no CLS/mean-patch concatenation to perform here.
    """
    if not image_paths:
        return torch.empty((0, FEATURE_DIM))
    if "model" not in model_cache:
        model_cache["model"], model_cache["transforms"] = load_uni_model(device)
    model, transforms = model_cache["model"], model_cache["transforms"]
    storage_dtype = torch.float16 if dtype == "float16" else torch.float32
    loader = DataLoader(
        _ImagePathDataset(image_paths, transforms),
        batch_size=batch_size,
        num_workers=_loader_worker_count(),
    )
    rows: list[torch.Tensor] = []
    with torch.inference_mode():
        for images in loader:
            output = model(images.to(device)).float().cpu().to(storage_dtype)
            rows.extend(output[index] for index in range(len(output)))
    return torch.stack(rows)


def _provenance(dtype: str) -> dict[str, Any]:
    return {
        "encoder": "UNI2-h",
        "encoder_lock_sha256": compute_sha256(_LOCK_PATH),
        "pooling": "token",
        "feature_dim": FEATURE_DIM,
        "dtype": dtype,
    }


def write_or_verify_provenance(feature_root: Path, dtype: str) -> None:
    """Write or verify the immutable UNI2-h cache-provenance metadata.

    Mirrors ``imbalance_benchmark.datasets.feature_provenance.validate_feature_cache``
    but stays experiment-local: it never touches the shared Virchow2-only cache guard.
    """
    metadata_path = feature_root / "feature_provenance.json"
    provenance = _provenance(dtype)
    if metadata_path.exists():
        recorded = json.loads(metadata_path.read_text(encoding="utf-8"))
        if recorded != provenance:
            raise ValueError(
                "Cached UNI2-h feature provenance differs from the requested run"
            )
        return
    if any(feature_root.glob("*.pt")):
        raise ValueError("Cached UNI2-h features lack feature provenance metadata")
    temporary = metadata_path.with_suffix(
        metadata_path.suffix + f".{uuid4().hex}.partial"
    )
    write_json(temporary, provenance)
    os.replace(temporary, metadata_path)
