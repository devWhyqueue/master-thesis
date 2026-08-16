from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, cast

import timm
import torch
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers.mlp import SwiGLUPacked
from torch.utils.data import DataLoader, Dataset

from imbalance_benchmark.datasets.features.attach import attach_extracted_features
from imbalance_benchmark.datasets.features.cache import (
    load_feature_row,
    load_slide_features,
)
from imbalance_benchmark.datasets.feature_provenance import (
    FEATURE_DIM,
    VIRCHOW2_MODEL,
    VIRCHOW2_REVISION,
    VIRCHOW2_WEIGHTS_SHA256,
    patch_sort_key,
    resolve_feature_provenance,
    resolve_feature_snapshot,
)

__all__ = [
    "load_feature_model",
    "embed_image_batch",
    "extract_slide_features",
    "attach_extracted_features",
    "load_slide_features",
    "load_feature_row",
    "patch_sort_key",
    "resolve_feature_provenance",
]


def load_feature_model(
    model_name: str,
    device: torch.device,
    revision: str = VIRCHOW2_REVISION,
    weights_sha256: str = VIRCHOW2_WEIGHTS_SHA256,
) -> tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor]]:
    """Load the frozen Virchow2 feature encoder and its input transform."""
    snapshot = resolve_feature_snapshot(
        {
            "model_name": model_name,
            "revision": revision,
            "weights_sha256": weights_sha256,
        }
    )
    model = timm.create_model(
        f"local-dir:{snapshot.as_posix()}",
        pretrained=False,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    model.load_state_dict(load_safetensors(snapshot / "model.safetensors"))
    model = model.eval().to(device)
    transforms = cast(
        Callable[[Image.Image], torch.Tensor],
        create_transform(**resolve_data_config(model.pretrained_cfg, model=model)),
    )
    return model, transforms


def _virchow2_pool(output: torch.Tensor) -> torch.Tensor:
    """Concatenate the CLS token and mean patch token into a 2560-d feature."""
    class_token = output[:, 0]
    patch_tokens = output[:, 5:]
    return torch.cat([class_token, patch_tokens.mean(1)], dim=-1).float().cpu()


def embed_image_batch(
    model: torch.nn.Module, images: torch.Tensor, dtype: str, device: torch.device
) -> list[torch.Tensor]:
    """Embed a batch of images into per-image Virchow2 (CLS + mean-patch) features."""
    amp_dtype = torch.float16 if dtype == "float16" else torch.float32
    with (
        torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=device.type != "cpu"
        ),
        torch.inference_mode(),
    ):
        output = model(images)
    features = _virchow2_pool(output)
    return [features[index] for index in range(len(features))]


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
    """Parallel image-decode workers, sized to the job's allocated CPUs.

    Image I/O/decode otherwise runs single-threaded ahead of the GPU forward
    pass, which starves an 8-CPU allocation down to one core's throughput.
    """
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    total = int(slurm_cpus) if slurm_cpus else os.cpu_count() or 1
    return max(0, min(total - 1, 7))


def _resolve_model(
    model_cache: dict[str, Any] | None,
    model_name: str,
    device: torch.device,
    revision: str,
    weights_sha256: str,
) -> tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor]]:
    """Load once and reuse via ``model_cache``; load fresh when uncached."""
    if model_cache is None:
        return load_feature_model(model_name, device, revision, weights_sha256)
    if "model" not in model_cache:
        model_cache["model"], model_cache["transforms"] = load_feature_model(
            model_name, device, revision, weights_sha256
        )
    return model_cache["model"], model_cache["transforms"]


def _resolve_extraction_options(
    options: dict[str, Any],
) -> tuple[str, int, str, str, str, torch.device]:
    resolved_device = options.get("device") or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    return (
        str(options.get("model_name", VIRCHOW2_MODEL)),
        int(options.get("batch_size", 64)),
        str(options.get("dtype", "float16")),
        str(options.get("revision", VIRCHOW2_REVISION)),
        str(options.get("weights_sha256", VIRCHOW2_WEIGHTS_SHA256)),
        resolved_device,
    )


def extract_slide_features(
    image_paths: list[str],
    options: dict[str, Any],
    model_cache: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Embed a slide's ordered patch images into a stacked (n_patches, 2560) tensor.

    ``options`` carries the same model-config bag ``attach_extracted_features``
    assembles (``model_name``, ``batch_size``, ``dtype``, ``device``,
    ``revision``, ``weights_sha256``).
    """
    model_name, batch_size, dtype, revision, weights_sha256, resolved_device = (
        _resolve_extraction_options(options)
    )
    if not image_paths:
        return torch.empty((0, FEATURE_DIM))
    model, transforms = _resolve_model(
        model_cache, model_name, resolved_device, revision, weights_sha256
    )
    loader = DataLoader(
        _ImagePathDataset(image_paths, transforms),
        batch_size=batch_size,
        num_workers=int(options.get("loader_workers", _loader_worker_count())),
    )
    rows: list[torch.Tensor] = []
    for images in loader:
        rows.extend(
            embed_image_batch(model, images.to(resolved_device), dtype, resolved_device)
        )
    return torch.stack(rows)
