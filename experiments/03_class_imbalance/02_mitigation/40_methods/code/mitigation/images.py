"""Decode patch JPGs once into a uint8 tensor; normalize on-device per batch.

Caching decoded patches as uint8 instead of normalized float32 keeps one arm's
training images in a fraction of the memory (e.g. TCGA-UT's T=19,200 patches
at 224x224x3 uint8 is ~2.9 GB), so a training draw's images can live resident
on the GPU for the whole stage instead of being re-decoded every step.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as tv_transforms
from imbalance_benchmark.datasets.features import _ImagePathDataset

__all__ = ["decode_transform", "decode_images", "normalize_batch"]


def decode_transform(data_config: dict[str, Any]) -> tv_transforms.Compose:
    """Resize/crop-only transform matching Virchow2's eval input size, no normalization."""
    size = data_config["input_size"][-1]
    crop_pct = data_config.get("crop_pct") or 1.0
    resize = round(size / crop_pct)
    return tv_transforms.Compose(
        [
            tv_transforms.Resize(resize),
            tv_transforms.CenterCrop(size),
            tv_transforms.PILToTensor(),
        ]
    )


def _decode_workers() -> int:
    """Parallel image-decode workers, sized to the job's allocated CPUs."""
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    total = int(slurm_cpus) if slurm_cpus else os.cpu_count() or 1
    return max(0, min(total - 1, 7))


def decode_images(
    image_paths: list[str], transform: tv_transforms.Compose, batch_size: int = 256
) -> torch.Tensor:
    """Decode a list of patch images once into one stacked uint8 (N, 3, H, W) tensor."""
    if not image_paths:
        return torch.empty((0, 3, 0, 0), dtype=torch.uint8)
    loader = DataLoader(
        _ImagePathDataset(image_paths, transform),
        batch_size=batch_size,
        num_workers=_decode_workers(),
    )
    return torch.cat([batch for batch in loader], dim=0)


def normalize_batch(
    uint8_batch: torch.Tensor, mean: tuple[float, ...], std: tuple[float, ...]
) -> torch.Tensor:
    """Cast a uint8 image batch to float and apply per-channel mean/std normalization."""
    device = uint8_batch.device
    mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, -1, 1, 1)
    return uint8_batch.float().div_(255.0).sub_(mean_t).div_(std_t)
