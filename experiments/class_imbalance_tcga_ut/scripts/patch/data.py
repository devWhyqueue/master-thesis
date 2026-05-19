from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, RandomSampler, WeightedRandomSampler


class PatchImageDataset(Dataset):
    """Image dataset for controlled TCGA-UT patch manifests."""

    def __init__(
        self,
        frame: pd.DataFrame,
        class_to_idx: dict[str, int],
        image_size: int,
    ) -> None:
        self.rows = frame.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.rows.iloc[idx]
        image = Image.open(Path(str(row["image_path"]))).convert("RGB")
        image = image.resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return tensor, self.class_to_idx[str(row["cancer_type"])]


def patch_loader(
    dataset: PatchImageDataset,
    labels: np.ndarray,
    method: str,
    batch_size: int,
    seed: int,
    num_workers: int,
    samples_per_epoch: int | None = None,
) -> DataLoader:
    """Build deterministic patch training loaders."""
    generator = torch.Generator().manual_seed(seed)
    if samples_per_epoch is not None:
        sampler = RandomSampler(
            dataset,
            replacement=False,
            num_samples=min(samples_per_epoch, len(dataset)),
            generator=generator,
        )
        return _data_loader(dataset, batch_size, num_workers, sampler=sampler)
    if method != "patch_balanced_sampler_ce":
        return _data_loader(
            dataset, batch_size, num_workers, shuffle=True, generator=generator
        )
    sampler = _balanced_sampler(labels, generator)
    return _data_loader(dataset, batch_size, num_workers, sampler=sampler)


def _data_loader(
    dataset: PatchImageDataset,
    batch_size: int,
    num_workers: int,
    sampler: RandomSampler | WeightedRandomSampler | None = None,
    shuffle: bool | None = None,
    generator: torch.Generator | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def _balanced_sampler(
    labels: np.ndarray, generator: torch.Generator
) -> WeightedRandomSampler:
    counts = np.bincount(labels)
    sample_weights = [float(1.0 / counts[label]) for label in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), True, generator)
