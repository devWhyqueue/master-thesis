from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from code.modeling.patch.data import uses_balanced_sampler
from code.modeling.patch_feature.cfal import CfalPrototypeClassifier
from code.modeling.training.support import _metric_payload


class PatchFeatureDataset(Dataset):
    """Frozen patch-feature dataset backed by one memmap array."""

    def __init__(
        self, frame: pd.DataFrame, features_path: Path, class_to_idx: dict[str, int]
    ) -> None:
        self.rows = frame.reset_index(drop=True)
        self.features = np.load(features_path, mmap_mode="r")
        self.indices = self.rows["feature_index"].to_numpy(dtype=np.int64)
        self.labels = torch.tensor(
            [class_to_idx[str(name)] for name in self.rows["cancer_type"]],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        feature = np.asarray(self.features[int(self.indices[idx])], dtype=np.float32)
        return torch.from_numpy(feature.copy()), int(self.labels[idx].item())


def select_patch_feature_rows(
    frame: pd.DataFrame,
    method: str,
    smoke: bool,
    config: dict,
    *,
    seed: int = 0,
    tuning_params: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Select manifest rows for one patch-feature run."""
    params = tuning_params or {}
    if method != "patch_feature_progan_aug":
        frame = cast(pd.DataFrame, frame[~frame["is_synthetic"].astype(bool)])
    else:
        frame = _select_progan_variant(frame, int(params.get("final_depth_epochs", 25)))
    if smoke:
        parts = [part.head(8) for _, part in frame.groupby("split", sort=False)]
        return pd.concat(parts, ignore_index=True)
    settings = config["patch_feature_training"]
    return _slice(frame, settings.get("max_train_rows"), settings.get("max_eval_rows"))


def split_patch_feature_datasets(
    frame: pd.DataFrame, features_path: Path, class_names: list[str]
) -> tuple[PatchFeatureDataset, PatchFeatureDataset, PatchFeatureDataset]:
    """Split the patch-feature manifest into train, validation, and test datasets."""
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    return tuple(
        PatchFeatureDataset(
            cast(pd.DataFrame, frame[frame["split"] == split]),
            features_path,
            class_to_idx,
        )
        for split in ["train", "val", "test"]
    )  # type: ignore[return-value]


def build_patch_feature_model(
    dataset: PatchFeatureDataset, settings: dict, n_classes: int, device: torch.device
) -> torch.nn.Module:
    """Build the frozen-feature classifier head."""
    sample, _ = dataset[0]
    model = torch.nn.Sequential(
        torch.nn.Linear(int(sample.numel()), int(settings["hidden_dim"])),
        torch.nn.ReLU(),
        torch.nn.Dropout(float(settings["dropout"])),
        torch.nn.Linear(int(settings["hidden_dim"]), n_classes),
    )
    return model.to(device)


def evaluate_patch_feature_model(
    model: torch.nn.Module,
    dataset: PatchFeatureDataset,
    class_names: list[str],
    device: torch.device,
) -> dict[str, object]:
    """Evaluate a patch-feature classifier on one split."""
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[list[float]] = []
    model.eval()
    with torch.no_grad():
        for features, targets in loader:
            scores = model(features.to(device))
            logits = (
                torch.log(scores.clamp(min=1e-8))
                if isinstance(model, CfalPrototypeClassifier)
                else scores
            )
            probs = torch.softmax(logits, dim=1)
            y_true.extend(targets.numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            probabilities.extend(probs.cpu().numpy().tolist())
    return _metric_payload(y_true, y_pred, probabilities, class_names)


def patch_feature_train_loader(
    dataset: PatchFeatureDataset,
    labels: np.ndarray,
    method: str,
    batch_size: int,
    seed: int,
    sampler_power: float = 1.0,
) -> DataLoader:
    """Build the training loader for one patch-feature method."""
    generator = torch.Generator().manual_seed(seed)
    patch_method = method.replace("patch_feature", "patch", 1)
    if not uses_balanced_sampler(patch_method):
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=True, generator=generator
        )
    counts = np.bincount(labels)
    sample_weights = [
        float((1.0 / counts[int(label)]) ** sampler_power) for label in labels
    ]
    sampler = WeightedRandomSampler(sample_weights, len(labels), True, generator)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def _select_progan_variant(
    frame: pd.DataFrame, final_depth_epochs: int
) -> pd.DataFrame:
    """Keep all real rows and only synthetic rows for the requested epoch variant."""
    real = cast(pd.DataFrame, frame[~frame["is_synthetic"].astype(bool)])
    synthetic = cast(pd.DataFrame, frame[frame["is_synthetic"].astype(bool)])
    if synthetic.empty:
        return real
    variant = cast(
        pd.DataFrame,
        synthetic[synthetic["final_depth_epochs"].astype(int) == final_depth_epochs],
    )
    return pd.concat([real, variant], ignore_index=True)


def _slice(
    frame: pd.DataFrame, max_train: int | None, max_eval: int | None
) -> pd.DataFrame:
    rows = []
    for split, max_rows in [
        ("train", max_train),
        ("val", max_eval),
        ("test", max_eval),
    ]:
        part = cast(pd.DataFrame, frame[frame["split"] == split])
        rows.append(part.head(int(max_rows)) if max_rows else part)
    return pd.concat(rows, ignore_index=True)
