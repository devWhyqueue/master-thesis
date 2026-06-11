import json
from pathlib import Path
from typing import cast

import pandas as pd
import torch
from torch.utils.data import Dataset

OPTIONAL_ARGS: tuple[tuple[str, type, float | int], ...] = (
    ("--epochs", int, 30),
    ("--bag-batch-size", int, 32),
    ("--learning-rate", float, 0.001),
    ("--weight-decay", float, 0.0001),
    ("--hidden-dim", int, 256),
    ("--dropout", float, 0.1),
    ("--focal-gamma", float, 2.0),
    ("--max-instances-per-bag", int, 30),
    ("--rankmix-teacher-epochs", int, 30),
    ("--rankmix-alpha", float, 1.0),
    ("--sc-mil-temperature", float, 1.0),
    ("--mde-mil-consistency-weight", float, 0.25),
    ("--sampler-power", float, 1.0),
    ("--weight-power", float, 1.0),
    ("--max-bags-per-class", int, 0),
)


class ConstructedBagDataset(Dataset):
    """WSI-bag dataset assembled from constructed feature-row manifests."""

    def __init__(
        self,
        frame: pd.DataFrame,
        class_to_idx: dict[str, int],
        max_instances: int | None,
        max_bags_per_class: int | None = None,
    ) -> None:
        self.max_instances = max_instances
        grouped = bag_rows(frame)
        if max_bags_per_class:
            grouped = grouped.groupby("cancer_type", group_keys=False).head(
                max_bags_per_class
            )
        self.feature_paths = grouped["feature_paths"].tolist()
        self.labels = torch.tensor(
            [class_to_idx[str(name)] for name in grouped["cancer_type"].tolist()],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.feature_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        paths = cast(list[str], self.feature_paths[idx])
        bag = load_bag(paths, self.max_instances)
        return bag, int(self.labels[idx].item())


def bag_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one row per slide bag, preserving replacement-sampled instances."""
    group_columns = ["slide_id"]
    has_instances = "sample_instance" in frame and bool(
        frame["sample_instance"].notna().any()
    )
    if has_instances:
        group_columns.append("sample_instance")
    rows = []
    for _, group in frame.sort_values(group_columns + ["feature_path"]).groupby(
        group_columns, sort=False, dropna=False
    ):
        rows.append(
            {
                "slide_id": str(group["slide_id"].iloc[0]),
                "cancer_type": str(group["cancer_type"].iloc[0]),
                "feature_paths": group["feature_path"].astype(str).tolist(),
            }
        )
    return pd.DataFrame(rows)


def load_bag(paths: list[str], max_instances: int | None) -> torch.Tensor:
    """Load feature rows for one slide as a capped bag tensor."""
    tensors = [_load_feature(path) for path in paths]
    bag = torch.cat(tensors, dim=0)
    if max_instances and len(bag) > max_instances:
        indices = torch.linspace(0, len(bag) - 1, max_instances).long()
        bag = bag.index_select(0, indices)
    return bag


def _load_feature(path: str) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu")
    if isinstance(tensor, dict):
        tensor = next(value for value in tensor.values() if torch.is_tensor(value))
    feature = tensor.float()
    if feature.ndim == 1:
        feature = feature.unsqueeze(0)
    if feature.ndim > 2:
        feature = feature.reshape(-1, feature.shape[-1])
    return feature


def write_json(path: Path, payload: object) -> None:
    """Write a JSON file with stable formatting."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
