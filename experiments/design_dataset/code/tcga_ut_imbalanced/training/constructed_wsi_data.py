import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from tcga_ut_imbalanced.data.feature_store import load_feature_row, load_slide_features
from tcga_ut_imbalanced.evaluation.tuning_params import parse_tuning_params

OPTIONAL_ARGS: tuple[tuple[str, type, object], ...] = (
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
    ("--bag-cache-dir", str, ""),
)


class ConstructedBagDataset(Dataset):
    """WSI-bag dataset assembled from constructed feature-row manifests."""

    def __init__(
        self,
        frame: pd.DataFrame,
        class_to_idx: dict[str, int],
        max_instances: int | None,
        max_bags_per_class: int | None = None,
        cache_dir: str | None = None,
        split: str | None = None,
    ) -> None:
        self.max_instances = max_instances
        grouped = bag_rows(frame)
        if max_bags_per_class:
            grouped = grouped.groupby("cancer_type", group_keys=False).head(
                max_bags_per_class
            )
        self._use_cache = (
            cache_dir is not None
            and split is not None
            and _cache_exists(Path(cache_dir), split)
        )
        if self._use_cache:
            assert cache_dir is not None and split is not None
            cache_path = Path(cache_dir)
            self.features = np.load(cache_path / f"{split}_features.npy", mmap_mode="r")
            self.offsets = np.load(cache_path / f"{split}_offsets.npy")
            self.feature_paths: list[list[str]] = []
            self.feature_indices = None
        else:
            self.features = np.empty((0, 0), dtype=np.float32)
            self.offsets = np.zeros(0, dtype=np.int64)
            self.feature_paths = grouped["feature_paths"].tolist()
            self.feature_indices = (
                grouped["feature_indices"].tolist()
                if "feature_indices" in grouped.columns
                else None
            )
        self.labels = torch.tensor(
            [class_to_idx[str(name)] for name in grouped["cancer_type"].tolist()],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        if self._use_cache:
            bag = self._bag_from_cache(idx)
        else:
            paths = cast(list[str], self.feature_paths[idx])
            indices = None
            if self.feature_indices is not None:
                indices = cast(list[int], self.feature_indices[idx])
            bag = load_bag_paths(paths, indices, self.max_instances)
        return bag, int(self.labels[idx].item())

    def _bag_from_cache(self, idx: int) -> torch.Tensor:
        start, end = int(self.offsets[idx]), int(self.offsets[idx + 1])
        array = np.asarray(self.features[start:end], dtype=np.float32)
        bag = torch.from_numpy(array.copy())
        if self.max_instances and len(bag) > self.max_instances:
            indices = torch.linspace(0, len(bag) - 1, self.max_instances).long()
            bag = bag.index_select(0, indices)
        return bag


def bag_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one row per slide bag, preserving explicit sampled instances."""
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
        paths = group["feature_path"].astype(str).tolist()
        indices = None
        if "feature_index" in group.columns:
            indices = [int(value) for value in group["feature_index"].tolist()]
        rows.append(
            {
                "slide_id": str(group["slide_id"].iloc[0]),
                "cancer_type": str(group["cancer_type"].iloc[0]),
                "feature_paths": paths,
                "feature_indices": indices,
            }
        )
    return pd.DataFrame(rows)


def load_bag_rows(row: pd.Series) -> torch.Tensor:
    """Load one bag tensor from a grouped manifest row."""
    paths = cast(list[str], row["feature_paths"])
    indices = row.get("feature_indices")
    index_list = None
    if indices is not None and not (isinstance(indices, float) and np.isnan(indices)):
        index_list = cast(list[int], indices)
    return load_bag_paths(paths, index_list, None)


def load_bag_paths(
    paths: list[str],
    indices: list[int] | None,
    max_instances: int | None,
) -> torch.Tensor:
    """Load feature rows for one slide bag."""
    if indices is not None and len(set(paths)) == 1:
        bag = load_slide_features(paths[0]).index_select(
            0, torch.tensor(indices, dtype=torch.long)
        )
    else:
        tensors = [
            _load_feature(path, index)
            for path, index in zip(
                paths,
                indices if indices is not None else [None] * len(paths),
                strict=True,
            )
        ]
        bag = torch.cat(tensors, dim=0)
    if max_instances and len(bag) > max_instances:
        select = torch.linspace(0, len(bag) - 1, max_instances).long()
        bag = bag.index_select(0, select)
    return bag


def _load_feature(path: str, index: int | None) -> torch.Tensor:
    if index is not None:
        return load_feature_row(path, index).unsqueeze(0)
    tensor = load_slide_features(path)
    if tensor.shape[0] == 1:
        return tensor
    return tensor


def _cache_exists(cache_dir: Path, split: str) -> bool:
    return (cache_dir / f"{split}_features.npy").exists() and (
        cache_dir / f"{split}_offsets.npy"
    ).exists()


def write_json(path: Path, payload: object) -> None:
    """Write a JSON file with stable formatting."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def write_training_outputs(
    output_dir: Path,
    args: argparse.Namespace,
    class_names: list[str],
    results: dict[str, dict[str, object]],
    diagnostics: dict[str, int],
) -> None:
    """Write validation, test, and run metadata for one WSI job."""
    write_json(output_dir / "validation_results.json", results["val"])
    write_json(output_dir / "test_results.json", results["test"])
    write_json(
        output_dir / "args.json",
        _args_payload(args, class_names, diagnostics),
    )
    write_json(output_dir / "run.json", _run_payload(args, results, diagnostics))


def _args_payload(
    args: argparse.Namespace,
    class_names: list[str],
    diagnostics: dict[str, int],
) -> dict[str, object]:
    return {
        **vars(args),
        "class_names": class_names,
        "diagnostics": diagnostics,
        "benchmark": "wsi_bag",
    }


def _run_payload(
    args: argparse.Namespace,
    results: dict[str, dict[str, object]],
    diagnostics: dict[str, int],
) -> dict[str, object]:
    return {
        "benchmark": "wsi_bag",
        "method": args.method,
        "seed": args.seed,
        "smoke": False,
        "tuning_id": args.tuning_id,
        "tuning_params": parse_tuning_params("wsi", args.method, args.tuning_params),
        "model_path": "model.pt",
        "diagnostics": diagnostics,
        "splits": results,
    }
