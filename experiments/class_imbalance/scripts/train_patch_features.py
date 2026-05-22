from __future__ import annotations
import argparse
from pathlib import Path
from typing import cast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from scripts.common import ensure_dirs, load_config, write_json
from scripts.patch.artifacts import seed_patch_run
from scripts.patch.data import uses_balanced_sampler
from scripts.patch.losses import (
    PatchFocalLoss,
    ScholzCombinedLoss,
    inverse_frequency_weights,
)
from scripts.patch_feature_cache import patch_feature_cache_dir
from scripts.training.support import _metric_payload, _resolve_device


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


def parse_args() -> argparse.Namespace:
    """Parse patch-feature training arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Train one patch-level method on frozen Virchow2 features."""
    args = parse_args()
    config = load_config(args.config)
    seed_patch_run(args.seed)
    paths = ensure_dirs(config)
    cache_dir = patch_feature_cache_dir(config, args.seed)
    frame = _select_rows(
        pd.read_csv(cache_dir / "manifest.csv"), args.method, args.smoke, config
    )
    class_names = sorted(frame["cancer_type"].unique().tolist())
    datasets = _split_datasets(frame, cache_dir / "features.npy", class_names)
    model = _fit_model(args.method, datasets[0], class_names, config, args.seed)
    device = _resolve_device(str(config["patch_feature_training"]["device"]))
    result_dir = paths["results"] / "patch_feature" / args.method / f"seed={args.seed}"
    result_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), result_dir / "model.pt")
    write_json(
        result_dir / "val_results.json",
        _evaluate(model, datasets[1], class_names, device),
    )
    write_json(
        result_dir / "test_results.json",
        _evaluate(model, datasets[2], class_names, device),
    )


def _fit_model(
    method: str,
    train_set: PatchFeatureDataset,
    class_names: list[str],
    config: dict,
    seed: int,
) -> torch.nn.Module:
    settings = config["patch_feature_training"]
    device = _resolve_device(str(settings["device"]))
    model = _build_model(train_set, settings, len(class_names), device)
    labels = train_set.labels.cpu().numpy()
    criterion = _criterion(method, labels, len(class_names), settings, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    _train(
        model, train_set, labels, criterion, optimizer, method, settings, device, seed
    )
    return model


def _select_rows(
    frame: pd.DataFrame, method: str, smoke: bool, config: dict
) -> pd.DataFrame:
    if method != "patch_feature_progan_aug":
        frame = cast(pd.DataFrame, frame[~frame["is_synthetic"].astype(bool)])
    if smoke:
        parts = [part.head(8) for _, part in frame.groupby("split", sort=False)]
        return pd.concat(parts, ignore_index=True)
    settings = config["patch_feature_training"]
    return _slice(frame, settings.get("max_train_rows"), settings.get("max_eval_rows"))


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


def _split_datasets(
    frame: pd.DataFrame, features_path: Path, class_names: list[str]
) -> tuple[PatchFeatureDataset, PatchFeatureDataset, PatchFeatureDataset]:
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    return tuple(
        PatchFeatureDataset(
            cast(pd.DataFrame, frame[frame["split"] == split]),
            features_path,
            class_to_idx,
        )
        for split in ["train", "val", "test"]
    )  # type: ignore[return-value]


def _build_model(
    dataset: PatchFeatureDataset, settings: dict, n_classes: int, device: torch.device
) -> torch.nn.Module:
    sample, _ = dataset[0]
    model = torch.nn.Sequential(
        torch.nn.Linear(int(sample.numel()), int(settings["hidden_dim"])),
        torch.nn.ReLU(),
        torch.nn.Dropout(float(settings["dropout"])),
        torch.nn.Linear(int(settings["hidden_dim"]), n_classes),
    )
    return model.to(device)


def _criterion(
    method: str,
    labels: np.ndarray,
    n_classes: int,
    settings: dict,
    device: torch.device,
) -> torch.nn.Module:
    if method == "patch_feature_weighted_ce":
        return torch.nn.CrossEntropyLoss(
            weight=inverse_frequency_weights(labels, n_classes).to(device)
        )
    if method == "patch_feature_focal":
        return PatchFocalLoss(float(settings["focal_gamma"]))
    if method == "patch_feature_ce_soft_f1_balanced":
        return ScholzCombinedLoss(n_classes, "f1")
    if method == "patch_feature_ce_soft_mcc_balanced":
        return ScholzCombinedLoss(n_classes, "mcc")
    return torch.nn.CrossEntropyLoss()


def _train(
    model: torch.nn.Module,
    dataset: PatchFeatureDataset,
    labels: np.ndarray,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    method: str,
    settings: dict,
    device: torch.device,
    seed: int,
) -> None:
    loader = _loader(dataset, labels, method, int(settings["batch_size"]), seed)
    for _ in range(1, int(settings["epochs"]) + 1):
        model.train()
        for features, targets in loader:
            loss = criterion(model(features.to(device)), targets.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def _loader(
    dataset: PatchFeatureDataset,
    labels: np.ndarray,
    method: str,
    batch_size: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    if not uses_balanced_sampler(method.replace("patch_feature", "patch", 1)):
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=True, generator=generator
        )
    counts = np.bincount(labels)
    sample_weights = [float(1.0 / counts[int(label)]) for label in labels]
    sampler = WeightedRandomSampler(sample_weights, len(labels), True, generator)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def _evaluate(
    model: torch.nn.Module,
    dataset: PatchFeatureDataset,
    class_names: list[str],
    device: torch.device,
) -> dict[str, object]:
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[list[float]] = []
    model.eval()
    with torch.no_grad():
        for features, targets in loader:
            logits = model(features.to(device))
            probs = torch.softmax(logits, dim=1)
            y_true.extend(targets.numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            probabilities.extend(probs.cpu().numpy().tolist())
    return _metric_payload(y_true, y_pred, probabilities, class_names)


if __name__ == "__main__":
    main()
