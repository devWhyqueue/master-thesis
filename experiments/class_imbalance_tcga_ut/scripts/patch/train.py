from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.common import ensure_dirs, load_config, write_json, write_progress
from scripts.metadata import benchmark_metadata
from scripts.patch.data import PatchImageDataset, patch_loader
from scripts.patch.losses import (
    PatchFocalLoss,
    cfal_loss,
    effective_number_weights,
    gaussian_affinity,
    inverse_frequency_weights,
)
from scripts.patch.models import PatchClassifier
from scripts.patch.synthetic import generate_patch_gan_manifest
from scripts.training.support import _metric_payload, _resolve_device

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse patch-training CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _load_manifest(
    paths: dict[str, Path], seed: int, smoke: bool, config: dict
) -> pd.DataFrame:
    frame = pd.read_csv(paths["data"] / f"patch_manifest_seed={seed}.csv")
    training = config["patch_training"]
    if not smoke:
        return frame
    limits = {
        "train": int(training.get("max_train_rows") or 64),
        "val": int(training.get("max_eval_rows") or 32),
        "test": int(training.get("max_eval_rows") or 32),
    }
    parts = [part.head(limits[str(name)]) for name, part in frame.groupby("split")]
    return pd.concat(parts, ignore_index=True)


def _split_datasets(
    frame: pd.DataFrame, config: dict
) -> tuple[PatchImageDataset, PatchImageDataset, PatchImageDataset, list[str]]:
    class_names = sorted(frame["cancer_type"].unique().tolist())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    image_size = int(config["patch_training"]["image_size"])
    datasets = []
    for split in ["train", "val", "test"]:
        datasets.append(
            PatchImageDataset(
                cast(pd.DataFrame, frame[frame["split"] == split]),
                class_to_idx,
                image_size,
            )
        )
    return datasets[0], datasets[1], datasets[2], class_names


def _labels(dataset: PatchImageDataset) -> np.ndarray:
    return np.asarray(
        [dataset.class_to_idx[str(name)] for name in dataset.rows["cancer_type"]],
        dtype=np.int64,
    )


def _criterion(
    method: str, labels: np.ndarray, n_classes: int, gamma: float, device: torch.device
) -> nn.Module:
    if method == "patch_weighted_ce":
        return nn.CrossEntropyLoss(
            weight=inverse_frequency_weights(labels, n_classes).to(device)
        )
    if method == "patch_focal":
        return PatchFocalLoss(gamma)
    return nn.CrossEntropyLoss()


def _evaluate(
    model: PatchClassifier,
    dataset: PatchImageDataset,
    class_names: list[str],
    device: torch.device,
    prototypes: torch.Tensor | None = None,
    cfal_sigma: float | None = None,
) -> dict[str, object]:
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[list[float]] = []
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            logits, embeddings = model(images.to(device))
            if prototypes is not None and cfal_sigma is not None:
                logits = gaussian_affinity(embeddings, prototypes, cfal_sigma)
            probs = torch.softmax(logits, dim=1)
            y_true.extend(targets.numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            probabilities.extend(probs.cpu().numpy().tolist())
    return _metric_payload(y_true, y_pred, probabilities, class_names)


def _train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    paths = ensure_dirs(config)
    frame = _load_manifest(paths, args.seed, args.smoke, config)
    original_train_rows = int((frame["split"] == "train").sum())
    if args.method == "patch_progan_aug":
        manifest = generate_patch_gan_manifest(config, args.seed, args.smoke)
        frame = pd.concat([frame, pd.read_csv(manifest)], ignore_index=True)
    train_set, val_set, test_set, class_names = _split_datasets(frame, config)
    labels = _labels(train_set)
    settings = config["patch_training"]
    device = _resolve_device(str(settings["device"]))
    model = PatchClassifier(
        int(settings["hidden_dim"]), len(class_names), float(settings["dropout"])
    ).to(device)
    prototypes = nn.Parameter(
        torch.randn(len(class_names), int(settings["hidden_dim"]), device=device)
    )
    params = list(model.parameters()) + (
        [prototypes] if args.method == "patch_cfal" else []
    )
    optimizer = torch.optim.AdamW(
        params,
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    criterion = _criterion(
        args.method, labels, len(class_names), float(settings["focal_gamma"]), device
    )
    loader = patch_loader(
        train_set,
        labels,
        args.method,
        int(settings["batch_size"]),
        args.seed,
        int(settings.get("num_workers", 0)),
        original_train_rows if args.method == "patch_progan_aug" else None,
    )
    result_dir = paths["patch_results"] / args.method / f"seed={args.seed}"
    result_dir.mkdir(parents=True, exist_ok=True)
    class_weights = effective_number_weights(
        labels, len(class_names), float(settings["cfal_beta"])
    ).to(device)
    for epoch in range(1, int(settings["epochs"]) + 1):
        model.train()
        losses: list[float] = []
        for images, targets in loader:
            logits, embeddings = model(images.to(device))
            targets = targets.to(device)
            if args.method == "patch_cfal":
                loss = cfal_loss(
                    embeddings,
                    targets,
                    prototypes,
                    class_weights,
                    float(settings["cfal_sigma"]),
                    float(settings["cfal_margin"]),
                    float(settings["cfal_gamma"]),
                )
            else:
                loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        write_progress(
            result_dir / "progress.json",
            {
                "method": args.method,
                "seed": args.seed,
                "epoch": epoch,
                "epochs": int(settings["epochs"]),
                "loss": float(np.mean(losses)),
            },
        )
    torch.save(model.state_dict(), result_dir / "model.pt")
    write_json(
        result_dir / "config.json",
        {
            "benchmark": "patch",
            "method": args.method,
            "seed": args.seed,
            "method_metadata": benchmark_metadata("patch", args.method),
        },
    )
    write_json(
        result_dir / "val_results.json",
        _evaluate(
            model,
            val_set,
            class_names,
            device,
            prototypes if args.method == "patch_cfal" else None,
            float(settings["cfal_sigma"]) if args.method == "patch_cfal" else None,
        ),
    )
    write_json(
        result_dir / "test_results.json",
        _evaluate(
            model,
            test_set,
            class_names,
            device,
            prototypes if args.method == "patch_cfal" else None,
            float(settings["cfal_sigma"]) if args.method == "patch_cfal" else None,
        ),
    )


def main() -> None:
    """Train one patch-level benchmark method."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    _train(parse_args())


if __name__ == "__main__":
    main()
