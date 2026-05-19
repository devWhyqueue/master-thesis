from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from torch import nn
from scripts.common import ensure_dirs, load_config, write_json, write_progress
from scripts.patch.artifacts import (
    cfal_checkpoint_settings,
    copy_synthetic_artifacts,
    evaluate_patch_dataset,
    save_patch_checkpoint,
    seed_patch_run,
    write_patch_config,
)
from scripts.patch.data import PatchImageDataset, patch_loader
from scripts.patch.losses import (
    PatchFocalLoss,
    cfal_loss,
    effective_number_weights,
    inverse_frequency_weights,
)
from scripts.patch.models import PatchClassifier
from scripts.progan.manifest import (
    generate_patch_gan_manifest,
    merge_patch_gan_manifest,
)
from scripts.training.support import _resolve_device

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse patch-training CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--skip-synthetic-generation",
        action="store_true",
        help="Use an existing merged ProGAN manifest (after parallel GAN jobs).",
    )
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


def _train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    paths = ensure_dirs(config)
    deterministic = seed_patch_run(args.seed)
    frame = _load_manifest(paths, args.seed, args.smoke, config)
    original_train_rows = int((frame["split"] == "train").sum())
    synthetic_manifest: Path | None = None
    if args.method == "patch_progan_aug":
        synthetic_manifest = (
            merge_patch_gan_manifest
            if args.skip_synthetic_generation
            else generate_patch_gan_manifest
        )(config, args.seed, args.smoke)
    if synthetic_manifest is not None:
        frame = pd.concat([frame, pd.read_csv(synthetic_manifest)], ignore_index=True)
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
    if synthetic_manifest is not None:
        copy_synthetic_artifacts(synthetic_manifest, result_dir)
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
    save_patch_checkpoint(
        result_dir,
        args.method,
        args.seed,
        model,
        class_names,
        prototypes if args.method == "patch_cfal" else None,
        cfal_checkpoint_settings(settings) if args.method == "patch_cfal" else None,
    )
    write_patch_config(result_dir, args.method, args.seed, class_names, deterministic)
    write_json(
        result_dir / "val_results.json",
        evaluate_patch_dataset(
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
        evaluate_patch_dataset(
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
