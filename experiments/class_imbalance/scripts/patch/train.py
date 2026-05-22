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
    copy_synthetic_artifacts,
    evaluate_patch_dataset,
    load_patch_checkpoint,
    load_training_checkpoint,
    save_patch_checkpoint,
    save_training_checkpoint,
    seed_patch_run,
    write_patch_config,
)
from scripts.patch.data import PatchImageDataset, patch_loader
from scripts.patch.losses import (
    PatchFocalLoss,
    ScholzCombinedLoss,
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from checkpoint_latest.pt in the result directory.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore checkpoint_latest.pt and train from scratch.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run validation and test evaluation from checkpoint_latest.pt or checkpoint.pt.",
    )
    parser.add_argument(
        "--staged-manifest",
        default=None,
        help="Train from a node-local staged manifest (see scripts.staging.patch).",
    )
    return parser.parse_args()


def _load_manifest(
    paths: dict[str, Path],
    seed: int,
    smoke: bool,
    config: dict,
    staged_manifest: str | None = None,
) -> pd.DataFrame:
    manifest_path = staged_manifest or str(
        paths["data"] / f"patch_manifest_seed={seed}.csv"
    )
    frame = pd.read_csv(manifest_path)
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
    if method == "patch_ce_soft_f1_balanced":
        return ScholzCombinedLoss(n_classes, "f1")
    if method == "patch_ce_soft_mcc_balanced":
        return ScholzCombinedLoss(n_classes, "mcc")
    return nn.CrossEntropyLoss()


def _resolve_checkpoint_path(result_dir: Path) -> Path | None:
    latest = result_dir / "checkpoint_latest.pt"
    if latest.exists():
        return latest
    final = result_dir / "checkpoint.pt"
    return final if final.exists() else None


def _write_eval_results(
    result_dir: Path,
    model: PatchClassifier,
    val_set: PatchImageDataset,
    test_set: PatchImageDataset,
    class_names: list[str],
    device: torch.device,
) -> None:
    write_json(
        result_dir / "val_results.json",
        evaluate_patch_dataset(model, val_set, class_names, device),
    )
    write_json(
        result_dir / "test_results.json",
        evaluate_patch_dataset(model, test_set, class_names, device),
    )


def _train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    paths = ensure_dirs(config)
    deterministic = seed_patch_run(args.seed)
    frame = _load_manifest(
        paths, args.seed, args.smoke, config, staged_manifest=args.staged_manifest
    )
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
    epochs = int(settings["epochs"])
    model = PatchClassifier(
        int(settings["hidden_dim"]), len(class_names), float(settings["dropout"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    result_dir = paths["patch_results"] / args.method / f"seed={args.seed}"
    result_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        checkpoint_path = _resolve_checkpoint_path(result_dir)
        if checkpoint_path is None:
            raise FileNotFoundError(f"No checkpoint found under {result_dir}")
        class_names = load_patch_checkpoint(checkpoint_path, model, device)
        _write_eval_results(result_dir, model, val_set, test_set, class_names, device)
        save_patch_checkpoint(result_dir, args.method, args.seed, model, class_names)
        return

    criterion = _criterion(
        args.method, labels, len(class_names), float(settings["focal_gamma"]), device
    ).to(device)
    loader = patch_loader(
        train_set,
        labels,
        args.method,
        int(settings["batch_size"]),
        args.seed,
        int(settings.get("num_workers", 0)),
        original_train_rows if args.method == "patch_progan_aug" else None,
    )
    if synthetic_manifest is not None:
        copy_synthetic_artifacts(synthetic_manifest, result_dir)

    start_epoch = 1
    resume_path = result_dir / "checkpoint_latest.pt"
    if args.resume and not args.fresh and resume_path.exists():
        start_epoch, class_names = load_training_checkpoint(
            resume_path, model, optimizer, device
        )
        logger.info(
            "Resuming %s seed=%s from epoch %s/%s",
            args.method,
            args.seed,
            start_epoch,
            epochs,
        )
        if start_epoch > epochs:
            logger.info("Training already complete; running evaluation only.")
            _write_eval_results(
                result_dir, model, val_set, test_set, class_names, device
            )
            save_patch_checkpoint(
                result_dir, args.method, args.seed, model, class_names
            )
            write_patch_config(
                result_dir, args.method, args.seed, class_names, deterministic
            )
            return

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        losses: list[float] = []
        for images, targets in loader:
            logits, _ = model(images.to(device))
            targets = targets.to(device)
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
                "epochs": epochs,
                "loss": float(np.mean(losses)),
            },
        )
        save_training_checkpoint(
            result_dir,
            args.method,
            args.seed,
            model,
            optimizer,
            class_names,
            epoch,
            epochs,
        )

    save_patch_checkpoint(result_dir, args.method, args.seed, model, class_names)
    write_patch_config(result_dir, args.method, args.seed, class_names, deterministic)
    _write_eval_results(result_dir, model, val_set, test_set, class_names, device)


def main() -> None:
    """Train one patch-level benchmark method."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    _train(parse_args())


if __name__ == "__main__":
    main()
