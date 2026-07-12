from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from code.common import ensure_dirs, load_config, write_progress
from code.modeling.patch.artifacts import (
    copy_synthetic_artifacts,
    load_patch_checkpoint,
    load_training_checkpoint,
    save_patch_checkpoint,
    save_training_checkpoint,
    seed_patch_run,
    _resolve_checkpoint_path,
    _write_run_record,
)
from code.modeling.patch.data import (
    patch_loader,
    _labels,
    _load_manifest,
    _split_datasets,
)
from code.modeling.patch.losses import _criterion
from code.modeling.patch.models import PatchClassifier
from code.data.progan.manifest import (
    generate_patch_gan_manifest,
    merge_patch_gan_manifest_reference,
)
from code.modeling.training.support import _resolve_device

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse patch-training CLI arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--method", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--skip-synthetic-generation", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--staged-manifest", default=None)
    return p.parse_args()


def _train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    paths = ensure_dirs(config)
    deterministic = seed_patch_run(args.seed)
    frame = _load_manifest(
        paths, args.seed, args.smoke, config, staged_manifest=args.staged_manifest
    )
    # Count only real (non-synthetic) train rows so the ProGAN epoch cap matches the
    # unaugmented budget regardless of whether a staged manifest already has synthetics.
    if "is_synthetic" in frame.columns:
        original_train_rows = int(
            ((frame["split"] == "train") & ~frame["is_synthetic"]).sum()
        )
    else:
        original_train_rows = int((frame["split"] == "train").sum())
    synthetic_manifest: Path | None = None
    if args.method == "patch_progan_aug":
        if args.skip_synthetic_generation:
            synthetic_manifest = merge_patch_gan_manifest_reference(
                config, args.seed, args.smoke
            )
        else:
            generate_patch_gan_manifest(config, args.seed, args.smoke)
            synthetic_manifest = merge_patch_gan_manifest_reference(
                config, args.seed, args.smoke
            )
    # Only append synthetic rows when staging has not already merged them.
    # A staged manifest produced with --include-synthetic already contains
    # the per-epoch synthetic images; appending again would double-count them.
    if synthetic_manifest is not None and args.staged_manifest is None:
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
        _write_run_record(
            result_dir,
            args.method,
            args.seed,
            class_names,
            deterministic,
            model,
            val_set,
            test_set,
            device,
        )
        save_patch_checkpoint(result_dir, args.method, args.seed, model, class_names)
        return

    criterion = _criterion(
        args.method, labels, len(class_names), float(settings["focal_gamma"]), device
    ).to(device)
    _w = os.environ.get("PATCH_TRAINING_NUM_WORKERS")
    num_workers = int(_w) if _w is not None else int(settings.get("num_workers", 0))
    loader = patch_loader(
        train_set,
        labels,
        args.method,
        int(settings["batch_size"]),
        args.seed,
        num_workers,
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
            _write_run_record(
                result_dir,
                args.method,
                args.seed,
                class_names,
                deterministic,
                model,
                val_set,
                test_set,
                device,
            )
            save_patch_checkpoint(
                result_dir, args.method, args.seed, model, class_names
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
    _write_run_record(
        result_dir,
        args.method,
        args.seed,
        class_names,
        deterministic,
        model,
        val_set,
        test_set,
        device,
    )


def main() -> None:
    """Train one patch-level benchmark method."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    _train(parse_args())


if __name__ == "__main__":
    main()
