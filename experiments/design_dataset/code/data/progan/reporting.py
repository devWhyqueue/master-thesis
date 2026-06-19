from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common_code.progan import ProGanSettings, save_class_diagnostics
from data.progan.manifest import tail_classes, train_rows


def write_class_diagnostics(
    variant_dir: Path,
    class_name: str,
    variant: int,
    generated_patches: int,
    real_train_patches: int,
    training: list[dict[str, object]],
) -> None:
    """Write one per-class diagnostics payload for a ProGAN variant."""
    save_class_diagnostics(
        variant_dir,
        {
            "class_name": class_name,
            "final_depth_epochs": variant,
            "generated_patches": generated_patches,
            "real_train_patches": real_train_patches,
            "training": training,
        },
    )


def write_summary(
    args: argparse.Namespace,
    manifest: pd.DataFrame,
    combined: pd.DataFrame,
    settings: ProGanSettings,
) -> None:
    """Write one run-level summary of the constructed ProGAN augmentation."""
    train = train_rows(manifest)
    combined_train = train_rows(combined)
    summary = {
        "cache_path": str(args.file_save_path),
        "final_depth_epochs": list(settings.final_depth_epoch_grid),
        "manifest_path": str(args.manifest_save_path),
        "n_real_train_rows": int(len(train)),
        "n_synthetic_train_rows": int(
            combined_train["is_synthetic"].astype(bool).sum()
        ),
        "seed": int(args.seed),
        "settings": settings.__dict__,
        "tail_classes": tail_classes(manifest, settings.max_classes),
        "train_counts_after": combined_train["cancer_type"].value_counts().to_dict(),
        "train_counts_before": train["cancer_type"].value_counts().to_dict(),
    }
    path = Path(args.synthetic_root) / "synthetic_patch_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
