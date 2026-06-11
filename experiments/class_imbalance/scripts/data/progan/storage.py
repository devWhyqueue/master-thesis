from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import cast

import pandas as pd
import torch

from scripts.common import write_json
from scripts.data.progan.core import ProGanSettings
from scripts.data.progan.fid import fid_for_paths


def synthetic_output_root(outputs_root: Path, seed: int) -> Path:
    """Return the per-seed synthetic image root directory (seed-level, not variant-level)."""
    return outputs_root / "synthetic_patch_images" / f"seed={seed}"


def variant_output_root(outputs_root: Path, seed: int, variant: int) -> Path:
    """Return the per-variant synthetic image directory for one seed and epoch count."""
    return synthetic_output_root(outputs_root, seed) / f"epochs={variant}"


def count_generated(class_dir: Path) -> int:
    """Count generated JPEG patches for one class."""
    if not class_dir.exists():
        return 0
    return sum(1 for _ in class_dir.glob("*.jpg"))


def class_is_complete(class_dir: Path, expected: int) -> bool:
    """Return whether a class directory contains the expected patch count."""
    return expected == 0 or count_generated(class_dir) == expected


def diagnostics_path(output_root: Path, class_name: str) -> Path:
    """Return the diagnostics JSON path for one generated class (in variant root)."""
    return output_root / ".diagnostics" / f"{class_name}.json"


def load_class_diagnostics(path: Path) -> dict[str, object]:
    """Load per-class ProGAN diagnostics from disk."""
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def save_class_diagnostics(output_root: Path, diagnostics: dict[str, object]) -> None:
    """Persist per-class ProGAN diagnostics next to generated images."""
    class_name = str(diagnostics["class_name"])
    path = diagnostics_path(output_root, class_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, diagnostics)


def fid_payload(
    image_paths: list[Path],
    generated: list[dict[str, object]],
    device: torch.device,
) -> dict[str, object]:
    """Compute FID for one generated class."""
    generated_paths = [Path(str(row["image_path"])) for row in generated]
    value = fid_for_paths(image_paths, generated_paths, device)
    return {
        "status": "computed" if value is not None else "unavailable",
        "value": value,
    }


def collect_rows(output_root: Path) -> list[dict[str, object]]:
    """Collect manifest rows from per-class synthetic directories inside a variant root."""
    rows: list[dict[str, object]] = []
    for path in sorted(output_root.glob("*/*.jpg")):
        if path.parent.name == ".diagnostics":
            continue
        rows.append({"cancer_type": path.parent.name, "image_path": str(path)})
    return rows


def load_diagnostics(output_root: Path) -> list[dict[str, object]]:
    """Load all per-class diagnostics for one variant root."""
    diagnostics: list[dict[str, object]] = []
    for path in sorted((output_root / ".diagnostics").glob("*.json")):
        diagnostics.append(load_class_diagnostics(path))
    return diagnostics


def generated_counts_match(
    rows: list[dict[str, object]], expected: dict[str, int]
) -> bool:
    """Return whether generated rows match the expected per-class counts."""
    counts = pd.DataFrame(rows)["cancer_type"].value_counts().to_dict()
    return counts == expected


def write_variant_manifest(
    variant_root: Path,
    rows: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
    seed: int,
    variant: int,
    settings: ProGanSettings,
) -> Path:
    """Write per-variant synthetic manifest and summary under variant_root."""
    variant_root.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        manifest = pd.DataFrame(
            {"cancer_type": pd.Series(dtype=str), "image_path": pd.Series(dtype=str)}
        )
    manifest["split"] = "train"
    manifest["slide_id"] = manifest["image_path"].map(lambda path: Path(str(path)).stem)
    manifest["resolution"] = "synthetic"
    manifest["final_depth_epochs"] = variant
    path = variant_root / "synthetic_patch_manifest.csv"
    manifest.to_csv(path, index=False)
    write_json(
        variant_root / "synthetic_patch_summary.json",
        {
            "seed": seed,
            "final_depth_epochs": variant,
            "n_patches": int(len(manifest)),
            "counts_by_class": manifest["cancer_type"].value_counts().to_dict(),
            "image_root": str(variant_root),
            "per_class": diagnostics,
            "settings": asdict(settings),
        },
    )
    return path


def write_combined_manifest(seed_root: Path, settings: ProGanSettings) -> Path:
    """Combine per-variant manifests into one manifest covering all epoch variants.

    This combined manifest is used by Virchow2 feature extraction so all variants
    are embedded in a single pass.  Each row carries a final_depth_epochs column so
    _select_progan_variant can filter to the desired variant at training time.
    """
    frames: list[pd.DataFrame] = []
    for variant in sorted(settings.final_depth_epoch_grid):
        variant_manifest = seed_root / f"epochs={variant}" / "synthetic_patch_manifest.csv"
        if variant_manifest.exists():
            frames.append(pd.read_csv(variant_manifest))
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    path = seed_root / "synthetic_patch_manifest.csv"
    combined.to_csv(path, index=False)
    return path


# Legacy alias kept for callers that still write a single-variant manifest directly
# (used only by the image-level patch_progan_aug path via manifest.py).
def write_manifest(
    output_root: Path,
    rows: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
    seed: int,
    settings: ProGanSettings,
    variant: int | None = None,
) -> Path:
    """Write synthetic manifest and summary.  If variant is given, tags rows and
    writes to the variant subdirectory; otherwise writes to output_root directly
    (legacy behaviour for callers that manage their own root)."""
    if variant is not None:
        return write_variant_manifest(output_root, rows, diagnostics, seed, variant, settings)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        manifest = pd.DataFrame(
            {"cancer_type": pd.Series(dtype=str), "image_path": pd.Series(dtype=str)}
        )
    manifest["split"] = "train"
    manifest["slide_id"] = manifest["image_path"].map(lambda path: Path(str(path)).stem)
    manifest["resolution"] = "synthetic"
    path = output_root / "synthetic_patch_manifest.csv"
    manifest.to_csv(path, index=False)
    write_json(
        output_root / "synthetic_patch_summary.json",
        {
            "seed": seed,
            "n_patches": int(len(manifest)),
            "counts_by_class": manifest["cancer_type"].value_counts().to_dict(),
            "image_root": str(output_root),
            "per_class": diagnostics,
            "settings": asdict(settings),
        },
    )
    return path
