from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from common_code.progan.core import ProGanSettings


def synthetic_output_root(outputs_root: Path, seed: int) -> Path:
    """Return the per-seed synthetic image root directory."""
    return outputs_root / "synthetic_patch_images" / f"seed={seed}"


def diagnostics_path(output_root: Path, class_name: str) -> Path:
    """Return the diagnostics JSON path for one generated class."""
    return output_root / ".diagnostics" / f"{class_name}.json"


def load_class_diagnostics(path: Path) -> dict[str, object]:
    """Load one class diagnostics payload from disk."""
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def save_class_diagnostics(output_root: Path, diagnostics: dict[str, object]) -> None:
    """Persist per-class ProGAN diagnostics next to generated images."""
    class_name = str(diagnostics["class_name"])
    path = diagnostics_path(output_root, class_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")


def collect_rows(output_root: Path) -> list[dict[str, object]]:
    """Collect manifest rows from one variant directory."""
    rows: list[dict[str, object]] = []
    for path in sorted(output_root.glob("*/*.jpg")):
        if path.parent.name == ".diagnostics":
            continue
        rows.append({"cancer_type": path.parent.name, "image_path": str(path)})
    return rows


def load_diagnostics(output_root: Path) -> list[dict[str, object]]:
    """Load all per-class diagnostics for one variant directory."""
    diagnostics: list[dict[str, object]] = []
    for path in sorted((output_root / ".diagnostics").glob("*.json")):
        diagnostics.append(load_class_diagnostics(path))
    return diagnostics


def write_variant_manifest(
    variant_root: Path,
    rows: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
    seed: int,
    variant: int,
    settings: ProGanSettings,
) -> Path:
    """Write the synthetic manifest and summary for one final-depth variant."""
    variant_root.mkdir(parents=True, exist_ok=True)
    manifest = _variant_manifest_frame(rows)
    manifest["split"] = "train"
    manifest["slide_id"] = manifest["image_path"].map(lambda path: Path(str(path)).stem)
    manifest["resolution"] = "synthetic"
    manifest["final_depth_epochs"] = variant
    path = variant_root / "synthetic_patch_manifest.csv"
    manifest.to_csv(path, index=False)
    summary = _variant_summary(
        variant_root, manifest, diagnostics, seed, variant, settings
    )
    (variant_root / "synthetic_patch_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return path


def write_combined_manifest(seed_root: Path, settings: ProGanSettings) -> Path:
    """Combine all per-variant manifests into one synthetic manifest."""
    frames: list[pd.DataFrame] = []
    for variant in sorted(settings.final_depth_epoch_grid):
        manifest_path = seed_root / f"epochs={variant}" / "synthetic_patch_manifest.csv"
        if manifest_path.exists():
            frames.append(pd.read_csv(manifest_path))
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    path = seed_root / "synthetic_patch_manifest.csv"
    combined.to_csv(path, index=False)
    return path


def _variant_manifest_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=cast(Any, ["cancer_type", "image_path"]))


def _variant_summary(
    variant_root: Path,
    manifest: pd.DataFrame,
    diagnostics: list[dict[str, object]],
    seed: int,
    variant: int,
    settings: ProGanSettings,
) -> dict[str, object]:
    return {
        "seed": seed,
        "final_depth_epochs": variant,
        "n_patches": len(manifest),
        "counts_by_class": manifest["cancer_type"].value_counts().to_dict(),
        "image_root": str(variant_root),
        "balance_target": settings.balance_target,
        "max_classes": settings.max_classes,
        "max_real_patches_per_class": settings.max_real_patches_per_class,
        "per_class": diagnostics,
        "settings": asdict(settings),
    }
