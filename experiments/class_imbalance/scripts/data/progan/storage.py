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
    """Return the per-seed synthetic image root directory."""
    return outputs_root / "synthetic_patch_images" / f"seed={seed}"


def count_generated(class_dir: Path) -> int:
    """Count generated JPEG patches for one class."""
    if not class_dir.exists():
        return 0
    return sum(1 for _ in class_dir.glob("*.jpg"))


def class_is_complete(class_dir: Path, expected: int) -> bool:
    """Return whether a class directory contains the expected patch count."""
    return expected == 0 or count_generated(class_dir) == expected


def diagnostics_path(output_root: Path, class_name: str) -> Path:
    """Return the diagnostics JSON path for one generated class."""
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
    """Collect manifest rows from per-class synthetic directories."""
    rows: list[dict[str, object]] = []
    for path in sorted(output_root.glob("*/*.jpg")):
        if path.parent.name == ".diagnostics":
            continue
        rows.append({"cancer_type": path.parent.name, "image_path": str(path)})
    return rows


def load_diagnostics(output_root: Path) -> list[dict[str, object]]:
    """Load all per-class diagnostics for one seed."""
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


def write_manifest(
    output_root: Path,
    rows: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
    seed: int,
    settings: ProGanSettings,
) -> Path:
    """Write merged synthetic manifest and summary files for one seed."""
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
