"""Synthetic source world shared by the preflight/fit/analyze tests: manifests, freezes, adapted records."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imbalance_benchmark.common import compute_sha256, write_run_record

from centre import N_DRAWS, N_SPLITS
from mitigation.grid import stage1_jobs, stage2_jobs
from prevalence.fit import _shard_context, class_counts

from lora_ablation import ARM, STAGE1_RECIPES, protocol_grid
from lora_ablation import source_run_dir

CLASSES = ["A", "B"]
G = 2
DIM = 4
PINNED = {
    "encoder_id": "hf-hub:paige-ai/Virchow2",
    "encoder_revision": "3158645804b69e3f3bc4439d4116edddf0840a72",
    "weights_sha256": "8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b",
    "pooling": "cls_plus_mean_patch_tokens",
    "feature_dim": 2560,
    "dtype": "float16",
}
CHECKPOINT = {
    "revision": PINNED["encoder_revision"],
    "weights_sha256": PINNED["weights_sha256"],
}


def make_manifest(feature_dir: Path, image_dir: Path) -> pd.DataFrame:
    """Two classes; three deep train patients each, two small validation/test patients each."""
    rows: list[dict[str, Any]] = []

    def add(split: str, cls: str, patient: str, n: int) -> None:
        for i in range(n):
            rows.append(
                {
                    "split": split,
                    "cancer_type": cls,
                    "case_id": patient,
                    "slide_id": f"{patient}_s0",
                    "patch_id": f"{patient}_p{i:03d}",
                    "image_path": str(image_dir / f"{patient}_{i}.jpg"),
                    "feature_path": str(feature_dir / "chunk0.pt"),
                    "feature_index": len(rows),
                }
            )

    for cls in CLASSES:
        for k in range(3):
            add("train", cls, f"{cls}tr{k}", 170)
        for split in ("validation", "test"):
            for k in range(2):
                add(split, cls, f"{cls}{split[:2]}{k}", 6)
    return pd.DataFrame(rows)


def expected_counts(train_df: pd.DataFrame, split: int, draw: int) -> list[int]:
    """Per-class training counts the exp-25 allocator gives one (split, draw) at r100."""
    shard = _shard_context(train_df, CLASSES, split, draw, G)
    return class_counts(ARM, shard.perm, shard.available, shard.pool_counts, shard.g)


def write_record(
    directory: Path,
    counts: dict[str, int],
    labels: np.ndarray,
    preds: np.ndarray,
    validation_ba: float = 0.5,
) -> None:
    """One minimal run record: class counts, test labels/preds, and a validation BA endpoint."""
    write_run_record(
        directory,
        {
            "class_counts": counts,
            "splits": {
                "validation": {"endpoints": {"patient_macro_balanced_accuracy": validation_ba}},
                "test": {"labels": labels, "preds": preds},
            },
        },
    )


def make_config(tmp_path: Path, dataset: str) -> dict[str, Any]:
    """A merged v2 config pointing at the synthetic world's directories."""
    grid = protocol_grid(dataset)
    return {
        "dataset": {"name": dataset},
        "prevalence": {"patients_per_class": G},
        "paths": {"outputs": str(tmp_path / "new_out" / "strong_mitigations_v2" / dataset / "patch")},
        "source_outputs": str(tmp_path / "src_out"),
        "slurm": {"exp2_outputs": str(tmp_path / "exp2")},
        "mitigation": {
            "cuda": {"gamma": 0.6, "check_size": 10, "max_strength": 10},
            "train": {"stage1": dict(STAGE1_RECIPES[dataset])},
            "grid": {ARM: copy.deepcopy(grid)},
        },
        "protocol": {
            "id": "strong-mitigations-v2",
            "margin_pp": 1.0,
            "init_seed_base": 10000,
            "grid": copy.deepcopy(grid),
        },
        "source_grid": {ARM: copy.deepcopy(grid)},
    }


def build_world(tmp_path: Path, dataset: str = "bracs", records: bool = True) -> dict[str, Any]:
    """Write exp-2 manifests/freezes, features, images, and source run records; return a config."""
    exp2, features, images = tmp_path / "exp2", tmp_path / "features", tmp_path / "images"
    features.mkdir()
    images.mkdir()
    (features / "feature_provenance.json").write_text(json.dumps(PINNED))
    manifest = make_manifest(features, images)
    for path in manifest["image_path"]:
        Path(path).touch()
    config = make_config(tmp_path, dataset)
    for s in range(N_SPLITS):
        data = exp2 / f"split={s}" / "data"
        data.mkdir(parents=True)
        manifest.to_csv(data / "manifest.csv", index=False)
        (data / "manifest_freeze.json").write_text(
            json.dumps({"class_names": CLASSES, "manifest_sha256": compute_sha256(data / "manifest.csv")})
        )
    if records:
        _write_source_records(config, manifest)
    return config


def _write_source_records(config: dict[str, Any], manifest: pd.DataFrame) -> None:
    test_y = np.array([CLASSES.index(c) for c in manifest.query("split == 'test'")["cancer_type"]])
    train_df = manifest.query("split == 'train'").reset_index(drop=True)
    for s in range(N_SPLITS):
        for d in range(N_DRAWS):
            counts = dict(zip(CLASSES, expected_counts(train_df, s, d)))
            for arm, method, param in (*stage1_jobs(config), *stage2_jobs(config)):
                write_record(source_run_dir(config, s, arm, method, param, d), counts, test_y, test_y)


def patch_freeze_loader(monkeypatch: Any) -> None:
    """Stand-ins for the signed-freeze loader and the cached-checkpoint probe (no signatures/downloads)."""
    from breadth.analyze import canonical
    from lora_ablation import preflight

    def load(paths: dict[str, Path]) -> dict[str, Any]:
        return json.loads((paths["data"] / "manifest_freeze.json").read_text())

    monkeypatch.setattr(preflight, "load_freeze_meta", load)
    monkeypatch.setattr(canonical, "load_freeze_meta", load)
    monkeypatch.setattr(preflight, "cached_checkpoint", lambda: dict(CHECKPOINT))
