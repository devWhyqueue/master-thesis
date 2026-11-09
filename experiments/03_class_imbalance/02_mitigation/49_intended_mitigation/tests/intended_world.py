"""Synthetic world for the exp-49 tests: exp-2 manifests/freezes, cached features, exp-25 records."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from imbalance_benchmark.common import compute_sha256

from breadth.fit import init_shard

from prevalence.fit import _fit_arm, _shard_context

from lora_ablation import CUDA_CONSTANTS, STAGE1_RECIPES

from intended import SPEC_GRIDS

CLASSES = ["A", "B", "C"]
G = 2
DIM = 4
DAMAGE_ARMS = ("r1", "r100")


def make_manifest(feature_dir: Path, image_dir: Path) -> pd.DataFrame:
    """Three classes; three deep train patients each, two small validation/test patients each."""
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


def _write_features(manifest: pd.DataFrame, path: Path) -> None:
    """Overlapping class clouds, so lambda selection and every mitigation have work to do."""
    rng = np.random.default_rng(0)
    labels = manifest["cancer_type"].map(CLASSES.index).to_numpy()
    features = 0.8 * np.eye(DIM)[labels] + rng.normal(size=(len(manifest), DIM))
    torch.save(torch.from_numpy(features.astype(np.float32)), path)


def make_config(tmp_path: Path, dataset: str) -> dict[str, Any]:
    """A merged exp-49 config pointing at the synthetic world's directories."""
    return {
        "dataset": {"name": dataset},
        "prevalence": {"patients_per_class": G},
        "paths": {"outputs": str(tmp_path / "own" / dataset)},
        "slurm": {"exp2_outputs": str(tmp_path / "exp2")},
        "source_outputs": {
            "damage": str(tmp_path / "damage"),
            "mitigation": str(tmp_path / "lora"),
            "frozen_cuda": str(tmp_path / "frozen_cuda"),
        },
        "mitigation": {
            "cuda": dict(CUDA_CONSTANTS),
            "gcl": {"s": 30, "a": 0.9999, "b": 1e-5},
            "train": {
                "stage1": dict(STAGE1_RECIPES[dataset]),
                "stage2": {"batch_size": 16, "steps": 5, "lr": 0.001, "weight_decay": 0.0},
            },
            "grid": copy.deepcopy(SPEC_GRIDS[dataset]),
        },
        "protocol": {"id": "intended-mitigation-v1"},
    }


def build_world(tmp_path: Path, dataset: str = "bracs") -> dict[str, Any]:
    """Write exp-2 manifests/freezes, features and images; return the exp-49 config."""
    exp2, features, images = tmp_path / "exp2", tmp_path / "features", tmp_path / "images"
    features.mkdir()
    images.mkdir()
    manifest = make_manifest(features, images)
    _write_features(manifest, features / "chunk0.pt")
    for path in manifest["image_path"]:
        Path(path).touch()
    for s in range(3):
        data = exp2 / f"split={s}" / "data"
        data.mkdir(parents=True)
        manifest.to_csv(data / "manifest.csv", index=False)
        (data / "manifest_freeze.json").write_text(
            json.dumps({"class_names": CLASSES, "manifest_sha256": compute_sha256(data / "manifest.csv")})
        )
    return make_config(tmp_path, dataset)


def write_damage_records(config: dict[str, Any], shards: list[tuple[int, int]]) -> None:
    """Fit exp-25's own r1/r100 records into the world's damage root with exp-25's code."""
    damage = {**config, "paths": {"outputs": config["source_outputs"]["damage"]}}
    for split, draw in shards:
        train_df, names, evals, paths = init_shard(damage, split)
        shard = _shard_context(train_df, names, split, draw, G)
        for arm in DAMAGE_ARMS:
            out_dir = paths["results"] / arm / f"draw_{draw}"
            _fit_arm(damage, out_dir, arm, shard, evals, draw)


def patch_freeze_loader(monkeypatch: Any) -> None:
    """Stand-in for the signed-freeze loader (no signatures in the synthetic world)."""
    from breadth import fit as breadth_fit
    from mitigation import data as mitigation_data

    def load(paths: dict[str, Path]) -> dict[str, Any]:
        return json.loads((paths["data"] / "manifest_freeze.json").read_text())

    monkeypatch.setattr(breadth_fit, "load_freeze_meta", load)
    monkeypatch.setattr(mitigation_data, "load_freeze_meta", load)
