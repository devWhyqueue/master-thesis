"""Synthetic world for the exp-49 tests: exp-2 manifests/freezes, cached features, exp-25 records."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from imbalance_benchmark.common import compute_sha256, write_run_record

from breadth.fit import init_shard


from prevalence.fit import _fit_arm, _shard_context

from lora_ablation import CUDA_CONSTANTS, STAGE1_RECIPES

from intended import ARMS, SPEC_GRIDS
from intended.preflight import reused_runs

CLASSES = ["A", "B", "C"]
G = 2
DIM = 4
DAMAGE_ARMS = ("r1", "r100")
LORA_GRID = {
    arm: {
        "stage1": {"ce": [None], "gcl": [0.05] if arm == "r1" else [0.02, 0.05]},
        "stage2": {"crt": [None], "gcl2": [0.05] if arm == "r1" else [0.02, 0.05]},
    }
    for arm in ARMS
}


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
        "sources": {
            "mitigation_grid": copy.deepcopy(LORA_GRID),
            "frozen_cuda_grid": {"stage1": {"ce": [None], "cuda": [0.25, 0.5, 1.0]}},
        },
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
    config = make_config(tmp_path, dataset)
    _write_source_configs(tmp_path, config)
    return config


def _yaml(path: Path, payload: dict[str, Any]) -> str:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return str(path)


def _write_source_configs(tmp_path: Path, config: dict[str, Any]) -> None:
    """Damage, LoRA and frozen-CUDA source configs as YAML files, as the real config references them."""
    base = {"dataset": config["dataset"], "prevalence": config["prevalence"], "slurm": config["slurm"]}
    outputs = config["source_outputs"]
    config["damage_config"] = _yaml(tmp_path / "damage.yaml", {**base, "paths": {"outputs": outputs["damage"]}})
    config["mitigation_config"] = _yaml(
        tmp_path / "mitigation.yaml", {**base, "paths": {"outputs": outputs["mitigation"]}}
    )
    frozen = {"source_config": config["mitigation_config"], "paths": {"outputs": outputs["frozen_cuda"]}}
    config["frozen_cuda_config"] = _yaml(tmp_path / "frozen_cuda.yaml", frozen)


def swap_class_order(config: dict[str, Any]) -> None:
    """Point the LoRA source config at an exp-2 copy whose freeze lists the classes reversed."""
    exp2 = Path(config["slurm"]["exp2_outputs"])
    swapped = exp2.parent / "exp2_swapped"
    shutil.copytree(exp2, swapped)
    for freeze in swapped.glob("split=*/data/manifest_freeze.json"):
        payload = json.loads(freeze.read_text())
        freeze.write_text(json.dumps({**payload, "class_names": CLASSES[::-1]}))
    mitigation = yaml.safe_load(Path(config["mitigation_config"]).read_text(encoding="utf-8"))
    _yaml(Path(config["mitigation_config"]), {**mitigation, "slurm": {"exp2_outputs": str(swapped)}})


def write_source_records(config: dict[str, Any]) -> None:
    """Minimal reused records (test labels/preds, validation BA) for every shard and source run."""
    manifest = pd.read_csv(Path(config["slurm"]["exp2_outputs"]) / "split=0" / "data" / "manifest.csv")
    test_y = manifest.query("split == 'test'")["cancer_type"].map(CLASSES.index).to_numpy()
    for _, directory in reused_runs(config):
        write_run_record(
            directory,
            {
                "selected_lambda": 1e-2,
                "splits": {
                    "validation": {"endpoints": {"patient_macro_balanced_accuracy": 0.5}},
                    "test": {"labels": test_y, "preds": test_y},
                },
            },
        )


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

    from intended import preflight

    def load(paths: dict[str, Path]) -> dict[str, Any]:
        return json.loads((paths["data"] / "manifest_freeze.json").read_text())

    monkeypatch.setattr(breadth_fit, "load_freeze_meta", load)
    monkeypatch.setattr(mitigation_data, "load_freeze_meta", load)
    monkeypatch.setattr(preflight, "load_freeze_meta", load)
