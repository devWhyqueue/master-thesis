"""Locked four-arm design, loss weights, block seeds, and protocol config loading (exp-48).

Arm ``B`` trains on the balanced ``r1`` allocation with uniform class influence, ``F`` keeps those
rows but weights the loss to the paired ``r100`` class shares, ``C`` trains on the ``r100`` rows
with uniform class influence, and ``R`` trains on the ``r100`` rows unweighted. A fixed row weight
``w_c = q_c / (n_c / N)`` moves class influence without changing which rows are sampled.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.common import REPO_ROOT, load_config

from centre import N_DRAWS, N_SPLITS

__all__ = [
    "ARMS",
    "BALANCED",
    "EXPERIMENT_DIR",
    "IMBALANCED",
    "PROTOCOL_ID",
    "TRAIN_ROWS",
    "arm_allocations",
    "block_seed",
    "config_digest",
    "digest",
    "is_smoke",
    "load_protocol_config",
    "loss_weights",
    "validate_protocol",
    "workload_counts",
]

PROTOCOL_ID = "lora-decomposition-v1"
ARMS: tuple[str, ...] = ("B", "F", "C", "R")
BALANCED = "r1"
IMBALANCED = "r100"
SEED_BASE = 48000
EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
_ALLOCATIONS = {
    "B": (BALANCED, BALANCED),
    "F": (BALANCED, IMBALANCED),
    "C": (IMBALANCED, BALANCED),
    "R": (IMBALANCED, IMBALANCED),
}
_RECIPES = {
    "bracs": {"batch_size": 256, "steps": 600, "lr": 0.0001, "weight_decay": 0.0001},
    "tcga_ut": {"batch_size": 256, "steps": 2400, "lr": 0.0001, "weight_decay": 0.0001},
}
_PATIENTS = {"bracs": 10, "tcga_ut": 20}
TRAIN_ROWS = {"bracs": 2240, "tcga_ut": 19200}
_LORA = {"r": 8, "alpha": 16, "dropout": 0.0}
_DROPPED_MITIGATION_KEYS = ("grid", "inline_stage2", "gcl", "cuda")


def arm_allocations(arm: str) -> tuple[str, str]:
    """(data allocation, loss-target allocation) of one arm."""
    if arm not in _ALLOCATIONS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    return _ALLOCATIONS[arm]


def loss_weights(data_counts: np.ndarray, target_counts: np.ndarray) -> np.ndarray:
    """Per-class row weight ``q_c / (n_c / N)``; the dataset-average weight is one."""
    data = np.asarray(data_counts, dtype=np.float64)
    target = np.asarray(target_counts, dtype=np.float64)
    if data.shape != target.shape or data.ndim != 1:
        raise ValueError(f"count shapes differ: {data.shape} vs {target.shape}")
    if np.any(data <= 0) or np.any(target <= 0):
        raise ValueError("every class needs a positive data and target count")
    return (target / target.sum()) / (data / data.sum())


def block_seed(split_idx: int, draw_idx: int) -> int:
    """Initialization/sampler seed shared by the four arms of one (split, draw) block."""
    if split_idx not in range(N_SPLITS) or draw_idx not in range(N_DRAWS):
        raise ValueError(f"no block ({split_idx}, {draw_idx})")
    return SEED_BASE + 100 * split_idx + draw_idx


def is_smoke(config: dict[str, Any]) -> bool:
    """A smoke config: shortened budget, isolated root, never a scientific run."""
    return "smoke" in config


def load_protocol_config(path: str | Path) -> dict[str, Any]:
    """Source dataset/training settings plus exp-48 outputs, protocol, and SLURM overrides."""
    own = load_config(path)
    source = load_config(REPO_ROOT / own["source_config"])
    config = copy.deepcopy(source)
    for key in _DROPPED_MITIGATION_KEYS:
        config["mitigation"].pop(key, None)
    config["mitigation"]["train"] = {"stage1": config["mitigation"]["train"]["stage1"]}
    config.pop("analyze", None)
    config["paths"] = {"outputs": own["paths"]["outputs"]}
    config["slurm"].update(own["slurm"])
    config["source_config"] = own["source_config"]
    config["config_path"] = str(Path(path).resolve())
    config["protocol"] = own["protocol"]
    if "smoke" in own:
        config["smoke"] = own["smoke"]
        config["mitigation"]["train"]["stage1"]["steps"] = int(own["smoke"]["steps"])
    validate_protocol(config)
    return config


def _check_recipe(config: dict[str, Any], dataset: str) -> None:
    mitigation = config["mitigation"]
    expected = dict(_RECIPES[dataset])
    if is_smoke(config):
        expected["steps"] = int(config["smoke"]["steps"])
    stage1 = mitigation["train"]["stage1"]
    if {k: float(stage1.get(k, -1)) for k in expected} != {
        k: float(v) for k, v in expected.items()
    }:
        raise ValueError(f"stage-one recipe differs from the protocol: {stage1}")
    lora = mitigation.get("lora", {})
    if {k: float(lora.get(k, -1)) for k in _LORA} != _LORA or "target_regex" in lora:
        raise ValueError(f"LoRA recipe differs from the protocol: {lora}")
    if mitigation.get("encoder", "virchow2") != "virchow2":
        raise ValueError(f"unsupported encoder {mitigation['encoder']!r}")


def _check_outputs(config: dict[str, Any]) -> None:
    outputs = (REPO_ROOT / config["paths"]["outputs"]).resolve()
    allowed = EXPERIMENT_DIR / "outputs"
    if is_smoke(config):
        allowed = allowed / "smoke"
    if not outputs.is_relative_to(allowed):
        raise ValueError(f"output root {outputs} lies outside {allowed}")


def validate_protocol(config: dict[str, Any]) -> None:
    """Reject any deviation from the locked design: id, allocations, recipe, encoder, outputs."""
    protocol = config["protocol"]
    if protocol.get("id") != PROTOCOL_ID:
        raise ValueError(f"protocol id {protocol.get('id')!r}; need {PROTOCOL_ID!r}")
    allocations = (
        protocol.get("balanced_allocation"),
        protocol.get("imbalanced_allocation"),
    )
    if allocations != (BALANCED, IMBALANCED):
        raise ValueError(
            f"allocations must be {(BALANCED, IMBALANCED)}, got {allocations}"
        )
    dataset = config["dataset"]["name"]
    if dataset not in _RECIPES:
        raise ValueError(f"unsupported dataset {dataset!r}")
    if int(config["prevalence"]["patients_per_class"]) != _PATIENTS[dataset]:
        raise ValueError("patients per class differs from the mitigation allocation")
    _check_recipe(config, dataset)
    _check_outputs(config)


def workload_counts(config: dict[str, Any]) -> dict[str, int]:
    """Matched blocks and fresh fits of one dataset."""
    validate_protocol(config)
    blocks = N_SPLITS * N_DRAWS
    return {"blocks": blocks, "fits": blocks * len(ARMS)}


def digest(payload: object) -> str:
    """SHA-256 of a canonical JSON rendering."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def config_digest(config: dict[str, Any]) -> str:
    """Fingerprint of everything that defines the fits: dataset, allocation, recipe, protocol."""
    keys = ("dataset", "mitigation", "prevalence", "protocol", "smoke")
    return digest([config.get(key) for key in keys])
