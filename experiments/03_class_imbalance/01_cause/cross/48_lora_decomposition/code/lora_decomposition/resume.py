"""Completion and compatibility of exp-48 runs: light sidecar marker and full validation.

``is_marked`` reads only the sidecar and is safe on the login node. ``validate_run`` also
recomputes the arm's allocation from the current manifest, compares the source and input digests,
and checks the stored test arrays, so allocation drift or reordered predictions raise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.common import EVAL_ARRAYS_NAME, RUN_RECORD_NAME

from analyze.paths import result_paths, run_dir

from mitigation.data import ImageShard, load_shard, train_arm
from mitigation.train import ArmBatch

from prevalence import patients_per_class

from lora_decomposition import protocol
from lora_decomposition.protocol import (
    PROTOCOL_ID,
    arm_allocations,
    block_seed,
    config_digest,
)
from lora_decomposition.provenance import (
    PROVENANCE_NAME,
    check_record,
    frame_digest,
    labels_digest,
    sequence_digest,
    source_digest,
)

__all__ = [
    "arm_batch",
    "cached_shard",
    "fit_dir",
    "is_marked",
    "run_identity",
    "training_identity",
    "validate_run",
]

_IDENTITY_CACHE: dict[tuple[str, int], dict[str, Any]] = {}


def fit_dir(config: dict[str, Any], split_idx: int, draw_idx: int, arm: str) -> Path:
    """Standard exp-40 result directory of one arm, ``<arm>_ce/draw_<d>``."""
    return run_dir(result_paths(config)[split_idx], arm, "ce", None, draw_idx)


def cached_shard(
    config: dict[str, Any], split_idx: int
) -> tuple[ImageShard, dict[str, Any]]:
    """One split's shard and its class order and ordered input/evaluation digests (cached)."""
    key = (str(config.get("slurm", {}).get("exp2_outputs")), split_idx)
    if key not in _IDENTITY_CACHE:
        shard = load_shard(config, split_idx)
        frames = (shard.train_df, shard.val_df, shard.test_df)
        identity = {
            "class_names": list(shard.classes),
            "input_manifest_sha256": sequence_digest([frame_digest(f) for f in frames]),
            "validation_identity_sha256": frame_digest(shard.val_identity),
            "test_identity_sha256": frame_digest(shard.test_identity),
        }
        _IDENTITY_CACHE[key] = {"shard": shard, "identity": identity}
    entry = _IDENTITY_CACHE[key]
    return entry["shard"], entry["identity"]


def _light_identity(
    config: dict[str, Any], split_idx: int, draw_idx: int, arm: str
) -> dict[str, Any]:
    """Sidecar fields checkable without manifests, arrays, or source parsing."""
    return {
        "protocol_id": PROTOCOL_ID,
        "config_sha256": config_digest(config),
        "arm": arm,
        "split": split_idx,
        "draw": draw_idx,
        "seed": block_seed(split_idx, draw_idx),
    }


def _raise_on_mismatch(
    out_dir: Path, stored: dict[str, Any], expected: dict[str, Any]
) -> None:
    mismatched = sorted(k for k, v in expected.items() if stored.get(k) != v)
    if mismatched:
        raise ValueError(f"completed run at {out_dir} is incompatible: {mismatched}")


def _stored_provenance(out_dir: Path) -> dict[str, Any] | None:
    """The sidecar of a run whose record, arrays, and sidecar all exist, else ``None``."""
    names = (PROVENANCE_NAME, RUN_RECORD_NAME, EVAL_ARRAYS_NAME)
    if not all((out_dir / name).is_file() for name in names):
        return None
    return json.loads((out_dir / PROVENANCE_NAME).read_text(encoding="utf-8"))


def is_marked(config: dict[str, Any], split_idx: int, draw_idx: int, arm: str) -> bool:
    """Light completeness test (sidecar only, login-node safe); raises on a changed config."""
    arm_allocations(arm)
    out_dir = fit_dir(config, split_idx, draw_idx, arm)
    stored = _stored_provenance(out_dir)
    if stored is None:
        return False
    _raise_on_mismatch(
        out_dir, stored, _light_identity(config, split_idx, draw_idx, arm)
    )
    return True


def training_identity(batch: ArmBatch) -> str:
    """Digest of the ordered training patch paths and their labels."""
    return sequence_digest([sequence_digest(batch.image_paths), labels_digest(batch.y)])


def _full_identity(
    config: dict[str, Any], split_idx: int, draw_idx: int, arm: str
) -> dict[str, Any]:
    """Every field a compatible run must match, the allocation recomputed from the manifest."""
    shard, identity = cached_shard(config, split_idx)
    batch, target_counts = arm_batch(config, shard, split_idx, draw_idx, arm)
    return {
        **_light_identity(config, split_idx, draw_idx, arm),
        "source_sha256": source_digest(),
        **identity,
        "training_identity_sha256": training_identity(batch),
        "data_counts": batch.counts.tolist(),
        "target_counts": target_counts.tolist(),
    }


def validate_run(
    config: dict[str, Any], split_idx: int, draw_idx: int, arm: str
) -> bool:
    """False if missing/incomplete; raise if completed but incompatible; True if reusable."""
    if not is_marked(config, split_idx, draw_idx, arm):
        return False
    out_dir = fit_dir(config, split_idx, draw_idx, arm)
    stored = _stored_provenance(out_dir) or {}
    _raise_on_mismatch(
        out_dir, stored, _full_identity(config, split_idx, draw_idx, arm)
    )
    check_record(out_dir, stored)
    return True


def arm_batch(
    config: dict[str, Any], shard: ImageShard, split_idx: int, draw_idx: int, arm: str
) -> tuple[ArmBatch, np.ndarray]:
    """The arm's training rows and the realized counts of its loss-target allocation."""
    data_arm, target_arm = arm_allocations(arm)
    g = patients_per_class(config)
    paths, y, data_counts = train_arm(shard, split_idx, draw_idx, g, data_arm)
    target_counts = np.asarray(train_arm(shard, split_idx, draw_idx, g, target_arm)[2])
    expected = protocol.TRAIN_ROWS[config["dataset"]["name"]]
    totals = (len(y), int(np.sum(data_counts)), int(target_counts.sum()))
    if totals != (expected,) * 3:
        raise ValueError(f"training budget {totals} differs from {expected} rows")
    batch = ArmBatch(paths, y, np.asarray(data_counts), len(shard.classes))
    return batch, target_counts


def run_identity(
    config: dict[str, Any],
    split_idx: int,
    draw_idx: int,
    arm: str,
    shard_identity: dict[str, Any],
) -> dict[str, Any]:
    """Identity fields a new run records before training."""
    return {
        **_light_identity(config, split_idx, draw_idx, arm),
        "source_sha256": source_digest(),
        **shard_identity,
    }
