"""One seeded exp-48 fit: arm rows, fixed CE weights, fresh LoRA + head, record, provenance.

Every arm starts from the pretrained encoder with the same block seed (Python, NumPy, Torch CPU
and CUDA are reseeded right before training), so the four arms of a block share their adapter and
head initialization; arms on the same allocation also share their sampled row stream. A run is
complete only once ``provenance.json`` exists, written atomically after the record checks pass.
Reproducibility is promised only for fixed software and hardware, not bitwise across devices.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from imbalance_benchmark.common import EVAL_ARRAYS_NAME, RUN_RECORD_NAME

from analyze.paths import result_paths, run_dir

from mitigation.data import ImageShard, load_shard, resolve_device, train_arm
from mitigation.fit import _eval_embeddings
from mitigation.train import (
    ArmBatch,
    EvalSplit,
    RunMeta,
    Stage1Extras,
    Stage1Output,
    evaluate_and_record,
    head_logits,
    run_stage1,
)

from prevalence import patients_per_class

from lora_decomposition import protocol
from lora_decomposition.protocol import (
    PROTOCOL_ID,
    arm_allocations,
    block_seed,
    config_digest,
    loss_weights,
    validate_protocol,
)
from lora_decomposition.provenance import (
    PROVENANCE_NAME,
    check_record,
    encoder_identity,
    frame_digest,
    labels_digest,
    sequence_digest,
    software_identity,
    source_digest,
    write_atomic_json,
)

__all__ = ["PROVENANCE_NAME", "fit_dir", "run_fit", "seed_everything", "validate_run"]

_IDENTITY_CACHE: dict[tuple[str, int], dict[str, Any]] = {}


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, Torch CPU and every CUDA device."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit_dir(config: dict[str, Any], split_idx: int, draw_idx: int, arm: str) -> Path:
    """Standard exp-40 result directory of one arm, ``<arm>_ce/draw_<d>``."""
    return run_dir(result_paths(config)[split_idx], arm, "ce", None, draw_idx)


def _shard_identity(
    config: dict[str, Any], split_idx: int, shard: ImageShard | None = None
) -> dict[str, Any]:
    """Class order and ordered input/validation/test digests of one split (cached)."""
    key = (str(config.get("slurm", {}).get("exp2_outputs")), split_idx)
    if key not in _IDENTITY_CACHE:
        shard = shard if shard is not None else load_shard(config, split_idx)
        frames = (shard.train_df, shard.val_df, shard.test_df)
        _IDENTITY_CACHE[key] = {
            "class_names": list(shard.classes),
            "input_manifest_sha256": sequence_digest([frame_digest(f) for f in frames]),
            "validation_identity_sha256": frame_digest(shard.val_identity),
            "test_identity_sha256": frame_digest(shard.test_identity),
        }
    return _IDENTITY_CACHE[key]


def _expected_identity(
    config: dict[str, Any],
    split_idx: int,
    draw_idx: int,
    arm: str,
    shard: ImageShard | None = None,
) -> dict[str, Any]:
    """Fields a compatible completed run's provenance must carry exactly."""
    return {
        "protocol_id": PROTOCOL_ID,
        "config_sha256": config_digest(config),
        "source_sha256": source_digest(),
        "arm": arm,
        "split": split_idx,
        "draw": draw_idx,
        "seed": block_seed(split_idx, draw_idx),
        **_shard_identity(config, split_idx, shard),
    }


def validate_run(
    config: dict[str, Any], split_idx: int, draw_idx: int, arm: str
) -> bool:
    """False if missing/incomplete; raise if completed but incompatible; True if reusable."""
    arm_allocations(arm)
    out_dir = fit_dir(config, split_idx, draw_idx, arm)
    names = (PROVENANCE_NAME, RUN_RECORD_NAME, EVAL_ARRAYS_NAME)
    if not all((out_dir / name).is_file() for name in names):
        return False
    stored = json.loads((out_dir / PROVENANCE_NAME).read_text(encoding="utf-8"))
    expected = _expected_identity(config, split_idx, draw_idx, arm)
    mismatched = sorted(k for k, v in expected.items() if stored.get(k) != v)
    if mismatched:
        raise ValueError(f"completed run at {out_dir} is incompatible: {mismatched}")
    check_record(out_dir, stored)
    return True


def _arm_batch(
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


def _record(
    config: dict[str, Any],
    shard: ImageShard,
    out_dir: Path,
    arm: str,
    output: Stage1Output,
    batch: ArmBatch,
) -> np.ndarray:
    """Score validation/test, refuse nonfinite logits, write the standard record; test labels."""
    device = resolve_device()
    evals = _eval_embeddings(output, shard, device)
    val_logits = head_logits(output.head, evals.val_embeddings.to(device), None)
    test_logits = head_logits(output.head, evals.test_embeddings.to(device), None)
    if not (np.isfinite(val_logits).all() and np.isfinite(test_logits).all()):
        raise FloatingPointError(f"nonfinite evaluation logits for {out_dir}")
    counts = dict(zip(shard.classes, (int(c) for c in batch.counts)))
    evaluate_and_record(
        out_dir,
        RunMeta(config, "ce", None, arm, counts),
        EvalSplit(val_logits, evals.val_y, shard.val_identity),
        EvalSplit(test_logits, evals.test_y, shard.test_identity),
    )
    return evals.test_y


def _provenance(
    identity: dict[str, Any],
    config: dict[str, Any],
    batch: ArmBatch,
    target_counts: np.ndarray,
    test_y: np.ndarray,
    audit: dict[str, Any],
) -> dict[str, Any]:
    """Completion marker: identity, allocation and weights, ordered digests, environment, audit."""
    paths_digest = sequence_digest(batch.image_paths)
    return {
        **identity,
        "training_identity_sha256": sequence_digest(
            [paths_digest, labels_digest(batch.y)]
        ),
        "data_counts": batch.counts.tolist(),
        "target_counts": target_counts.tolist(),
        "target_shares": (target_counts / target_counts.sum()).tolist(),
        "class_weights": loss_weights(batch.counts, target_counts).tolist(),
        "n_train": len(batch.y),
        "n_test": len(test_y),
        "test_labels_sha256": labels_digest(test_y),
        "recipe": config["mitigation"]["train"]["stage1"],
        "encoder": encoder_identity(),
        "software": software_identity(),
        "training_audit": audit,
    }


def _train(
    config: dict[str, Any], batch: ArmBatch, target_counts: np.ndarray, seed: int
) -> tuple[Stage1Output, dict[str, Any]]:
    """Reseed everything, then train a fresh adapter + head with this arm's fixed weights."""
    seed_everything(seed)
    audit: dict[str, Any] = {}
    extras = Stage1Extras(loss_weights(batch.counts, target_counts), audit)
    output = run_stage1(config, resolve_device(), batch, "ce", 0.0, seed, extras=extras)
    return output, audit


def run_fit(config: dict[str, Any], split_idx: int, draw_idx: int, arm: str) -> Path:
    """Train, record, and mark one arm of one block, unless a compatible run exists."""
    validate_protocol(config)
    out_dir = fit_dir(config, split_idx, draw_idx, arm)
    if validate_run(config, split_idx, draw_idx, arm):
        return out_dir
    (out_dir / PROVENANCE_NAME).unlink(missing_ok=True)
    shard = load_shard(config, split_idx)
    identity = _expected_identity(config, split_idx, draw_idx, arm, shard)
    batch, target_counts = _arm_batch(config, shard, split_idx, draw_idx, arm)
    output, audit = _train(config, batch, target_counts, identity["seed"])
    test_y = _record(config, shard, out_dir, arm, output, batch)
    provenance = _provenance(identity, config, batch, target_counts, test_y, audit)
    check_record(out_dir, provenance)
    write_atomic_json(out_dir / PROVENANCE_NAME, provenance)
    return out_dir
