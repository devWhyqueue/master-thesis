"""One seeded exp-48 fit: arm rows, fixed CE weights, fresh LoRA + head, record, provenance.

Every arm starts from the pretrained encoder with the same block seed (Python, NumPy, Torch CPU
and CUDA are reseeded right before training), so the four arms of a block share their adapter and
head initialization; arms on the same allocation also share their sampled row stream. A run is
complete only once ``provenance.json`` exists, written atomically after the record checks pass.
Reproducibility is promised only for fixed software and hardware, not bitwise across devices.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mitigation.data import ImageShard, resolve_device
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

from lora_decomposition.protocol import loss_weights, validate_protocol
from lora_decomposition.provenance import (
    PROVENANCE_NAME,
    check_record,
    encoder_identity,
    labels_digest,
    software_identity,
    write_atomic_json,
)
from lora_decomposition.resume import (
    arm_batch,
    cached_shard,
    fit_dir,
    is_marked,
    run_identity,
    training_identity,
    validate_run,
)

__all__ = [
    "PROVENANCE_NAME",
    "fit_dir",
    "is_marked",
    "run_fit",
    "seed_everything",
    "validate_run",
]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, Torch CPU and every CUDA device."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    return {
        **identity,
        "training_identity_sha256": training_identity(batch),
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
    shard, shard_identity = cached_shard(config, split_idx)
    identity = run_identity(config, split_idx, draw_idx, arm, shard_identity)
    batch, target_counts = arm_batch(config, shard, split_idx, draw_idx, arm)
    output, audit = _train(config, batch, target_counts, identity["seed"])
    test_y = _record(config, shard, out_dir, arm, output, batch)
    provenance = _provenance(identity, config, batch, target_counts, test_y, audit)
    check_record(out_dir, provenance)
    write_atomic_json(out_dir / PROVENANCE_NAME, provenance)
    return out_dir
