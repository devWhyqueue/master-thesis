"""Shard -> arm rows -> stage-one run -> run record for the mitigation-methods CLI.

Trains a fresh LoRA-adapted encoder and head per (arm, method, param, draw)
and exports its embeddings for the two stage-two methods it feeds
(``STAGE2_SOURCE``); ``mitigation.fit_stage2`` fits those from the cache.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from imbalance_benchmark.common import RUN_RECORD_NAME

from centre.fit import decode_shard_index, shard_count

from prevalence import patients_per_class

from mitigation import STAGE1_METHODS, STAGE2_SOURCE
from mitigation.data import (
    ImageShard,
    eval_arm,
    load_shard,
    resolve_device,
    run_dir,
    train_arm,
)
from mitigation.images import decode_transform
from mitigation.train import (
    STAGE1_ARTIFACT_NAME,
    ArmBatch,
    EvalSplit,
    RunMeta,
    Stage1Artifacts,
    Stage1Output,
    evaluate_and_record,
    forward_embeddings,
    head_logits,
    run_stage1,
    save_stage1,
    stream_embeddings,
)

__all__ = ["decode_shard_index", "shard_count", "run_fit_stage1"]

_STAGE2_SOURCES = frozenset(STAGE2_SOURCE.values())


class _EvalEmbeddings(NamedTuple):
    """Pooled validation/test embeddings and their integer targets for one arm."""

    val_embeddings: torch.Tensor
    val_y: np.ndarray
    test_embeddings: torch.Tensor
    test_y: np.ndarray


def _eval_embeddings(
    output: Stage1Output, shard: ImageShard, device: torch.device
) -> _EvalEmbeddings:
    """Decode and pool the arm's validation/test images through the trained stage-one encoder.

    Streamed in bounded-memory chunks (``stream_embeddings``): the locked
    validation/test partitions are the whole dataset's split (TCGA-UT:
    ~243k/238k patches), not just this arm's rows, so decoding either as one
    uint8 tensor would need tens of GB of host RAM.
    """
    transform = decode_transform(output.data_config)
    mean, std = tuple(output.data_config["mean"]), tuple(output.data_config["std"])
    val_paths, val_y = eval_arm(shard.val_df, shard.classes)
    test_paths, test_y = eval_arm(shard.test_df, shard.classes)
    val_embeddings = stream_embeddings(
        output.encoder, val_paths, transform, mean, std, device
    )
    test_embeddings = stream_embeddings(
        output.encoder, test_paths, transform, mean, std, device
    )
    return _EvalEmbeddings(val_embeddings, val_y, test_embeddings, test_y)


def _save_stage1_artifacts(
    out_dir: Path,
    output: Stage1Output,
    arm_batch: ArmBatch,
    method: str,
    param: float | None,
    device: torch.device,
    evals: _EvalEmbeddings,
) -> None:
    """Export train/val/test embeddings for the stage-two methods this arm feeds."""
    mean, std = tuple(output.data_config["mean"]), tuple(output.data_config["std"])
    train_embeddings = forward_embeddings(
        output.encoder, output.images, mean, std, device
    )
    artifacts = Stage1Artifacts(
        train_embeddings,
        arm_batch.y,
        evals.val_embeddings,
        evals.val_y,
        evals.test_embeddings,
        evals.test_y,
        arm_batch.counts,
        output.head.state_dict(),
        method == "gcl",
        output.cosine_scale,
        param if method == "gcl" else None,
    )
    save_stage1(out_dir / STAGE1_ARTIFACT_NAME, artifacts)


def _record_stage1_run(
    out_dir: Path,
    meta: RunMeta,
    shard: ImageShard,
    output: Stage1Output,
    evals: _EvalEmbeddings,
    device: torch.device,
) -> None:
    """Score the trained head on validation/test and write the run record."""
    val_logits = head_logits(
        output.head, evals.val_embeddings.to(device), output.cosine_scale
    )
    test_logits = head_logits(
        output.head, evals.test_embeddings.to(device), output.cosine_scale
    )
    evaluate_and_record(
        out_dir,
        meta,
        EvalSplit(val_logits, evals.val_y, shard.val_identity),
        EvalSplit(test_logits, evals.test_y, shard.test_identity),
    )


def run_fit_stage1(
    config: dict[str, Any],
    split_idx: int,
    draw_idx: int,
    arm: str,
    method: str,
    param: float | None,
) -> None:
    """Train and record one stage-one arm, unless its run record already exists."""
    if method not in STAGE1_METHODS:
        raise ValueError(f"not a stage-one method: {method}")
    shard = load_shard(config, split_idx)
    out_dir = run_dir(shard, arm, method, param, draw_idx)
    if (out_dir / RUN_RECORD_NAME).exists():
        return
    device = resolve_device()
    image_paths, y, counts = train_arm(
        shard, split_idx, draw_idx, patients_per_class(config), arm
    )
    arm_batch = ArmBatch(image_paths, y, counts, len(shard.classes))
    output = run_stage1(config, device, arm_batch, method, param or 0.0, draw_idx)
    evals = _eval_embeddings(output, shard, device)

    class_counts = dict(zip(shard.classes, (int(c) for c in counts)))
    meta = RunMeta(config, method, param, arm, class_counts)
    _record_stage1_run(out_dir, meta, shard, output, evals, device)
    if method in _STAGE2_SOURCES:
        _save_stage1_artifacts(out_dir, output, arm_batch, method, param, device, evals)
