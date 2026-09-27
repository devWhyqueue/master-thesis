"""Fit every pending stage-two method of one shard from cached stage-one embeddings.

Each stage-two method (``posthoc_la``, ``crt``, ``disalign``, ``gcl2``) is fit
from the one stage-one run it sources (``STAGE2_SOURCE``); the encoder and its
LoRA update are never re-run, matching ``report/40_methods.tex``'s two-stage
methods (encoder frozen in stage two).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from imbalance_benchmark.common import RUN_RECORD_NAME

from mitigation import STAGE2_SOURCE
from mitigation.data import ImageShard, load_shard, resolve_device, run_dir
from mitigation.grid import stage2_jobs
from mitigation.train import (
    STAGE1_ARTIFACT_NAME,
    EvalSplit,
    RunMeta,
    Stage1Artifacts,
    evaluate_and_record,
    load_stage1,
    run_crt,
    run_disalign,
    run_gcl2,
    run_posthoc_la,
)

__all__ = ["run_fit_stage2"]


def _source_run_param(config: dict[str, Any], method: str) -> float | None:
    """The single configured param of a stage-two method's stage-one source run."""
    source_method = STAGE2_SOURCE[method]
    if source_method == "ce":
        return None
    values = (
        config.get("mitigation", {})
        .get("grid", {})
        .get("stage1", {})
        .get(source_method, [])
    )
    if len(values) != 1:
        raise ValueError(
            f"{method} needs exactly one configured {source_method} param, got {values}"
        )
    return None if values[0] is None else float(values[0])


def _dispatch(
    method: str,
    param: float | None,
    artifacts: Stage1Artifacts,
    num_classes: int,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one stage-two method; return its (val_logits, test_logits) arrays."""
    if method == "posthoc_la":
        return run_posthoc_la(artifacts, param or 0.0, num_classes, device)
    if method == "crt":
        _, val_logits, test_logits = run_crt(
            artifacts, num_classes, config, device, seed
        )
        return val_logits, test_logits
    if method == "disalign":
        return run_disalign(artifacts, param or 0.0, num_classes, config, device, seed)
    _, val_logits, test_logits = run_gcl2(artifacts, num_classes, config, device, seed)
    return val_logits, test_logits


def _fit_one(
    config: dict[str, Any],
    device: torch.device,
    shard: ImageShard,
    job: tuple[str, str, float | None],
    out_dir: Path,
    seed: int,
) -> None:
    """Dispatch one (arm, method, param) stage-two job and write its run record."""
    arm, method, param = job
    source_dir = run_dir(
        shard, arm, STAGE2_SOURCE[method], _source_run_param(config, method), seed
    )
    artifacts = load_stage1(source_dir / STAGE1_ARTIFACT_NAME)
    val_logits, test_logits = _dispatch(
        method, param, artifacts, len(shard.classes), config, device, seed
    )
    class_counts = dict(zip(shard.classes, (int(c) for c in artifacts.counts)))
    meta = RunMeta(config, method, param, arm, class_counts)
    evaluate_and_record(
        out_dir,
        meta,
        EvalSplit(val_logits, artifacts.val_y, shard.val_identity),
        EvalSplit(test_logits, artifacts.test_y, shard.test_identity),
    )


def run_fit_stage2(config: dict[str, Any], split_idx: int, draw_idx: int) -> None:
    """Fit every pending stage-two method of one (split, draw) shard."""
    shard = load_shard(config, split_idx)
    device = resolve_device()
    for job in stage2_jobs(config):
        arm, method, param = job
        out_dir = run_dir(shard, arm, method, param, draw_idx)
        if (out_dir / RUN_RECORD_NAME).exists():
            continue
        _fit_one(config, device, shard, job, out_dir, draw_idx)
