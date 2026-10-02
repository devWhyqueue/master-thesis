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

__all__ = ["run_fit_stage2", "run_stage2_from_source"]


def _source_run_param(
    config: dict[str, Any], arm: str, method: str, param: float | None
) -> float | None:
    """The stage-one source run's param for one stage-two job.

    For a ``ce``-sourced method the source is always paramless. For ``gcl2``
    the job's own param *is* its source ``gcl`` run's sigma (one gcl2 job per
    configured sigma); validated against the arm's configured ``gcl`` grid
    rather than assumed, so a stale grid can't silently source the wrong run.
    """
    source_method = STAGE2_SOURCE[method]
    if source_method == "ce":
        return None
    raw_values = (
        config.get("mitigation", {})
        .get("grid", {})
        .get(arm, {})
        .get("stage1", {})
        .get(source_method, [])
    )
    values = [None if v is None else float(v) for v in raw_values]
    if param not in values:
        raise ValueError(
            f"{method} param {param} not among configured {arm} {source_method} params {values}"
        )
    return param


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
        shard,
        arm,
        STAGE2_SOURCE[method],
        _source_run_param(config, arm, method, param),
        seed,
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


def run_stage2_from_source(
    config: dict[str, Any],
    shard: ImageShard,
    device: torch.device,
    arm: str,
    source: tuple[str, float | None],
    draw_idx: int,
) -> None:
    """Fit every pending stage-two job of ``arm`` sourced from one stage-one (method, param) run."""
    for job_arm, method, param in stage2_jobs(config):
        if (
            job_arm != arm
            or (STAGE2_SOURCE[method], _source_run_param(config, arm, method, param))
            != source
        ):
            continue
        out_dir = run_dir(shard, arm, method, param, draw_idx)
        if not (out_dir / RUN_RECORD_NAME).exists():
            _fit_one(config, device, shard, (arm, method, param), out_dir, draw_idx)


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
