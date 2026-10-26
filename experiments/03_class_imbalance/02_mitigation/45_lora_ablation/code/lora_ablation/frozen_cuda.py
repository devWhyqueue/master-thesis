"""CUDA with a truly frozen foundation model: the curriculum augments images, only the head learns.

The pinned pretrained Virchow2 is loaded directly (no adapter), frozen in evaluation mode, and
still forwards every augmented image; only a fresh linear head is optimized. Exp-40's
``StepModel``/``build_loss_step`` supply the augmentation, per-class levels, classifier checks
and loss, and its fixed-step loop supplies the schedule, so the curriculum is not reimplemented.
Evaluation uses the unchanged pretrained feature caches once they match direct forwards.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, NamedTuple

import torch
import torch.nn as nn
from imbalance_benchmark.datasets.feature_provenance import (
    VIRCHOW2_MODEL,
    VIRCHOW2_REVISION,
    VIRCHOW2_WEIGHTS_SHA256,
)
from imbalance_benchmark.datasets.features import load_feature_model
from timm.data.config import resolve_data_config

from breadth.sampling import load_features_for_df

from mitigation import cuda_constants, train_hparams
from mitigation.data import (
    ImageShard,
    eval_arm,
    load_shard,
    resolve_device,
    run_dir,
    train_arm,
)
from mitigation.grid import stage1_jobs
from mitigation.images import decode_images
from mitigation.methods import sample_weights
from mitigation.train import (
    ArmBatch,
    EvalSplit,
    RunMeta,
    evaluate_and_record,
    head_logits,
)
from mitigation.train.loop import run_training_steps
from mitigation.train.stage1_step import StepModel, build_loss_step

from prevalence import patients_per_class

from lora_ablation import ARM, CUDA_MICRO_BATCH, init_seed
from lora_ablation.heads import (
    check_cached_features,
    direct_features,
    encoder_fingerprint,
    extraction_transform,
    freeze_encoder,
    make_optimizer,
)
from lora_ablation.resume import (
    expected_sidecar,
    is_complete,
    measure,
    require_inventory,
    write_sidecar,
)

logger = logging.getLogger(__name__)

__all__ = [
    "check_cached_features",
    "encoder_fingerprint",
    "fit_cuda_draw",
    "run_frozen_cuda",
]

CHECK_ROWS = 32
_PRECISION = (
    "training forwards: bf16 autocast on cuda, fp32 on cpu; cache check: fp16 autocast (extraction recipe); "
    "fp16 cached eval features; fp32 head"
)


class CudaTask(NamedTuple):
    """The frozen encoder and one draw's resident training images."""

    encoder: nn.Module
    data_config: dict[str, Any]
    arm: ArmBatch
    images: torch.Tensor


def _new_head(pooled_dim: int, num_classes: int, seed: int) -> nn.Linear:
    """A fresh linear head from an isolated CPU RNG context."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return nn.Linear(pooled_dim, num_classes)


def run_frozen_cuda(
    config: dict[str, Any],
    device: torch.device,
    task: CudaTask,
    param: float,
    seeds: tuple[int, int],
) -> nn.Module:
    """Train one CUDA candidate's head; ``seeds`` are (sampling/augmentation, head initialization)."""
    encoder, data_config, arm, images = task
    freeze_encoder(encoder)
    hparams = train_hparams(config, "stage1")._replace(micro_batch=CUDA_MICRO_BATCH)
    steps_per_epoch = -(-int(arm.counts.sum()) // hparams.batch_size)
    cuda_params = (*cuda_constants(config), steps_per_epoch, hparams.batch_size)
    head = _new_head(2 * int(getattr(encoder, "embed_dim")), arm.num_classes, seeds[1])
    head = head.to(device)
    optimizer, scheduler = make_optimizer(head, hparams)
    mean, std = tuple(data_config["mean"]), tuple(data_config["std"])
    y_t = torch.from_numpy(arm.y).to(device)
    step = build_loss_step(
        "cuda",
        param,
        StepModel(encoder, head, images, y_t, mean, std),
        device,
        (None, None, None, cuda_params),
        torch.Generator(device=device).manual_seed(seeds[0]),
    )
    weights = sample_weights(arm.y, arm.num_classes, "power", 0.0)
    generator = torch.Generator().manual_seed(seeds[0])
    run_training_steps(weights, hparams, optimizer, scheduler, generator, step)
    return head


def _load_frozen_encoder(device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """The pinned pretrained Virchow2 from the local cache only (offline), frozen."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    encoder = load_feature_model(VIRCHOW2_MODEL, device)[0]
    data_config = resolve_data_config(encoder.pretrained_cfg, model=encoder)
    freeze_encoder(encoder)
    return encoder, data_config


def _check_cache(
    shard: ImageShard,
    encoder: nn.Module,
    data_config: dict[str, Any],
    device: torch.device,
) -> None:
    """First 32 validation rows: cached features against direct untouched-encoder forwards."""
    rows = shard.val_df.head(CHECK_ROWS)
    images = decode_images(
        rows["image_path"].astype(str).tolist(), extraction_transform(data_config)
    )
    direct = direct_features(encoder, images, data_config, device)
    check_cached_features(torch.from_numpy(load_features_for_df(rows)), direct)


def _settings(device: torch.device) -> dict[str, Any]:
    return {
        "stage": "stage1",
        "device": device.type,
        "precision": _PRECISION,
        "micro_batch": CUDA_MICRO_BATCH,
        "checkpoint": {
            "revision": VIRCHOW2_REVISION,
            "weights_sha256": VIRCHOW2_WEIGHTS_SHA256,
        },
    }


def _record(
    shard: ImageShard,
    out_dir: Path,
    run: tuple[float | None, dict[str, Any], nn.Module],
    features: tuple[torch.Tensor, torch.Tensor],
) -> None:
    param, counts, head = run
    evaluate_and_record(
        out_dir,
        RunMeta(counts["config"], "cuda", param, ARM, counts["by_class"]),
        EvalSplit(
            head_logits(head, features[0], None),
            eval_arm(shard.val_df, shard.classes)[1],
            shard.val_identity,
        ),
        EvalSplit(
            head_logits(head, features[1], None),
            eval_arm(shard.test_df, shard.classes)[1],
            shard.test_identity,
        ),
    )


def _fit_params(
    config: dict[str, Any],
    shard: ImageShard,
    where: tuple[int, int],
    pending: list[float | None],
    expected: dict[str, Any],
) -> None:
    """Load the encoder and images once, then fit every pending strength of this draw."""
    split_idx, draw_idx = where
    device = resolve_device()
    encoder, data_config = _load_frozen_encoder(device)
    before = encoder_fingerprint(encoder)
    _check_cache(shard, encoder, data_config, device)
    paths, y, counts = train_arm(
        shard, split_idx, draw_idx, patients_per_class(config), ARM
    )
    arm = ArmBatch(paths, y, counts, len(shard.classes))
    images = decode_images(paths, extraction_transform(data_config)).to(device)
    task = CudaTask(encoder, data_config, arm, images)
    val_x, test_x = (
        torch.from_numpy(load_features_for_df(df)).to(torch.float16).to(device)
        for df in (shard.val_df, shard.test_df)
    )
    by_class = dict(zip(shard.classes, (int(c) for c in counts)))
    seeds = (
        draw_idx,
        init_seed(split_idx, draw_idx, int(config["protocol"]["init_seed_base"])),
    )
    for param in pending:
        out_dir = run_dir(shard, ARM, "cuda", param, draw_idx)
        with measure() as resources:
            head = run_frozen_cuda(config, device, task, float(param or 0.0), seeds)
            run = (param, {"config": config, "by_class": by_class}, head)
            _record(shard, out_dir, run, (val_x, test_x))
        if device.type == "cuda":
            resources["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2**20
        write_sidecar(out_dir, expected, {**_settings(device), "resources": resources})
        logger.info(
            "cuda %s split %d draw %d: %.1fs", param, *where, resources["seconds"]
        )
    if encoder_fingerprint(encoder) != before:
        raise ValueError("frozen encoder changed during CUDA training")


def fit_cuda_draw(config: dict[str, Any], split_idx: int, draw_idx: int) -> None:
    """Fit all pending CUDA strengths of one draw with one loaded encoder and one decode of its images."""
    fingerprint = require_inventory(config)
    shard = load_shard(config, split_idx)
    expected = expected_sidecar(config, fingerprint, split_idx, draw_idx)
    params = [j.param for j in stage1_jobs(config) if j.method == "cuda"]
    pending = [
        p
        for p in params
        if not is_complete(run_dir(shard, ARM, "cuda", p, draw_idx), expected)
    ]
    if pending:
        _fit_params(config, shard, (split_idx, draw_idx), pending, expected)
