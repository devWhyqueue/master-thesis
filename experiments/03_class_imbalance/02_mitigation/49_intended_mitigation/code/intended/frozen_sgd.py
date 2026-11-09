"""Frozen-encoder SGD runs of one (split, draw): CE_SGD at r1 and r100, CUDA at r1.

Every run goes through exp-45's ``run_frozen_cuda`` (pinned pretrained Virchow2, frozen, image
forwards under bf16 autocast, fresh linear head), CE with augmentation off, so CUDA's control
shares its code path. Seeds equal exp-45's CUDA r100 runs. The encoder loads once per draw and
each arm's images decode once; a run is complete when its record and sidecar exist.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from imbalance_benchmark.common import RUN_RECORD_NAME

from breadth.sampling import load_features_for_df

from prevalence import patients_per_class

from analyze.paths import run_dir
from mitigation.data import ImageShard, eval_arm, load_shard, resolve_device, train_arm
from mitigation.images import decode_images
from mitigation.train import (
    ArmBatch,
    EvalSplit,
    RunMeta,
    evaluate_and_record,
    head_logits,
)

from lora_ablation import CUDA_MICRO_BATCH, init_seed
from lora_ablation import frozen_cuda
from lora_ablation.heads import encoder_fingerprint, extraction_transform

from intended import ARMS, sgd_paths

logger = logging.getLogger(__name__)

__all__ = ["SIDECAR_NAME", "fit_sgd_draw"]

SIDECAR_NAME = "sgd.json"

Job = tuple[str, str, float | None]


def _pending(config: dict[str, Any], paths: dict[str, Path], draw: int) -> list[Job]:
    jobs = [
        (arm, method, param)
        for arm in ARMS
        for method, params in config["mitigation"]["grid"][arm]["sgd"].items()
        for param in params
    ]
    return [
        job
        for job in jobs
        if not all(
            (run_dir(paths, *job, draw) / name).is_file()
            for name in (RUN_RECORD_NAME, SIDECAR_NAME)
        )
    ]


def _record(
    shard: ImageShard,
    out_dir: Path,
    run: tuple[Job, dict[str, Any], torch.nn.Module],
    features: tuple,
) -> None:
    (arm, method, param), meta, head = run
    evaluate_and_record(
        out_dir,
        RunMeta(meta["config"], method, param, arm, meta["by_class"]),
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


def _fit_arm(
    config: dict[str, Any], context: dict[str, Any], arm: str, jobs: list[Job]
) -> None:
    """Decode one arm's training images once and fit its pending CE/CUDA runs."""
    shard, device, encoder, data_config = (
        context[k] for k in ("shard", "device", "encoder", "data_config")
    )
    split, draw = context["where"]
    image_paths, y, counts = train_arm(
        shard, split, draw, patients_per_class(config), arm
    )
    images = decode_images(image_paths, extraction_transform(data_config)).to(device)
    task = frozen_cuda.CudaTask(
        encoder,
        data_config,
        ArmBatch(image_paths, y, counts, len(shard.classes)),
        images,
    )
    seeds = (draw, init_seed(split, draw))
    meta = {
        "config": config,
        "by_class": dict(zip(shard.classes, (int(c) for c in counts))),
    }
    for job in jobs:
        _, method, param = job
        head = frozen_cuda.run_frozen_cuda(
            config, device, task, float(param or 0.0), seeds, method=method
        )
        out_dir = run_dir(context["paths"], *job, draw)
        _record(shard, out_dir, (job, meta, head), context["features"])
        sidecar = {
            "method": method,
            "seeds": list(seeds),
            "device": device.type,
            "micro_batch": CUDA_MICRO_BATCH,
        }
        logger.info("%s %s %s split %d draw %d done", arm, method, param, split, draw)
        (out_dir / SIDECAR_NAME).write_text(
            json.dumps(sidecar, indent=2), encoding="utf-8"
        )


def _draw_context(config: dict[str, Any], split: int, draw: int) -> dict[str, Any]:
    """Shard, frozen encoder (cache-checked) and cached eval features of one draw."""
    shard = load_shard(config, split)
    device = resolve_device()
    encoder, data_config = frozen_cuda._load_frozen_encoder(device)
    frozen_cuda._check_cache(shard, encoder, data_config, device)
    features = tuple(
        torch.from_numpy(load_features_for_df(df)).to(torch.float16).to(device)
        for df in (shard.val_df, shard.test_df)
    )
    return {
        "shard": shard,
        "device": device,
        "encoder": encoder,
        "data_config": data_config,
        "where": (split, draw),
        "paths": sgd_paths(config)[split],
        "features": features,
    }


def fit_sgd_draw(config: dict[str, Any], split: int, draw: int) -> None:
    """Fit every pending CE_SGD (r1, r100) and CUDA r1 run of one draw with one loaded encoder."""
    pending = _pending(config, sgd_paths(config)[split], draw)
    if not pending:
        return
    context = _draw_context(config, split, draw)
    before = encoder_fingerprint(context["encoder"])
    for arm in ARMS:
        jobs = [job for job in pending if job[0] == arm]
        if jobs:
            _fit_arm(config, context, arm, jobs)
    if encoder_fingerprint(context["encoder"]) != before:
        raise ValueError("frozen encoder changed during SGD training")
