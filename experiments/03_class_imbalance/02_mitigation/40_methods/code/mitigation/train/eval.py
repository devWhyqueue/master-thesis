"""Eval-time forward passes, endpoint computation, and run-record writing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms as tv_transforms

from imbalance_benchmark.analysis.calibration import fit_temperature, softmax
from imbalance_benchmark.analysis.reporting.clustered_endpoints import (
    clustered_endpoints,
)
from imbalance_benchmark.common import write_run_record

from mitigation.encoder import pool_tokens
from mitigation.images import decode_images, normalize_batch

__all__ = [
    "RunMeta",
    "EvalSplit",
    "forward_embeddings",
    "stream_embeddings",
    "head_logits",
    "evaluate_and_record",
]


@dataclass
class RunMeta:
    """One run's identity: which config, method/param/arm, and training-set counts."""

    config: dict[str, Any]
    method: str
    param: float | None
    arm: str
    class_counts: dict[str, int]


@dataclass
class EvalSplit:
    """One split's scored logits, integer targets, and patient/slide identity."""

    logits: np.ndarray
    y: np.ndarray
    identity: pd.DataFrame


@torch.inference_mode()
def forward_embeddings(
    encoder: nn.Module,
    images: torch.Tensor,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    """Pool every image through the encoder in eval batches; returns fp16 embeddings on CPU."""
    encoder.eval()
    rows = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(device, non_blocking=True)
        x = normalize_batch(batch, mean, std)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            features = pool_tokens(encoder(x))
        rows.append(features.float().cpu().to(torch.float16))
    encoder.train()
    if not rows:
        raise ValueError("forward_embeddings needs at least one image")
    return torch.cat(rows, dim=0)


def stream_embeddings(
    encoder: nn.Module,
    image_paths: list[str],
    transform: tv_transforms.Compose,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    device: torch.device,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Decode and pool a (possibly huge) image list in bounded-memory chunks.

    A locked validation/test partition can be hundreds of thousands of patches
    (TCGA-UT: ~243k/238k); decoding it as one uint8 tensor needs tens of GB of
    host RAM. Chunking keeps memory to one ``chunk_size`` batch at a time.
    """
    chunks = [
        forward_embeddings(
            encoder,
            decode_images(image_paths[start : start + chunk_size], transform).to(
                device
            ),
            mean,
            std,
            device,
        )
        for start in range(0, len(image_paths), chunk_size)
    ]
    if not chunks:
        raise ValueError("stream_embeddings needs at least one image")
    return torch.cat(chunks, dim=0)


@torch.inference_mode()
def head_logits(
    head: nn.Module, embeddings: torch.Tensor, cosine_scale: float | None
) -> np.ndarray:
    """Run a trained head over embeddings; scales a cosine head's output by ``cosine_scale``."""
    out = head(embeddings.float())
    if cosine_scale is not None:
        out = cosine_scale * out
    return out.cpu().numpy().astype(np.float64)


def _endpoints_and_temperature(
    val: EvalSplit, test: EvalSplit
) -> tuple[dict[str, float], dict[str, float], float]:
    """Validation/test clustered endpoints and the validation-fitted temperature."""
    val_end = clustered_endpoints(
        val.y,
        val.logits.argmax(axis=1),
        softmax(val.logits),
        val.identity,
        is_mil=False,
    )
    test_end = clustered_endpoints(
        test.y,
        test.logits.argmax(axis=1),
        softmax(test.logits),
        test.identity,
        is_mil=False,
    )
    temperature = fit_temperature(val.logits, val.y).temperature
    return val_end, test_end, temperature


def evaluate_and_record(
    out_dir: Path, meta: RunMeta, val: EvalSplit, test: EvalSplit
) -> None:
    """Score val/test logits, fit a validation temperature, and write the run record."""
    val_end, test_end, temperature = _endpoints_and_temperature(val, test)
    record = {
        "dataset": meta.config.get("dataset", {}),
        "method": meta.method,
        "param": meta.param,
        "arm": meta.arm,
        "class_counts": meta.class_counts,
        "validation_temperature": temperature,
        "splits": {
            "validation": {"endpoints": val_end},
            "test": {
                "endpoints": test_end,
                "labels": test.y,
                "preds": test.logits.argmax(axis=1),
                "probabilities": softmax(test.logits),
                "logits": test.logits,
            },
        },
    }
    write_run_record(out_dir, record, keep_arrays=True)
