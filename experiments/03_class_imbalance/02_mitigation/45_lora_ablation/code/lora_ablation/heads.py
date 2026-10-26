"""Cached-feature heads: exact ``r100`` training rows and CE / train-time LA fitting.

Training rows come from exp-25's cohort/allocation helpers, the same ones the adapted image
pipeline uses, so patients, patches, order, labels, and class counts match the adapted runs.
CE and LA reuse exp-40's loop, ``la_loss`` and sampler with the source first-stage recipe.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as tv_transforms

from breadth.sampling import load_features_for_df

from mitigation import TrainHParams, train_hparams
from mitigation.data import ImageShard, eval_arm
from mitigation.encoder import pool_tokens
from mitigation.images import normalize_batch
from mitigation.methods import la_loss, sample_weights
from mitigation.train import Stage1Artifacts, head_logits
from mitigation.train.loop import run_training_steps

from prevalence.fit import _shard_context, arm_row_index, class_counts

from lora_ablation import ARM, init_seed

logger = logging.getLogger(__name__)

__all__ = [
    "check_cached_features",
    "direct_features",
    "encoder_fingerprint",
    "extraction_transform",
    "fit_candidate",
    "frozen_artifacts",
    "freeze_encoder",
    "load_head",
    "make_optimizer",
]

COSINE_MIN = 0.999
REL_L2_MAX = 0.01


def frozen_artifacts(
    shard: ImageShard,
    split_idx: int,
    draw_idx: int,
    patients_per_class: int,
    eval_features: tuple[torch.Tensor, torch.Tensor],
) -> Stage1Artifacts:
    """Pretrained fp16 features of one draw's exact ``r100`` training rows, as stage-one input."""
    draw = _shard_context(
        shard.train_df, shard.classes, split_idx, draw_idx, patients_per_class
    )
    counts = class_counts(ARM, draw.perm, draw.available, draw.pool_counts, draw.g)
    rows, y = arm_row_index(draw.train_df, draw.names, draw.patients, counts)
    train = torch.from_numpy(load_features_for_df(draw.train_df.loc[rows]))
    val_x, test_x = eval_features
    return Stage1Artifacts(
        train_embeddings=train.to(torch.float16),
        train_y=y,
        val_embeddings=val_x,
        val_y=eval_arm(shard.val_df, shard.classes)[1],
        test_embeddings=test_x,
        test_y=eval_arm(shard.test_df, shard.classes)[1],
        counts=np.asarray(counts, dtype=np.int64),
        head_state={},
        is_cosine=False,
        cosine_scale=None,
        sigma=None,
    )


def make_optimizer(
    head: nn.Module, hparams: TrainHParams
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """AdamW with the source cosine schedule."""
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hparams.steps
    )
    return optimizer, scheduler


def _log_prior(counts: np.ndarray) -> torch.Tensor:
    """Log of the realized class prior of the training rows."""
    return torch.log(torch.tensor(counts / counts.sum(), dtype=torch.float32))


def _loss_step(
    head: nn.Module,
    features: torch.Tensor,
    y_t: torch.Tensor,
    spec: tuple[str, float | None, torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Index batch -> CE, or logit-adjusted CE with the realized log-prior."""
    method, param, log_prior = spec

    def step(idx: torch.Tensor) -> torch.Tensor:
        """Loss of one index batch."""
        logits = head(features[idx])
        if method == "la":
            return la_loss(logits, y_t[idx], log_prior, float(param or 0.0))
        return F.cross_entropy(logits, y_t[idx])

    return step


def fit_candidate(
    config: dict[str, Any],
    artifacts: Stage1Artifacts,
    method: str,
    param: float | None,
    split_idx: int,
    draw_idx: int,
) -> tuple[nn.Module, np.ndarray, np.ndarray]:
    """One CE or train-time LA head on cached features with the source first-stage recipe."""
    if method not in ("ce", "la"):
        raise ValueError(f"cached fits cover only ce and la, not {method!r}")
    hparams = train_hparams(config, "stage1")
    features = artifacts.train_embeddings.float()
    num_classes = len(artifacts.counts)
    base = int(config["protocol"]["init_seed_base"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed(split_idx, draw_idx, base))
        head = nn.Linear(features.shape[1], num_classes)
        optimizer, scheduler = make_optimizer(head, hparams)
        step = _loss_step(
            head,
            features,
            torch.from_numpy(artifacts.train_y),
            (method, param, _log_prior(artifacts.counts)),
        )
        weights = sample_weights(artifacts.train_y, num_classes, "power", 0.0)
        generator = torch.Generator().manual_seed(draw_idx)
        run_training_steps(weights, hparams, optimizer, scheduler, generator, step)
    return (
        head,
        head_logits(head, artifacts.val_embeddings, None),
        head_logits(head, artifacts.test_embeddings, None),
    )


def load_head(path: Path, dim: int, num_classes: int) -> dict[str, torch.Tensor] | None:
    """A stored CE head if it exists and fits this feature width; incompatible weights count as missing."""
    if not path.is_file():
        return None
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, EOFError, RuntimeError, pickle.UnpicklingError):
        return None  # corrupt file: treated as missing so the head is refit
    shapes = {k: tuple(v.shape) for k, v in state.items()}
    return (
        state
        if shapes == {"weight": (num_classes, dim), "bias": (num_classes,)}
        else None
    )


def check_cached_features(cached: torch.Tensor, direct: torch.Tensor) -> None:
    """Cached features must match direct untouched-encoder forwards row by row, else ``ValueError``."""
    if cached.shape != direct.shape:
        raise ValueError(
            f"cached feature shape {tuple(cached.shape)} != direct {tuple(direct.shape)}"
        )
    a, b = cached.double(), direct.double()
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("cached or direct features are not finite")
    norms_a, norms_b = a.norm(dim=1), b.norm(dim=1)
    if (norms_a == 0).any() or (norms_b == 0).any():
        raise ValueError("zero feature vector in the consistency check")
    cosine = (a * b).sum(dim=1) / (norms_a * norms_b)
    relative = (a - b).norm(dim=1) / norms_b
    if (cosine < COSINE_MIN).any():
        raise ValueError(
            f"cached features differ from direct forwards: min cosine {cosine.min():.5f}"
        )
    if (relative > REL_L2_MAX).any():
        raise ValueError(
            f"cached features differ from direct forwards: max relative L2 {relative.max():.5f}"
        )


def encoder_fingerprint(encoder: nn.Module) -> str:
    """SHA-256 over every parameter and buffer byte, to prove the encoder never changed."""
    digest = hashlib.sha256()
    for name, tensor in (*encoder.named_parameters(), *encoder.named_buffers()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def freeze_encoder(encoder: nn.Module) -> None:
    """Evaluation mode, no gradients; an adapter-wrapped encoder is rejected."""
    if hasattr(encoder, "peft_config") or any(
        "lora" in n for n, _ in encoder.named_parameters()
    ):
        raise ValueError("frozen CUDA must not use a LoRA adapter")
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)


def extraction_transform(data_config: dict[str, Any]) -> tv_transforms.Compose:
    """Uint8 resize/crop of the extraction (timm bicubic); exp-40 resizes bilinearly, 5-9% off the cache."""
    size = data_config["input_size"][-1]
    resize = round(size / (data_config.get("crop_pct") or 1.0))
    return tv_transforms.Compose(
        [
            tv_transforms.Resize(
                resize, interpolation=tv_transforms.InterpolationMode.BICUBIC
            ),
            tv_transforms.CenterCrop(size),
            tv_transforms.PILToTensor(),
        ]
    )


@torch.inference_mode()
def direct_features(
    encoder: nn.Module,
    images: torch.Tensor,
    data_config: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """Untouched-encoder features with the extraction recipe: fp16 autocast, CLS plus mean patch."""
    x = normalize_batch(
        images.to(device), tuple(data_config["mean"]), tuple(data_config["std"])
    )
    with torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
    ):
        features = pool_tokens(encoder(x))
    return features.float().cpu()
