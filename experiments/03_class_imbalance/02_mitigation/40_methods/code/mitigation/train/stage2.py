"""Stage-two, cached-embedding methods: posthoc LA, cRT, DisAlign, GCL-2.

Every method here starts from one stage-one run's exported embeddings
(``mitigation.train.artifacts.Stage1Artifacts``): the encoder and its LoRA
update are never re-run, matching the report's frozen-representation claim.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from imbalance_benchmark.analysis.calibration import balanced_decision_logits

from mitigation import gcl_constants, train_hparams
from mitigation.methods import (
    CosineHead,
    DisAlign,
    disalign_class_weights,
    gcl_delta,
    gcl_loss,
    reinit_linear_head,
    sample_weights,
)
from mitigation.train.artifacts import Stage1Artifacts
from mitigation.train.eval import head_logits
from mitigation.train.loop import run_training_steps

__all__ = ["run_posthoc_la", "run_crt", "run_disalign", "run_gcl2"]


def _rebuild_linear_head(
    state: dict[str, Any], dim: int, num_classes: int, device: torch.device
) -> nn.Linear:
    """Reconstruct stage one's frozen linear head from its saved state."""
    head = nn.Linear(dim, num_classes).to(device)
    head.load_state_dict(state)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head


def run_posthoc_la(
    artifacts: Stage1Artifacts, tau: float, num_classes: int, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Post-hoc logit adjustment (Eq. la-posthoc): shift the source CE run's logits by tau*log(pi)."""
    dim = artifacts.train_embeddings.shape[1]
    head = _rebuild_linear_head(artifacts.head_state, dim, num_classes, device)
    val_logits = head_logits(head, artifacts.val_embeddings.to(device), None)
    test_logits = head_logits(head, artifacts.test_embeddings.to(device), None)
    pi_train = artifacts.counts / artifacts.counts.sum()
    return (
        balanced_decision_logits(
            val_logits, "post_hoc_logit_adjustment", tau, pi_train
        ),
        balanced_decision_logits(
            test_logits, "post_hoc_logit_adjustment", tau, pi_train
        ),
    )


def _train_embeddings_head(
    weights: np.ndarray,
    head: nn.Module,
    step_fn: Any,
    config: dict[str, Any],
    generator: torch.Generator,
) -> None:
    """Train a small head over fixed embeddings with the shared fixed-step loop."""
    hparams = train_hparams(config, "stage2")
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hparams.steps
    )
    run_training_steps(
        weights,
        hparams.steps,
        hparams.batch_size,
        optimizer,
        scheduler,
        generator,
        step_fn,
    )


def run_crt(
    artifacts: Stage1Artifacts,
    num_classes: int,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[nn.Module, np.ndarray, np.ndarray]:
    """cRT (Eq. crt): reinitialize the classifier and refit it with balanced sampling (s=1)."""
    embeddings = artifacts.train_embeddings.float().to(device)
    y_t = torch.from_numpy(artifacts.train_y).to(device)
    head = nn.Linear(embeddings.shape[1], num_classes).to(device)
    reinit_linear_head(head)
    generator = torch.Generator().manual_seed(seed)
    weights = sample_weights(artifacts.train_y, num_classes, "power", 1.0)

    def step(idx: torch.Tensor) -> torch.Tensor:
        """CE on one balanced index batch of cached embeddings."""
        idx = idx.to(device)
        return F.cross_entropy(head(embeddings[idx]), y_t[idx])

    _train_embeddings_head(weights, head, step, config, generator)
    val_logits = head_logits(head, artifacts.val_embeddings.to(device), None)
    test_logits = head_logits(head, artifacts.test_embeddings.to(device), None)
    return head, val_logits, test_logits


@torch.inference_mode()
def _disalign_eval_logits(
    artifacts: Stage1Artifacts,
    head_o: nn.Linear,
    calibration: DisAlign,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Calibrate the frozen head's validation/test logits with a trained DisAlign module."""
    val_embeddings = artifacts.val_embeddings.float().to(device)
    test_embeddings = artifacts.test_embeddings.float().to(device)
    val_logits = calibration(val_embeddings, head_o(val_embeddings))
    test_logits = calibration(test_embeddings, head_o(test_embeddings))
    return val_logits.cpu().numpy().astype(
        np.float64
    ), test_logits.cpu().numpy().astype(np.float64)


def run_disalign(
    artifacts: Stage1Artifacts,
    rho: float,
    num_classes: int,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """DisAlign (Eq. disalign-calibration/-loss): learn alpha, beta, v on frozen CE logits."""
    embeddings = artifacts.train_embeddings.float().to(device)
    y_t = torch.from_numpy(artifacts.train_y).to(device)
    head_o = _rebuild_linear_head(
        artifacts.head_state, embeddings.shape[1], num_classes, device
    )
    with torch.inference_mode():
        logits_o = head_o(embeddings)
    calibration = DisAlign(embeddings.shape[1], num_classes).to(device)
    class_weight = torch.tensor(
        disalign_class_weights(artifacts.counts, rho),
        dtype=torch.float32,
        device=device,
    )
    generator = torch.Generator().manual_seed(seed)
    weights = sample_weights(artifacts.train_y, num_classes, "power", 0.0)

    def step(idx: torch.Tensor) -> torch.Tensor:
        """Weighted CE on one index batch's DisAlign-calibrated logits."""
        idx = idx.to(device)
        calibrated = calibration(embeddings[idx], logits_o[idx])
        return F.cross_entropy(calibrated, y_t[idx], weight=class_weight)

    _train_embeddings_head(weights, calibration, step, config, generator)
    return _disalign_eval_logits(artifacts, head_o, calibration, device)


def run_gcl2(
    artifacts: Stage1Artifacts,
    num_classes: int,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[nn.Module, np.ndarray, np.ndarray]:
    """GCL stage two (Eq. gcl-cben): re-train a fresh cosine head with the effective-number sampler."""
    if artifacts.sigma is None:
        raise ValueError("gcl2 needs a GCL source run's sigma")
    sigma = artifacts.sigma
    s, a, b = gcl_constants(config)
    embeddings = artifacts.train_embeddings.float().to(device)
    y_t = torch.from_numpy(artifacts.train_y).to(device)
    delta = gcl_delta(artifacts.counts)
    delta_t = torch.tensor(delta, dtype=torch.float32, device=device)
    head = CosineHead(embeddings.shape[1], num_classes).to(device)
    generator = torch.Generator().manual_seed(seed)
    weights = sample_weights(
        artifacts.train_y, num_classes, "gcl", delta=delta, a=a, b=b
    )

    def step(idx: torch.Tensor) -> torch.Tensor:
        """GCL's clouded-logit loss on one effective-number-sampled index batch."""
        idx = idx.to(device)
        return gcl_loss(head(embeddings[idx]), y_t[idx], delta_t, s, sigma, generator)

    _train_embeddings_head(weights, head, step, config, generator)
    val_logits = head_logits(head, artifacts.val_embeddings.to(device), s)
    test_logits = head_logits(head, artifacts.test_embeddings.to(device), s)
    return head, val_logits, test_logits
