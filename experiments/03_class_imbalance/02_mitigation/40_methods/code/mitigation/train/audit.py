"""Opt-in stage-one training audit: initial trainable state, consumed rows, and loss health.

Used by matched-arm experiments to prove that two fits started from the same adapter/head
weights and drew the same row-index stream. Ordinary training never builds these objects.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn

__all__ = ["LOSS_WINDOW", "StepAudit", "audited_step", "trainable_digest"]

LOSS_WINDOW = 10


def trainable_digest(encoder: nn.Module, head: nn.Module) -> str:
    """SHA-256 over the names and float32 values of every trainable encoder/head tensor."""
    digest = hashlib.sha256()
    named = [
        (f"encoder.{n}", p) for n, p in encoder.named_parameters() if p.requires_grad
    ] + [(f"head.{n}", p) for n, p in head.named_parameters()]
    for name, parameter in named:
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class StepAudit:
    """Wrap a step closure; hash consumed indices, mark visited rows, and keep update loss means.

    Indices are hashed as consecutive little-endian int64 values, so the digest does not depend
    on how an update is split into micro-batches. A nonfinite loss raises immediately, before
    any caller can write a completed run record.
    """

    def __init__(
        self,
        step_fn: Callable[[torch.Tensor], torch.Tensor],
        batch_size: int,
        audit: dict[str, object],
    ) -> None:
        self._step_fn = step_fn
        self._audit = audit
        self._batch_size = batch_size
        self._visited: set[int] = set()
        self._digest = hashlib.sha256()
        self._rows_seen = 0
        self._update_sum = 0.0
        self.update_means: list[float] = []

    def __call__(self, idx: torch.Tensor) -> torch.Tensor:
        """Run the wrapped step on one (micro-)batch and record what it consumed."""
        rows = idx.detach().cpu().numpy().astype("<i8")
        self._digest.update(rows.tobytes())
        self._visited.update(rows.tolist())
        loss = self._step_fn(idx)
        value = float(loss.detach().float().item())
        if not math.isfinite(value):
            raise FloatingPointError(
                f"nonfinite training loss {value} after {self._rows_seen} rows"
            )
        self._rows_seen += len(rows)
        self._update_sum += value * len(rows)
        if self._rows_seen % self._batch_size == 0:
            self.update_means.append(self._update_sum / self._batch_size)
            self._update_sum = 0.0
        return loss

    def finish(self, y: np.ndarray, num_classes: int) -> None:
        """Fill the audit: sequence digest, sample count, per-class unique rows, loss means."""
        window = min(LOSS_WINDOW, len(self.update_means))
        visited = np.fromiter(sorted(self._visited), dtype=np.int64)
        self._audit.update(
            {
                "sample_sequence_sha256": self._digest.hexdigest(),
                "sample_count": self._rows_seen,
                "unique_rows_per_class": np.bincount(
                    y[visited], minlength=num_classes
                ).tolist(),
                "loss_finite": True,
                "first_loss_mean": float(np.mean(self.update_means[:window])),
                "last_loss_mean": float(np.mean(self.update_means[-window:])),
            }
        )


def audited_step(
    step_fn: Callable[[torch.Tensor], torch.Tensor],
    encoder: nn.Module,
    head: nn.Module,
    batch_size: int,
    audit: dict[str, object] | None,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], StepAudit | None]:
    """Record the initial trainable-state digest and wrap the step; unchanged without audit."""
    if audit is None:
        return step_fn, None
    audit["initialization_sha256"] = trainable_digest(encoder, head)
    recorder = StepAudit(step_fn, batch_size, audit)
    return recorder, recorder
