from __future__ import annotations

from typing import Any, cast
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from imbalance_benchmark.modeling.models import AttentionMil, DualExpertMil


def per_class_recall(
    preds: np.ndarray, targets: np.ndarray, n_classes: int
) -> list[float]:
    """Compute per-class recall from prediction and target arrays."""
    return [
        float((preds[targets == c] == c).sum()) / max(1, int((targets == c).sum()))
        for c in range(n_classes)
    ]


def _gather_and_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_mil: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather every batch's logits and targets over one evaluation pass."""
    was_training = model.training
    model.eval()
    all_logits, all_targets = [], []
    with torch.inference_mode():
        for batch in loader:
            if is_mil:
                bags, targets = batch
                logits = (
                    model.forward_ensemble(
                        [b.to(device, non_blocking=True) for b in bags]
                    )
                    if isinstance(model, DualExpertMil)
                    else cast(AttentionMil, model).forward_bags(
                        [b.to(device, non_blocking=True) for b in bags]
                    )[0]
                )
            else:
                logits = model(batch["features"].to(device, non_blocking=True))
                targets = batch["target"]
            all_logits.append(logits)
            all_targets.append(targets)
    model.train(was_training)
    logits = torch.cat(all_logits, dim=0).cpu()
    targets = torch.cat(all_targets, dim=0).long()
    return logits, targets


def _compute_metrics(
    preds: torch.Tensor, targets: torch.Tensor, logits: torch.Tensor, n_classes: int
) -> dict[str, float]:
    recalls = [
        float((preds[targets == c] == c).sum().item())
        / max(1, (targets == c).sum().item())
        for c in range(n_classes)
    ]
    f1s = []
    for c in range(n_classes):
        tp = float(((preds == c) & (targets == c)).sum().item())
        fp = float(((preds == c) & (targets != c)).sum().item())
        fn = float(((preds != c) & (targets == c)).sum().item())
        f1s.append(2 * tp / max(1.0, 2 * tp + fp + fn))
    return {
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "nll": float(F.cross_entropy(logits, targets).item()),
    }


def run_evaluation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_mil: bool,
    n_classes: int,
    logit_adj_tau: float | None = None,
    class_priors: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Evaluate model and return predictions, probs, and validation metrics."""
    logits, targets = _gather_and_eval(model, loader, device, is_mil)
    if logit_adj_tau is not None and class_priors is not None:
        logits = logits - logit_adj_tau * torch.log(class_priors.cpu() + 1e-8)
    probs = torch.softmax(logits, dim=-1)
    preds = probs.argmax(dim=-1)
    metrics = _compute_metrics(preds, targets, logits, n_classes)
    return {
        **metrics,
        "logits": logits.numpy(),
        "probs": probs.numpy(),
        "preds": preds.numpy(),
        "targets": targets.numpy(),
    }


def checkpoint_step(
    model: nn.Module,
    val_loader: DataLoader | None,
    device: torch.device,
    is_mil: bool,
    n_classes: int,
    best: dict[str, Any],
    step: int = 0,
) -> dict[str, Any]:
    """Evaluate the current model and keep it if it wins the BA -> F1 -> NLL tie-break."""
    if val_loader is None:
        return best
    logits, targets = _gather_and_eval(model, val_loader, device, is_mil)
    preds = logits.softmax(dim=-1).argmax(dim=-1)
    metrics = _compute_metrics(preds, targets, logits, n_classes)
    acc, f1, nll = metrics["balanced_accuracy"], metrics["macro_f1"], metrics["nll"]
    if acc > best["acc"] or (
        abs(acc - best["acc"]) < 1e-6
        and (f1 > best["f1"] or (abs(f1 - best["f1"]) < 1e-6 and nll < best["nll"]))
    ):
        return {
            "state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
            "acc": acc,
            "f1": f1,
            "nll": nll,
            "step": step,
        }
    return best


def initial_checkpoint(
    model: nn.Module,
    val_loader: DataLoader | None,
    device: torch.device,
    is_mil: bool,
    n_classes: int,
) -> dict[str, Any]:
    """Snapshot the untrained model as the initial best-checkpoint baseline."""
    best = {
        "state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
        "acc": -1.0,
        "f1": -1.0,
        "nll": float("inf"),
        "step": 0,
    }
    return checkpoint_step(model, val_loader, device, is_mil, n_classes, best)


class ClassAwareBatchSampler(Sampler[list[int]]):
    """Yield batches with at least two independent bags from every sampled class."""

    def __init__(self, labels: np.ndarray, batch_size: int, seed: int) -> None:
        if batch_size < 2:
            raise ValueError("SC-MIL requires a batch size of at least two")
        self.batch_size = batch_size
        self.seed = seed
        self.class_indices = {
            int(cls): np.flatnonzero(labels == cls)
            for cls in np.unique(labels)
            if int(np.sum(labels == cls)) >= 2
        }
        if not self.class_indices:
            raise ValueError(
                "SC-MIL requires two independent bags in at least one class"
            )
        self.n_batches = math.ceil(len(labels) / batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        classes = np.array(sorted(self.class_indices))
        for _ in range(self.n_batches):
            n_pairs = min(len(classes), self.batch_size // 2)
            selected_classes = rng.choice(classes, size=n_pairs, replace=False)
            batch = [
                int(index)
                for cls in selected_classes
                for index in rng.choice(
                    self.class_indices[int(cls)], size=2, replace=False
                )
            ]
            fill_classes = selected_classes if len(selected_classes) else classes
            while len(batch) < self.batch_size:
                cls = int(rng.choice(fill_classes))
                batch.append(int(rng.choice(self.class_indices[cls])))
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.n_batches


class _RecordingBatchSampler(Sampler[list[int]]):
    """Wrap a batch sampler, recording every yielded index as an actual exposure."""

    def __init__(self, base: Sampler[list[int]], exposed: set[int]) -> None:
        self.base = base
        self.exposed = exposed

    def __iter__(self):
        for batch in self.base:
            self.exposed.update(int(index) for index in batch)
            yield batch

    def __len__(self) -> int:
        return len(cast(Any, self.base))
