"""Prespecified secondary endpoints: probability quality and class-specific recall."""

from __future__ import annotations

from typing import Any

import numpy as np
from imbalance_benchmark.analysis.inference.context import BootstrapContext
from imbalance_benchmark.analysis.inference.gates import confidence_interval

__all__ = [
    "SECONDARY_SCALARS",
    "build_secondary_results",
    "class_recall_key",
    "draw_secondary_distributions",
    "pack_estimate",
]

SECONDARY_SCALARS: tuple[str, ...] = (
    "macro_nll",
    "expected_calibration_error",
    "patch_micro_balanced_accuracy",
)


def class_recall_key(class_name: str) -> str:
    """Return the endpoint key holding one class's patient-macro recall."""
    return f"recall:{class_name}"


def _patient_macro_recalls(
    ctx: BootstrapContext, labels: np.ndarray, preds: np.ndarray, n_classes: int
) -> np.ndarray:
    """Per-class patient-macro recall, whose class mean is the primary endpoint.

    Uses the same case-class divisor as
    :meth:`~imbalance_benchmark.analysis.inference.context.BootstrapContext.ba_distribution`,
    so the rows of the returned ``(n_classes, n_replicates)`` array average
    exactly to the reported balanced accuracy.
    """
    divisor = ctx.case_class_divisor
    out = np.zeros((n_classes, ctx.n_replicates), dtype=np.float64)
    for c in range(n_classes):
        mask = labels == c
        if not mask.any():
            continue
        class_weight = ctx.weights.sums(divisor, mask)
        correct_weight = ctx.weights.sums(divisor, mask & (preds == c))
        with np.errstate(divide="ignore", invalid="ignore"):
            out[c] = np.where(
                class_weight > 0, correct_weight / np.maximum(class_weight, 1e-12), 0.0
            )
    return out


def _patch_micro_balanced_accuracy(
    ctx: BootstrapContext, labels: np.ndarray, preds: np.ndarray, n_classes: int
) -> np.ndarray:
    """Class-macro recall with patches pooled within a class, not within a patient."""
    label_weight = ctx.weights.class_sums(1.0, labels, n_classes)
    correct_weight = ctx.weights.class_sums(
        (preds == labels).astype(np.float64), labels, n_classes
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(
            label_weight > 0,
            correct_weight / np.maximum(label_weight, 1e-12),
            np.nan,
        )
    return np.nanmean(recall, axis=0)


def _scalar_distributions(
    ctx: BootstrapContext,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    n_classes: int,
) -> dict[str, np.ndarray]:
    """Distributions of the three scalar secondary endpoints for one draw."""
    labels, preds, probs = arrays
    stacked = probs[np.newaxis, :, :]
    macro_nll = ctx.tail_nll_distribution(labels, stacked, list(range(n_classes)))
    if macro_nll is None:
        raise ValueError("Macro NLL needs at least one class")
    return {
        "macro_nll": macro_nll,
        "expected_calibration_error": ctx.ece_distribution(labels, stacked) * 100.0,
        "patch_micro_balanced_accuracy": _patch_micro_balanced_accuracy(
            ctx, labels, preds, n_classes
        )
        * 100.0,
    }


def draw_secondary_distributions(
    ctx: BootstrapContext,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    class_names: list[str],
) -> dict[str, np.ndarray]:
    """Bootstrap distributions of every secondary endpoint for one draw.

    Accuracies and the calibration error are in percentage points; the macro
    negative log-likelihood stays in nats.
    """
    labels, preds, _ = arrays
    n_classes = len(class_names)
    recalls = _patient_macro_recalls(ctx, labels, preds, n_classes)
    dists = _scalar_distributions(ctx, arrays, n_classes)
    dists.update(
        {
            class_recall_key(name): recalls[i] * 100.0
            for i, name in enumerate(class_names)
        }
    )
    return dists


def pack_estimate(dist: np.ndarray) -> dict[str, float]:
    """Extract the observed point estimate and the 95% interval bounds."""
    ci = confidence_interval(dist)
    return {
        "point": float(dist[0]),
        "ci_2_5": float(ci[0]),
        "ci_97_5": float(ci[1]),
    }


def build_secondary_results(
    secondaries: dict[tuple[int, int], dict[str, np.ndarray]],
    cells: tuple[tuple[int, int], ...],
    class_names: list[str],
) -> dict[str, Any]:
    """Pack cell-level secondary endpoints and their equal-budget contrasts."""
    broad, deep = secondaries[(20, 8)], secondaries[(5, 32)]
    return {
        "cells": {
            f"G{g}_m{m}": {
                key: pack_estimate(secondaries[(g, m)][key])
                for key in SECONDARY_SCALARS
            }
            for g, m in cells
        },
        "equal_budget_X": {
            key: pack_estimate(broad[key] - deep[key]) for key in SECONDARY_SCALARS
        },
        "class_recalls": {
            name: {
                "G5_m32": pack_estimate(deep[class_recall_key(name)]),
                "G20_m8": pack_estimate(broad[class_recall_key(name)]),
                "equal_budget_X": pack_estimate(
                    broad[class_recall_key(name)] - deep[class_recall_key(name)]
                ),
            }
            for name in class_names
        },
    }
