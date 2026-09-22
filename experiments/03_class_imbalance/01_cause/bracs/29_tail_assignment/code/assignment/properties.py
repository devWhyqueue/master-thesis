"""Class properties (r1 recall, r1 confusability) and their correlation with tail damage."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from scipy.stats import spearmanr

from breadth.analyze.canonical import canonical_permutation

from centre import N_DRAWS, N_SPLITS

from sites import allocation_dir

from assignment._io import paths_by_split, require_record

__all__ = ["class_properties", "correlations_by_rho"]


def _confusion_counts(config: dict[str, Any], names: list[str]) -> np.ndarray:
    """Pooled canonical-order confusion matrix (C, C) of r1's stored test predictions."""
    paths = paths_by_split(config)
    n = len(names)
    mat = np.zeros((n, n), dtype=np.int64)
    for s in range(N_SPLITS):
        inv = np.argsort(canonical_permutation(config, s, names))
        for d in range(N_DRAWS):
            test = require_record(
                allocation_dir(paths[s], "r1", d),
                splits=("test",),
                array_fields=("labels", "preds"),
            )["splits"]["test"]
            labels = inv[np.asarray(test["labels"])]
            preds = inv[np.asarray(test["preds"])]
            mat += np.bincount(labels * n + preds, minlength=n * n).reshape(n, n)
    return mat


def _confusability(mat: np.ndarray) -> np.ndarray:
    """Off-diagonal confusion mass to and from each class, as a share of every stored prediction."""
    diag = np.diag(mat)
    return (mat.sum(axis=1) - diag + mat.sum(axis=0) - diag) / mat.sum()


def class_properties(
    exp26_config: dict[str, Any], names: list[str], r1_own: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    """r1 recall (headroom) and r1 confusability, per class."""
    confusability = _confusability(_confusion_counts(exp26_config, names))
    return {
        c: {"r1_recall": float(r1_own[c][0]), "confusability": float(confusability[ci])}
        for ci, c in enumerate(names)
    }


def _spearman(x: list[float], y: list[float]) -> dict[str, float]:
    """Spearman rank correlation, descriptive (n = number of classes, no multiplicity correction)."""
    r, p = cast(tuple[float, float], spearmanr(x, y))
    return {"n": len(x), "spearman_r": float(r), "spearman_p": float(p)}


def correlations_by_rho(
    names: list[str],
    properties: dict[str, dict[str, float]],
    d_by_rho: dict[int, dict[str, np.ndarray]],
    own_loss_by_rho: dict[int, dict[str, np.ndarray]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Spearman correlation of r1 recall / confusability against tail damage, per ratio."""
    r1_recall = [properties[c]["r1_recall"] for c in names]
    confusability = [properties[c]["confusability"] for c in names]
    out: dict[str, dict[str, dict[str, float]]] = {}
    for rho, d_by_class in d_by_rho.items():
        d_points = [float(d_by_class[c][0]) for c in names]
        loss_points = [float(own_loss_by_rho[rho][c][0]) for c in names]
        out[str(rho)] = {
            "r1_recall_vs_D": _spearman(r1_recall, d_points),
            "r1_recall_vs_own_loss": _spearman(r1_recall, loss_points),
            "confusability_vs_D": _spearman(confusability, d_points),
            "confusability_vs_own_loss": _spearman(confusability, loss_points),
        }
    return out
