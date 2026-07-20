from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from imbalance_benchmark.analysis.inference.bootstrap import (
    build_strata,
    expand_to_rows,
    gather_seed_resampled,
    resample_patient_weights,
    resample_seed_indices,
    weighted_balanced_accuracy,
    weighted_ece,
    weighted_macro_nll,
)
from imbalance_benchmark.analysis.metrics import assign_tiers
from imbalance_benchmark.analysis.reporting.secondary_intervals.metrics import (
    secondary_seed_metrics,
)
from imbalance_benchmark.analysis.query import load_seed_predictions, load_test_identity

__all__ = ["BootstrapContext", "Baseline", "balanced_baseline"]


def _crossed_test_identity(paths: dict[str, Path], is_mil: bool) -> pd.DataFrame:
    """Load every available fixed test split for shared patient-block resampling."""
    frames = []
    for index in range(3):
        manifest = paths["root"].parent / f"split={index}" / "data" / "manifest.csv"
        if manifest.exists():
            frames.append(
                load_test_identity(manifest, is_mil).assign(patient_split=index)
            )
    if not frames:
        local_manifest = paths["data"] / "manifest.csv"
        if not local_manifest.exists():
            raise FileNotFoundError(
                "No prepared test manifest is available for bootstrap"
            )
        frames.append(
            load_test_identity(local_manifest, is_mil).assign(patient_split=0)
        )
    return pd.concat(frames, ignore_index=True)


class BootstrapContext:
    """Precomputed patient-block resampling shared by every gate/recovery comparison."""

    def __init__(
        self, paths: dict[str, Path], is_mil: bool, n_replicates: int, seed: int
    ) -> None:
        identity = load_test_identity(paths["data"] / "manifest.csv", is_mil)
        crossed_identity = _crossed_test_identity(paths, is_mil)
        strata = build_strata(crossed_identity)
        rng = np.random.default_rng(seed)
        unique_cases, patient_weights = resample_patient_weights(
            strata, n_replicates, rng
        )
        self.case_ids = identity["case_id"].astype(str).to_numpy()
        self.slide_ids = identity["slide_id"].astype(str).to_numpy()
        resampled = expand_to_rows(unique_cases, patient_weights, self.case_ids)
        # Replicate 0 is the observed cohort (all unit weights one): every metric
        # distribution therefore carries the observed-data point estimate at
        # index 0, which the reported effects, recovery ratios, and deficit gates
        # use, while replicates 1.. are the bootstrap draws for the interval.
        observed = np.ones((resampled.shape[0], 1), dtype=resampled.dtype)
        self.row_weights = np.concatenate([observed, resampled], axis=1)
        self.n_replicates = self.row_weights.shape[1]
        self._seed = seed
        self._seed_indices: dict[int, np.ndarray] = {}

    def _paired_seed_indices(self, n_seeds: int) -> np.ndarray:
        """Return the one fixed seed resample shared by every matched comparison."""
        if n_seeds not in self._seed_indices:
            idx = resample_seed_indices(
                n_seeds, self.n_replicates, np.random.default_rng(self._seed + n_seeds)
            )
            idx[0] = np.arange(n_seeds)  # observed replicate averages all seeds
            self._seed_indices[n_seeds] = idx
        return self._seed_indices[n_seeds]

    def ba_distribution(
        self, labels: np.ndarray, preds_stack: np.ndarray, n_classes: int
    ) -> np.ndarray:
        """Per-replicate weighted balanced accuracy, averaged over a paired seed resample."""
        per_seed = np.stack(
            [
                weighted_balanced_accuracy(
                    labels, preds_stack[i], self.row_weights, n_classes
                )
                for i in range(preds_stack.shape[0])
            ]
        )
        return gather_seed_resampled(
            per_seed, self._paired_seed_indices(preds_stack.shape[0])
        )

    def tail_nll_distribution(
        self, labels: np.ndarray, probs_stack: np.ndarray, tail_classes: list[int]
    ) -> np.ndarray | None:
        """Per-replicate weighted tail-group macro NLL, averaged over a paired seed resample."""
        if not tail_classes:
            return None
        per_seed = np.stack(
            [
                weighted_macro_nll(
                    labels, probs_stack[i], self.row_weights, tail_classes
                )
                for i in range(probs_stack.shape[0])
            ]
        )
        return gather_seed_resampled(
            per_seed, self._paired_seed_indices(probs_stack.shape[0])
        )

    def ece_distribution(
        self, labels: np.ndarray, probs_stack: np.ndarray
    ) -> np.ndarray:
        """Per-replicate fixed-bin ECE from the frozen crossed patient bootstrap."""
        per_seed = np.stack(
            [
                weighted_ece(labels, probs_stack[i], self.row_weights)
                for i in range(probs_stack.shape[0])
            ]
        )
        return gather_seed_resampled(
            per_seed, self._paired_seed_indices(probs_stack.shape[0])
        )

    def secondary_distributions(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        class_names: list[str],
        tiers: dict[str, str],
    ) -> dict[str, np.ndarray]:
        """Return paired-seed distributions for every secondary endpoint."""
        per_seed = [
            secondary_seed_metrics(
                labels,
                predictions[index],
                probabilities[index],
                self.row_weights,
                class_names,
                tiers,
                self.slide_ids,
                self.case_ids,
            )
            for index in range(predictions.shape[0])
        ]
        seed_indices = self._paired_seed_indices(predictions.shape[0])
        return {
            endpoint: gather_seed_resampled(
                np.stack([seed_metrics[endpoint] for seed_metrics in per_seed]),
                seed_indices,
            )
            for endpoint in per_seed[0]
        }


def _tail_classes(
    freeze: dict[str, Any], class_names: list[str], assignment: str, severity: str
) -> list[int]:
    """Class indices assigned to the tail tier under one condition's allocated support.

    Head/body/tail tiers are defined per comparison unit from that condition's
    realized allocation, so a moderate comparison must not read the severe
    allocation: with class-specific caps the two allocations can rank classes
    differently and yield different tail groups.
    """
    allocated = (
        freeze.get("assignment_conditions", {})
        .get(assignment, {})
        .get(severity, {})
        .get("allocated_counts", {})
    )
    if not allocated:
        return []
    tiers = assign_tiers(
        class_names,
        allocated,
        freeze.get("tail_assignments", {}).get(assignment, class_names),
    )
    return [i for i, name in enumerate(class_names) if tiers.get(name) == "tail"]


@dataclass
class Baseline:
    """The balanced-CE reference distributions every severity's gates/recovery compare against.

    The tail group is severity-specific, so ``tail_nll`` is computed per
    severity by the recovery layer from ``freeze``/``assignment`` rather than
    precomputed here for a single condition.
    """

    balanced: dict[str, Any]
    ctx: BootstrapContext
    n_classes: int
    n_perm: int
    ba: np.ndarray
    freeze: dict[str, Any]
    assignment: str


def balanced_baseline(
    paths: dict[str, Path],
    config: dict[str, Any],
    freeze: dict[str, Any],
    n_replicates: int,
    seed: int,
    assignment: str = "native",
) -> Baseline | None:
    """Load balanced CE's predictions and precompute its bootstrap BA distribution."""
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    balanced = load_seed_predictions(paths, "balanced", "ce")
    if not (paths["data"] / "manifest.csv").exists() or balanced is None:
        return None
    ctx = BootstrapContext(paths, is_mil, n_replicates, seed)
    n_classes = len(balanced["class_names"])
    ba = ctx.ba_distribution(balanced["labels"], balanced["preds"], n_classes)
    return Baseline(balanced, ctx, n_classes, 100_000, ba, freeze, assignment)
