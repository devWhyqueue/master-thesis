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
    weighted_macro_nll,
)
from imbalance_benchmark.analysis.metrics import assign_tiers
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
        self.case_ids = identity["case_id"].to_numpy()
        self.row_weights = expand_to_rows(unique_cases, patient_weights, self.case_ids)
        self.n_replicates = n_replicates
        self._seed = seed
        self._seed_indices: dict[int, np.ndarray] = {}

    def _paired_seed_indices(self, n_seeds: int) -> np.ndarray:
        """Return the one fixed seed resample shared by every matched comparison."""
        if n_seeds not in self._seed_indices:
            self._seed_indices[n_seeds] = resample_seed_indices(
                n_seeds, self.n_replicates, np.random.default_rng(self._seed + n_seeds)
            )
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


def _tail_classes(
    freeze: dict[str, Any], class_names: list[str], assignment: str
) -> list[int]:
    """Class indices assigned to the tail tier under the severe condition's allocated support."""
    allocated = (
        freeze.get("assignment_conditions", {})
        .get(assignment, {})
        .get("severe", {})
        .get("allocated_counts", {})
    )
    if not allocated:
        return []
    tiers = assign_tiers(class_names, allocated)
    return [i for i, name in enumerate(class_names) if tiers.get(name) == "tail"]


@dataclass
class Baseline:
    """The balanced-CE reference distributions every severity's gates/recovery compare against."""

    balanced: dict[str, Any]
    ctx: BootstrapContext
    n_classes: int
    tail_classes: list[int]
    n_perm: int
    ba: np.ndarray
    tail_nll: np.ndarray | None


def balanced_baseline(
    paths: dict[str, Path],
    config: dict[str, Any],
    freeze: dict[str, Any],
    n_replicates: int,
    seed: int,
    assignment: str = "native",
) -> Baseline | None:
    """Load balanced CE's predictions and precompute its bootstrap BA/tail-NLL distributions."""
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    balanced = load_seed_predictions(paths, "balanced", "ce")
    if not (paths["data"] / "manifest.csv").exists() or balanced is None:
        return None
    ctx = BootstrapContext(paths, is_mil, n_replicates, seed)
    n_classes = len(balanced["class_names"])
    tail_classes = _tail_classes(freeze, balanced["class_names"], assignment)
    ba = ctx.ba_distribution(balanced["labels"], balanced["preds"], n_classes)
    tail_nll = ctx.tail_nll_distribution(
        balanced["labels"], balanced["probs"], tail_classes
    )
    return Baseline(balanced, ctx, n_classes, tail_classes, 100_000, ba, tail_nll)
