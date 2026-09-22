from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from imbalance_benchmark.analysis.inference.bootstrap import (
    PatientWeights,
    build_strata,
    case_class_divisor,
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
from imbalance_benchmark.analysis.reporting.secondary_intervals.probability import (
    _secondary_distributions as probability_secondary_distributions,
)
from imbalance_benchmark.analysis.query import load_seed_predictions, load_test_identity
from imbalance_benchmark.manifest.construction_helpers import CONDITION_REFERENCE

__all__ = ["BootstrapContext", "Baseline", "CONDITION_REFERENCE", "balanced_baseline"]


def _seed_distributions(
    per_seed: list[dict[str, np.ndarray]], seed_indices: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        endpoint: gather_seed_resampled(
            np.stack([seed_metrics[endpoint] for seed_metrics in per_seed]),
            seed_indices,
        )
        for endpoint in per_seed[0]
    }


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
        # Factorized once (not per secondary-endpoint group-mean call) since
        # this context is reused across every key a worker processes.
        self.slide_codes, _ = pd.factorize(self.slide_ids, sort=False)
        self.case_codes, _ = pd.factorize(self.case_ids, sort=False)
        case_labels = identity["cancer_type"].astype(str).to_numpy()
        self.case_class_divisor = case_class_divisor(self.case_ids, case_labels)
        position = {c: i for i, c in enumerate(unique_cases)}
        row_patient = np.asarray([position[c] for c in self.case_ids])
        # Replicate 0 is the observed cohort (all-ones weight): every metric
        # carries the observed point estimate at index 0 (used by reported
        # effects/recovery/deficit gates); replicates 1.. give the interval.
        observed = np.ones((patient_weights.shape[0], 1), dtype=patient_weights.dtype)
        patient = np.concatenate([observed, patient_weights], axis=1).astype(np.float64)
        self.weights = PatientWeights(row_patient, patient)
        self.n_replicates = self.weights.n_replicates
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

    def _paired_seed_metric(self, n_seeds: int, per_seed_fn) -> np.ndarray:
        """Stack one metric over confirmation seeds, then average a paired seed resample."""
        per_seed = np.stack([per_seed_fn(i) for i in range(n_seeds)])
        return gather_seed_resampled(per_seed, self._paired_seed_indices(n_seeds))

    def ba_distribution(
        self, labels: np.ndarray, preds_stack: np.ndarray, n_classes: int
    ) -> np.ndarray:
        """Per-replicate case-macro balanced accuracy, averaged over a paired seed resample."""
        return self._paired_seed_metric(
            preds_stack.shape[0],
            lambda i: weighted_balanced_accuracy(
                labels, preds_stack[i], self.weights, n_classes, self.case_class_divisor
            ),
        )

    def tail_nll_distribution(
        self, labels: np.ndarray, probs_stack: np.ndarray, tail_classes: list[int]
    ) -> np.ndarray | None:
        """Per-replicate case-macro tail-group NLL, averaged over a paired seed resample."""
        if not tail_classes:
            return None
        return self._paired_seed_metric(
            probs_stack.shape[0],
            lambda i: weighted_macro_nll(
                labels,
                probs_stack[i],
                self.weights,
                tail_classes,
                self.case_class_divisor,
            ),
        )

    def ece_distribution(
        self, labels: np.ndarray, probs_stack: np.ndarray
    ) -> np.ndarray:
        """Per-replicate fixed-bin ECE from the frozen crossed patient bootstrap."""
        return self._paired_seed_metric(
            probs_stack.shape[0],
            lambda i: weighted_ece(labels, probs_stack[i], self.weights),
        )

    def secondary_distributions(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        class_names: list[str],
        tiers: dict[str, str],
        *,
        is_mil: bool = False,
        ordinal: bool = False,
    ) -> dict[str, np.ndarray]:
        """Return paired-seed distributions for every secondary endpoint."""
        per_seed = [
            secondary_seed_metrics(
                (labels, predictions[index], probabilities[index]),
                self.weights,
                class_names,
                tiers,
                (self.slide_codes, self.case_codes),
                is_mil=is_mil,
                ordinal=ordinal,
            )
            for index in range(predictions.shape[0])
        ]
        seed_indices = self._paired_seed_indices(predictions.shape[0])
        return _seed_distributions(per_seed, seed_indices)

    def probability_secondary_distributions(
        self,
        labels: np.ndarray,
        probabilities: np.ndarray,
        class_names: list[str],
        tiers: dict[str, str],
    ) -> dict[str, np.ndarray]:
        """Return paired-seed distributions for probability-only endpoints."""
        return probability_secondary_distributions(
            self, labels, probabilities, class_names, tiers
        )


def _tail_classes(
    freeze: dict[str, Any], class_names: list[str], assignment: str, severity: str
) -> list[int]:
    """Class indices assigned to the tail tier under one condition's allocated
    support -- head/body/tail tiers use that condition's realized allocation.
    """
    condition = (
        freeze.get("assignment_conditions", {}).get(assignment, {}).get(severity, {})
    )
    allocated = condition.get("allocated_counts", {})
    if not allocated:
        return []
    if spread_tail := condition.get("spread_tail_classes"):
        return [i for i, name in enumerate(class_names) if name in spread_tail]
    tiers = assign_tiers(
        class_names,
        allocated,
        freeze.get("tail_assignments", {}).get(assignment, class_names),
    )
    return [i for i, name in enumerate(class_names) if tiers.get(name) == "tail"]


@dataclass
class Baseline:
    """The balanced-CE reference distributions every severity's gates/recovery compare against.
    The tail group is severity-specific, so ``tail_nll`` is computed per severity
    by the recovery layer from ``freeze``/``assignment``, not precomputed here.
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
    condition: str = "moderate",
) -> Baseline | None:
    """Load a condition's prespecified CE reference and bootstrap BA distribution."""
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    balanced = load_seed_predictions(
        paths, CONDITION_REFERENCE[condition], "ce", assignment
    )
    if not (paths["data"] / "manifest.csv").exists() or balanced is None:
        return None
    ctx = BootstrapContext(paths, is_mil, n_replicates, seed)
    n_classes = len(balanced["class_names"])
    ba = ctx.ba_distribution(balanced["labels"], balanced["preds"], n_classes)
    return Baseline(balanced, ctx, n_classes, 100_000, ba, freeze, assignment)
