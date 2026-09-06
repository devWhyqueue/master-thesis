"""The deficit-gate thresholds, recomputed on this experiment's own 'random' cells."""

from __future__ import annotations

from typing import Any

from imbalance_benchmark.analysis.inference.context import BootstrapContext
from imbalance_benchmark.analysis.query import load_seed_predictions

# derive_deficit_thresholds.py is a top-level script sibling to
# imbalance_benchmark, not a package module; importable once __main__.py
# prepends experiments/2_benchmark_patch/code to sys.path (plan "Cluster
# wiring"). Reused so exp-3 recomputes its OWN sigma_seed by the benchmark's
# exact procedure rather than pasting a value derived from different
# training sets (plan Stage 4 / report Sec. "Endpoints, Estimands, and Gates").
import derive_deficit_thresholds as ddt

from diversity.analyze.common import CE, fixed_tail_classes, iter_splits

__all__ = ["gate_thresholds", "gate_passes"]


def _dataset_sigma_rows(config: dict[str, Any], dataset: str) -> list[Any]:
    """This experiment's own dispersion rows, from the 'random' anchor's CE fits."""
    rows: list[Any] = []
    for split_index, exp3_paths, freeze in iter_splits(config):
        class_names = list(freeze["class_names"])
        try:
            balanced = load_seed_predictions(
                exp3_paths, "balanced", CE, assignment="random"
            )
        except RuntimeError:
            continue
        if balanced is None:
            continue
        ctx = BootstrapContext(exp3_paths, False, n_replicates=1, seed=0)
        n_seeds = balanced["preds"].shape[0]
        rows.append(
            ddt.DispersionRow(
                dataset,
                split_index,
                None,
                "ba",
                ddt._ba_sigma(balanced, ctx, len(class_names), n_seeds),
            )
        )
        tail_classes = fixed_tail_classes(freeze, class_names)
        if tail_classes:
            rows.append(
                ddt.DispersionRow(
                    dataset,
                    split_index,
                    "severe",
                    "tail_nll",
                    ddt._tail_nll_sigma(balanced, ctx, tail_classes, n_seeds),
                )
            )
    return rows


def gate_thresholds(config: dict[str, Any], dataset: str) -> ddt.DatasetThresholds:
    """Dataset-specific ``max(1pp, 2*sigma_seed)``-style thresholds (plan Stage 4)."""
    return ddt._dataset_thresholds(dataset, _dataset_sigma_rows(config, dataset))


def gate_passes(effect: float, ci: tuple[float, float], threshold: float) -> bool:
    """Material when |effect| clears the dataset threshold and the CI excludes zero.

    Uses ``abs(effect)`` rather than exp-2's ``gates.discrimination_gate``/
    ``calibration_gate`` (which assume a fixed sign): eq. (4) is defined
    literally as wide-minus-narrow with no reorientation by endpoint, so its
    sign is not fixed in advance for tail-group NLL, and the report's own
    materiality rule is stated in terms of "magnitude" (Sec. "Endpoints,
    Estimands, and Gates").
    """
    ci_excludes_zero = ci[0] > 0.0 or ci[1] < 0.0
    return abs(effect) >= threshold and ci_excludes_zero
