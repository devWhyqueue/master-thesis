from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from imbalance_benchmark.analysis.inference.context import BootstrapContext
from imbalance_benchmark.analysis.inference.gates import confidence_interval
from imbalance_benchmark.analysis.metrics import assign_tiers
from imbalance_benchmark.analysis.query import load_seed_predictions
from imbalance_benchmark.analysis.reporting.calibration_intervals import (
    _complete_result_keys,
    write_crossed_calibration_table,
)
from imbalance_benchmark.analysis.reporting.secondary_intervals.costs import (
    write_cost_comparison_table,
)
from imbalance_benchmark.common import split_paths

__all__ = ["secondary_interval_rows", "write_interval_tables"]

logger = logging.getLogger(__name__)


def _locked_tiers(
    paths: dict[str, Path], assignment: str, condition: str, class_names: list[str]
) -> dict[str, str]:
    freeze_path = paths["data"] / "manifest_freeze.json"
    if not freeze_path.exists():
        return {}
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    allocated = (
        freeze.get("assignment_conditions", {})
        .get(assignment, {})
        .get(condition, {})
        .get("allocated_counts", {})
    )
    if not allocated:
        return {}
    order = freeze.get("tail_assignments", {}).get(assignment, class_names)
    return assign_tiers(class_names, allocated, order)


def _split_distributions(
    base_paths: dict[str, Path],
    contexts: list[BootstrapContext],
    is_mil: bool,
    ordinal: bool,
    assignment: str,
    condition: str,
    method: str,
) -> list[dict[str, np.ndarray]]:
    distributions = []
    for index in range(3):
        paths = split_paths(base_paths, index)
        record = load_seed_predictions(paths, condition, method, assignment)
        if record is None:
            raise RuntimeError(
                f"Missing secondary endpoints for {assignment}/{condition}/{method}"
            )
        class_names = list(record["class_names"])
        tiers = _locked_tiers(paths, assignment, condition, class_names)
        context = contexts[index]
        current = context.secondary_distributions(
            np.asarray(record["labels"]),
            np.asarray(record["preds"]),
            np.asarray(record["probs"]),
            class_names,
            tiers,
            is_mil=is_mil,
            ordinal=ordinal,
        )
        scaled = context.secondary_distributions(
            np.asarray(record["labels"]),
            np.asarray(record["preds"]),
            np.asarray(record["temperature_scaled_probs"]),
            class_names,
            tiers,
            is_mil=is_mil,
            ordinal=ordinal,
        )
        current.update(
            {
                f"temperature_scaled_{name}": values
                for name, values in scaled.items()
                if name
                in {
                    "negative_log_likelihood",
                    "macro_nll",
                    "brier_score",
                    "expected_calibration_error",
                }
                or name.startswith(("nll:", "brier:", "tier_nll:", "tier_brier:"))
                or name.endswith(("_macro_nll", "_macro_brier"))
                or name in {"patch_micro_nll", "patch_micro_brier"}
            }
        )
        distributions.append(current)
    return distributions


def _distributions_by_key(
    base_paths: dict[str, Path],
    is_mil: bool,
    ordinal: bool,
    n_replicates: int,
    seed: int,
) -> dict[tuple[str, str, str], dict[str, np.ndarray]]:
    keys = sorted(_complete_result_keys(base_paths))
    # Contexts depend only on the split (paths/is_mil/n_replicates/seed), never on
    # assignment/condition/method -- build the 3 once instead of once per key.
    contexts = [
        BootstrapContext(split_paths(base_paths, index), is_mil, n_replicates, seed)
        for index in range(3)
    ]
    distributions = {}
    for step, key in enumerate(keys, start=1):
        assignment, condition, method = key
        logger.info(
            "interval: %s/%s/%s %d/%d", assignment, condition, method, step, len(keys)
        )
        split_values = _split_distributions(
            base_paths, contexts, is_mil, ordinal, assignment, condition, method
        )
        distributions[key] = _average_split_values(split_values)
    return distributions


def secondary_interval_rows(
    base_paths: dict[str, Path], config: dict[str, Any], n_replicates: int, seed: int
) -> list[dict[str, object]]:
    """Return equal-split secondary estimates with crossed patient-bootstrap CIs."""
    dataset = config.get("dataset", {})
    is_mil = dataset.get("regime", "patch") == "wsi"
    ordinal = is_mil and dataset.get("name") == "panda"
    distributions = _distributions_by_key(
        base_paths, is_mil, ordinal, n_replicates, seed
    )
    return [
        row
        for key, values in distributions.items()
        for row in _result_rows(
            values, key, _reference_key(key, distributions), distributions
        )
    ]


def _average_split_values(
    split_values: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    return {
        endpoint: np.mean(
            np.stack([values[endpoint] for values in split_values]), axis=0
        )
        for endpoint in split_values[0]
    }


def _result_rows(
    values: dict[str, np.ndarray],
    identity: tuple[str, str, str],
    reference_key: tuple[str, str, str] | None,
    all_values: dict[tuple[str, str, str], dict[str, np.ndarray]],
) -> list[dict[str, object]]:
    reference = all_values.get(reference_key) if reference_key else None
    return [
        _endpoint_row(identity, endpoint, distribution, reference_key, reference)
        for endpoint, distribution in values.items()
    ]


def _reference_key(
    key: tuple[str, str, str],
    values: dict[tuple[str, str, str], dict[str, np.ndarray]],
) -> tuple[str, str, str] | None:
    assignment, condition, method = key
    if method != "ce":
        candidate = (assignment, condition, "ce")
        return candidate if candidate in values else None
    if condition not in {"natural", "balanced"}:
        candidate = ("unassigned", "balanced", "ce")
        return candidate if candidate in values else None
    return None


def _endpoint_row(
    identity: tuple[str, str, str],
    endpoint: str,
    distribution: np.ndarray,
    reference_key: tuple[str, str, str] | None,
    reference: dict[str, np.ndarray] | None,
) -> dict[str, object]:
    estimate_low, estimate_high = confidence_interval(distribution)
    # A cross-condition reference (e.g. balanced CE) can lack a tier-specific
    # endpoint the compared run has (balanced has no head/tail split).
    effect = None
    if reference is not None and endpoint in reference:
        effect = distribution - reference[endpoint]
    effect_ci = confidence_interval(effect) if effect is not None else (None, None)
    return {
        "assignment": identity[0],
        "condition": identity[1],
        "method": identity[2],
        "endpoint": endpoint,
        "estimate": float(distribution[0]),
        "ci_low": estimate_low,
        "ci_high": estimate_high,
        "reference": "/".join(reference_key) if reference_key else None,
        "effect": float(effect[0]) if effect is not None else None,
        "effect_ci_low": effect_ci[0],
        "effect_ci_high": effect_ci[1],
    }


def _write_secondary_interval_table(
    base_paths: dict[str, Path], config: dict[str, Any], n_replicates: int, seed: int
) -> None:
    table = pd.DataFrame(
        secondary_interval_rows(base_paths, config, n_replicates, seed)
    )
    text = table.to_latex(
        index=False,
        float_format="%.3f",
        escape=True,
        longtable=True,
        caption="Secondary endpoint estimates with crossed 95\\% confidence intervals.",
        label="tab:secondary-intervals",
    )
    base_paths["tables"].mkdir(parents=True, exist_ok=True)
    (base_paths["tables"] / "secondary_endpoint_intervals.tex").write_text(
        text, encoding="utf-8"
    )


def write_interval_tables(
    base_paths: dict[str, Path], config: dict[str, Any], n_replicates: int, seed: int
) -> None:
    """Write calibration, cost-comparison, and full secondary interval tables."""
    logger.info("interval: calibration table")
    write_crossed_calibration_table(base_paths, config, seed)
    logger.info("interval: cost comparison table")
    write_cost_comparison_table(base_paths, n_replicates, seed)
    logger.info("interval: secondary endpoint table")
    _write_secondary_interval_table(base_paths, config, n_replicates, seed)
