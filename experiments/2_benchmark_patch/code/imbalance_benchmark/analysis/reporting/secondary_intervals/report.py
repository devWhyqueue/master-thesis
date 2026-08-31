from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from imbalance_benchmark.analysis.inference.gates import confidence_interval
from imbalance_benchmark.analysis.reporting.secondary_intervals.calibration_intervals import (
    write_crossed_calibration_table,
)
from imbalance_benchmark.analysis.reporting.secondary_intervals.costs import (
    write_cost_comparison_table,
)
from imbalance_benchmark.analysis.reporting.secondary_intervals.interval_cache import (
    distributions_by_key,
)

__all__ = ["secondary_interval_rows", "write_interval_tables"]

logger = logging.getLogger(__name__)


def secondary_interval_rows(
    distributions: dict[tuple[str, str, str], dict[str, np.ndarray]],
) -> list[dict[str, object]]:
    """Return equal-split secondary estimates with crossed patient-bootstrap CIs."""
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
    base_paths: dict[str, Path],
    distributions: dict[tuple[str, str, str], dict[str, np.ndarray]],
) -> None:
    table = pd.DataFrame(secondary_interval_rows(distributions))
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
    dataset = config.get("dataset", {})
    is_mil = dataset.get("regime", "patch") == "wsi"
    ordinal = is_mil and dataset.get("name") == "panda"
    logger.info("interval: endpoint distributions")
    distributions = distributions_by_key(
        base_paths, is_mil, ordinal, n_replicates, seed
    )
    logger.info("interval: calibration table")
    write_crossed_calibration_table(base_paths, distributions)
    logger.info("interval: cost comparison table")
    write_cost_comparison_table(base_paths, n_replicates, seed)
    logger.info("interval: secondary endpoint table")
    _write_secondary_interval_table(base_paths, distributions)
