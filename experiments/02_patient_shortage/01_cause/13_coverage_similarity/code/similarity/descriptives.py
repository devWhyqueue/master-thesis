"""Per-class descriptive statistics for the census report (non-gating)."""

from __future__ import annotations

from typing import Any

import numpy as np

from breadth.analyze.diagnostics import _mean_leaves, tissue_source_site

from similarity.geometry import ClassGeometry
from similarity.search import CohortResult

__all__ = ["class_descriptives"]


def _cohort_leaf(result: CohortResult) -> dict[str, float]:
    return {
        "r_train": result.r_train,
        "r_val": result.r_val,
        "omega": result.omega,
        "neff": result.neff if result.neff is not None else float("nan"),
    }


def _nearest_distance_quantiles(
    result: CohortResult, geo: ClassGeometry
) -> dict[str, float]:
    """Validation nearest-cohort-distance quantiles for one cohort draw."""
    position = {p: i for i, p in enumerate(geo.pool)}
    idx = np.array([position[p] for p in result.patients])
    nearest = np.min(geo.d_val[:, idx], axis=1)
    return {
        "q25": float(np.quantile(nearest, 0.25)),
        "q50": float(np.quantile(nearest, 0.50)),
        "q75": float(np.quantile(nearest, 0.75)),
    }


def _site_diversity(draws: list[dict[str, CohortResult]], name: str) -> float:
    return float(
        np.mean([len({tissue_source_site(p) for p in r[name].patients}) for r in draws])
    )


def class_descriptives(
    draws: list[dict[str, CohortResult]], geo: ClassGeometry
) -> dict[str, Any]:
    """One class's cohort descriptives, validation-distance quantiles, and site diversity."""
    cohort_names = list(draws[0])
    return {
        "cohorts": {
            name: _mean_leaves([_cohort_leaf(r[name]) for r in draws])
            for name in cohort_names
        },
        "validation_nearest_quantiles": {
            name: _mean_leaves(
                [_nearest_distance_quantiles(r[name], geo) for r in draws]
            )
            for name in cohort_names
        },
        "distinct_sites": {name: _site_diversity(draws, name) for name in cohort_names},
        "unique_cohorts_across_draws": {
            name: len({tuple(sorted(r[name].patients)) for r in draws})
            for name in cohort_names
        },
    }
