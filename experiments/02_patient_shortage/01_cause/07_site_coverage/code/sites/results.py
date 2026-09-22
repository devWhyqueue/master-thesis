"""Result-payload assembly and distribution-array writing for the analyze stage."""

from __future__ import annotations

from typing import Any

import numpy as np
from imbalance_benchmark.common import output_root

from breadth.analyze.secondary import pack_estimate

from sites import ALLOCATIONS, THRESHOLD_PP
from sites.classify import classify

__all__ = [
    "allocation_payload",
    "build_results",
    "surface_payload",
    "write_distributions",
]


def _draw_contrasts(points: dict[str, np.ndarray]) -> dict[str, Any]:
    """Descriptive spread of the paired site-class contrasts across split-draw fits."""
    diffs = {
        "site_gain": points["broad10"] - points["broad5"],
        "first_step": points["broad5"] - points["deep"],
    }
    return {
        name: {
            "values": diff.tolist(),
            "mean": float(diff.mean()),
            "sd": float(np.std(diff)),
            "min": float(diff.min()),
            "max": float(diff.max()),
            "n_positive": int((diff > 0).sum()),
        }
        for name, diff in diffs.items()
    }


def allocation_payload(
    dists: dict[str, np.ndarray],
    all_dists: dict[str, np.ndarray],
    points: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Per-allocation accuracy payload and the split-draw spread of its contrasts."""
    return {
        "site_class": {
            name: {
                **pack_estimate(dists[name]),
                "draw_dispersion": float(np.std(points[name])),
            }
            for name in ALLOCATIONS
        },
        "all_classes": {name: pack_estimate(all_dists[name]) for name in ALLOCATIONS},
        "draw_contrasts": _draw_contrasts(points),
    }


def surface_payload(
    beta_dist: np.ndarray, gamma_dist: np.ndarray, neff_deep: float, neff_broad: float
) -> dict[str, Any]:
    """Refitted support-surface parameters and the two site-class Neff points."""
    return {
        "beta": pack_estimate(beta_dist),
        "gamma": pack_estimate(gamma_dist),
        "neff_deep": neff_deep,
        "neff_broad": neff_broad,
    }


def build_results(
    site_classes: list[str],
    allocation: dict[str, Any],
    delta_s: np.ndarray,
    b_w: np.ndarray,
    b_ref: np.ndarray,
    surface: dict[str, Any],
    strata: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the analysis.json payload."""
    ds_est, bw_est = pack_estimate(delta_s), pack_estimate(b_w)
    label = classify(
        (ds_est["ci_2_5"], ds_est["ci_97_5"]), (bw_est["ci_2_5"], bw_est["ci_97_5"])
    )
    return {
        "site_classes": site_classes,
        "n_site_classes": len(site_classes),
        "allocation_accuracy": allocation,
        "site_gain": ds_est,
        "within_site_residual": bw_est,
        "reference_residual": pack_estimate(b_ref),
        "within_site_minus_reference": pack_estimate(b_w - b_ref),
        "surface": surface,
        "strata": strata,
        "interpretation": {"label": label, "threshold_pp": THRESHOLD_PP},
    }


def write_distributions(
    config: dict[str, Any],
    allocation_dists: dict[str, np.ndarray],
    delta_s: np.ndarray,
    b_w: np.ndarray,
    b_ref: np.ndarray,
    beta_dist: np.ndarray,
    gamma_dist: np.ndarray,
) -> None:
    """Write the raw paired bootstrap distributions behind analysis.json."""
    np.savez(
        output_root(config) / "data" / "distributions.npz",
        a_deep=allocation_dists["deep"],
        a_broad5=allocation_dists["broad5"],
        a_broad10=allocation_dists["broad10"],
        delta_s=delta_s,
        b_w=b_w,
        b_ref=b_ref,
        beta=beta_dist,
        gamma=gamma_dist,
    )
