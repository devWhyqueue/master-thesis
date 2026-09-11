"""Analyze stage: coverage gain, within-neighbourhood residual, and interpretation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.common import ensure_dirs, output_root, split_paths, write_json

from breadth import N_SPLITS
from breadth.analyze.secondary import pack_estimate

from neighbours import ALLOCATIONS, THRESHOLD_PP
from neighbours.accuracy import SubgroupFit, fit, prepare
from neighbours.census import load_census
from neighbours.secondary import secondary_analysis

__all__ = ["classify", "run_analyze"]

logger = logging.getLogger(__name__)


def _above_threshold(ci: tuple[float, float]) -> bool:
    """Whether an interval sits entirely above the practical threshold."""
    return ci[0] > THRESHOLD_PP


def _within_threshold(ci: tuple[float, float]) -> bool:
    """Whether an interval sits entirely within +/- the practical threshold."""
    return ci[0] >= -THRESHOLD_PP and ci[1] <= THRESHOLD_PP


def classify(dc_ci: tuple[float, float], bn_ci: tuple[float, float]) -> str:
    """Interpretation label (report Table "accounts").

    Unlike the site experiment, the coverage account admits a negative
    within-neighbourhood residual, so ``bn_ci`` need not sit above zero to
    count as "below threshold" -- only within it.
    """
    if _above_threshold(dc_ci) and bn_ci[1] <= THRESHOLD_PP:
        return "patient_coverage"
    if _within_threshold(dc_ci) and _above_threshold(bn_ci):
        return "count_beyond_coverage"
    if _above_threshold(dc_ci) and _above_threshold(bn_ci):
        return "both"
    return "inconclusive"


def _allocation_payload(results: dict[str, SubgroupFit]) -> dict[str, Any]:
    """Per-subgroup, per-allocation accuracy and the split-draw contrast spread."""
    payload: dict[str, Any] = {}
    for subgroup, result in results.items():
        points = {
            "deep": result.points_deep,
            "neighbours": result.points_neighbours,
            "random": result.points_random,
        }
        dists = {
            "deep": result.a_deep,
            "neighbours": result.a_neighbours,
            "random": result.a_random,
        }
        first_step = points["neighbours"] - points["deep"]
        coverage_gain = points["random"] - points["neighbours"]
        payload[subgroup] = {
            "accuracy": {
                name: {
                    **pack_estimate(dists[name]),
                    "draw_dispersion": float(np.std(points[name])),
                }
                for name in dists
            },
            "draw_contrasts": {
                "coverage_gain": {
                    "values": coverage_gain.tolist(),
                    "mean": float(coverage_gain.mean()),
                    "sd": float(np.std(coverage_gain)),
                    "n_positive": int((coverage_gain > 0).sum()),
                },
                "first_step": {
                    "values": first_step.tolist(),
                    "mean": float(first_step.mean()),
                    "sd": float(np.std(first_step)),
                    "n_positive": int((first_step > 0).sum()),
                },
            },
            "surface": {
                "beta": pack_estimate(result.beta),
                "gamma": pack_estimate(result.gamma),
                "neff_deep": result.neff_deep,
                "neff_broad": result.neff_broad,
            },
            "delta_c": pack_estimate(result.delta_c),
            "b_n": pack_estimate(result.b_n),
            "b_ref": pack_estimate(result.b_ref),
            "b_n_minus_b_ref": pack_estimate(result.b_n - result.b_ref),
        }
    return payload


def _census_summary(census: dict[str, Any]) -> dict[str, Any]:
    """Kappa and site shares, unpacked from the signed census for the report."""
    return {
        s: {
            "r_deep": v["r_deep"],
            "r_neighbours": v["r_neighbours"],
            "r_random": v["r_random"],
            "kappa": v["kappa"],
        }
        for s, v in census["splits"].items()
    }


def _write_distributions(config: dict[str, Any], results: dict[str, Any]) -> None:
    """Write the raw paired bootstrap distributions behind analysis.json."""
    arrays: dict[str, Any] = {}
    for subgroup, result in results.items():
        arrays[f"{subgroup}_a_deep"] = result.a_deep
        arrays[f"{subgroup}_a_neighbours"] = result.a_neighbours
        arrays[f"{subgroup}_a_random"] = result.a_random
        arrays[f"{subgroup}_delta_c"] = result.delta_c
        arrays[f"{subgroup}_b_n"] = result.b_n
        arrays[f"{subgroup}_b_ref"] = result.b_ref
        arrays[f"{subgroup}_beta"] = result.beta
        arrays[f"{subgroup}_gamma"] = result.gamma
    np.savez(output_root(config) / "data" / "distributions.npz", **arrays)


def _build_results(
    all_result: SubgroupFit,
    results: dict[str, SubgroupFit],
    census: dict[str, Any],
    secondary: dict[str, Any],
    dc_est: dict[str, float],
    bn_est: dict[str, float],
    label: str,
) -> dict[str, Any]:
    """Assemble the analysis.json payload."""
    return {
        "allocations": list(ALLOCATIONS),
        "census": _census_summary(census),
        "grid_reference": {"cell": "G20_m8"},
        "subgroups": _allocation_payload(results),
        "coverage_gain": dc_est,
        "within_neighbourhood_residual": bn_est,
        "reference_residual": pack_estimate(all_result.b_ref),
        "within_neighbourhood_minus_reference": pack_estimate(
            all_result.b_n - all_result.b_ref
        ),
        "secondary": secondary,
        "interpretation": {"label": label, "threshold_pp": THRESHOLD_PP},
    }


def run_analyze(config: dict[str, Any]) -> Path:
    """Compute the coverage gain, within-neighbourhood residual, and their interpretation."""
    ctx = prepare(config)
    results = fit(config, ctx)
    all_result = results["all"]
    dc_est = pack_estimate(all_result.delta_c)
    bn_est = pack_estimate(all_result.b_n)
    label = classify(
        (dc_est["ci_2_5"], dc_est["ci_97_5"]), (bn_est["ci_2_5"], bn_est["ci_97_5"])
    )

    paths8 = {s: split_paths(ensure_dirs(config), s) for s in range(N_SPLITS)}
    secondary = secondary_analysis(config, paths8, ctx.class_names)
    census = load_census(config)

    out = _build_results(all_result, results, census, secondary, dc_est, bn_est, label)
    out_p = output_root(config) / "data" / "analysis.json"
    write_json(out_p, out)
    _write_distributions(config, results)
    logger.info("Patient-coverage analysis complete: %s", out_p)
    return out_p
