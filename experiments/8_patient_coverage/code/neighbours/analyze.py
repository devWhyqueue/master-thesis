"""Analyze stage: coverage gain, within-neighbourhood residual, and interpretation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, NamedTuple

import numpy as np
from imbalance_benchmark.common import ensure_dirs, output_root, split_paths, write_json

from breadth import N_SPLITS
from breadth.analyze.secondary import pack_estimate

from neighbours import THRESHOLD_PP
from neighbours.accuracy import SubgroupFit, fit, prepare
from neighbours.census import load_census
from neighbours.secondary import secondary_analysis

__all__ = ["LowArm", "NEIGHBOUR_ARM", "classify", "run_analyze"]

logger = logging.getLogger(__name__)


class LowArm(NamedTuple):
    """The low-coverage arm's allocation name and its report-facing key names."""

    allocation: str
    residual_key: str
    symbol: str


NEIGHBOUR_ARM = LowArm("neighbours", "within_neighbourhood_residual", "b_n")


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


def _allocation_payload(results: dict[str, SubgroupFit], arm: LowArm) -> dict[str, Any]:
    """Per-subgroup, per-allocation accuracy and the split-draw contrast spread."""
    payload: dict[str, Any] = {}
    for subgroup, result in results.items():
        points = {
            "deep": result.points_deep,
            arm.allocation: result.points_low,
            "random": result.points_random,
        }
        dists = {
            "deep": result.a_deep,
            arm.allocation: result.a_low,
            "random": result.a_random,
        }
        first_step = points[arm.allocation] - points["deep"]
        coverage_gain = points["random"] - points[arm.allocation]
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
            arm.symbol: pack_estimate(result.b_low),
            "b_ref": pack_estimate(result.b_ref),
            f"{arm.symbol}_minus_b_ref": pack_estimate(result.b_low - result.b_ref),
        }
    return payload


def _census_summary(census: dict[str, Any]) -> dict[str, Any]:
    """Kappa and site shares, unpacked from the signed census for the report."""
    return {
        s: {key: value for key, value in split.items() if key != "classes"}
        for s, split in census["splits"].items()
    }


def _write_distributions(
    config: dict[str, Any], results: dict[str, Any], arm: LowArm
) -> None:
    """Write the raw paired bootstrap distributions behind analysis.json."""
    arrays: dict[str, Any] = {}
    for subgroup, result in results.items():
        arrays[f"{subgroup}_a_deep"] = result.a_deep
        arrays[f"{subgroup}_a_{arm.allocation}"] = result.a_low
        arrays[f"{subgroup}_a_random"] = result.a_random
        arrays[f"{subgroup}_delta_c"] = result.delta_c
        arrays[f"{subgroup}_{arm.symbol}"] = result.b_low
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
    arm: LowArm,
) -> dict[str, Any]:
    """Assemble the analysis.json payload."""
    minus_ref_key = arm.residual_key.removesuffix("_residual") + "_minus_reference"
    return {
        "allocations": [arm.allocation, "random"],
        "census": _census_summary(census),
        "grid_reference": {"cell": "G20_m8"},
        "subgroups": _allocation_payload(results, arm),
        "coverage_gain": dc_est,
        arm.residual_key: bn_est,
        "reference_residual": pack_estimate(all_result.b_ref),
        minus_ref_key: pack_estimate(all_result.b_low - all_result.b_ref),
        "secondary": secondary,
        "interpretation": {"label": label, "threshold_pp": THRESHOLD_PP},
    }


def run_analyze(
    config: dict[str, Any],
    arm: LowArm = NEIGHBOUR_ARM,
    secondary: Callable[..., dict[str, Any]] = secondary_analysis,
) -> Path:
    """Compute the coverage gain, low-coverage residual, and their interpretation."""
    ctx = prepare(config)
    results = fit(config, ctx, arm.allocation)
    all_result = results["all"]
    dc_est = pack_estimate(all_result.delta_c)
    bn_est = pack_estimate(all_result.b_low)
    label = classify(
        (dc_est["ci_2_5"], dc_est["ci_97_5"]), (bn_est["ci_2_5"], bn_est["ci_97_5"])
    )

    paths8 = {s: split_paths(ensure_dirs(config), s) for s in range(N_SPLITS)}
    secondary_out = secondary(config, paths8, ctx.class_names)
    census = load_census(config)

    out = _build_results(
        all_result, results, census, secondary_out, dc_est, bn_est, label, arm
    )
    out_p = output_root(config) / "data" / "analysis.json"
    write_json(out_p, out)
    _write_distributions(config, results, arm)
    logger.info("Patient-coverage analysis complete: %s", out_p)
    return out_p
