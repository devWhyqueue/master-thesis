"""Analyze stage: selection gain, its interpretation, and four secondary analyses."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from imbalance_benchmark.common import output_root, write_json

from breadth import N_SPLITS
from breadth.analyze.diagnostics import _mean_leaves
from breadth.analyze.secondary import pack_estimate

from sites.recall import allocation_dirs

from neighbours.accuracy import recall_stack, subgroup_distribution
from neighbours.analyze import _above_threshold, _within_threshold
from neighbours.census import load_census

from coverage_redundancy.census import load_quantities

from composition.analyze import Context, prepare

from shortage import ALLOCATIONS, THRESHOLD_PP
from shortage.census import _ratios
from shortage.grid import log_split0_check, recovered_share
from shortage.relation import relation_secondary
from shortage.secondary import secondary_analysis

__all__ = ["classify", "run_analyze"]

logger = logging.getLogger(__name__)


class SubgroupFit(NamedTuple):
    """One subgroup's allocation accuracy and the selection-gain contrast."""

    a_random: np.ndarray
    a_dispersed: np.ndarray
    points_random: np.ndarray
    points_dispersed: np.ndarray
    delta_sel: np.ndarray


def classify(sel_ci: tuple[float, float]) -> str:
    """Interpretation label (report Table "readings")."""
    if _above_threshold(sel_ci):
        return "selection_mitigates_shortage"
    if _within_threshold(sel_ci):
        return "no_practical_gain"
    if sel_ci[1] < -THRESHOLD_PP:
        return "selection_harms"
    return "inconclusive"


def _subgroup_fit(
    ctx: Context, subgroup: str, stacks: dict[str, np.ndarray]
) -> SubgroupFit:
    idx = ctx.subgroup_idx[subgroup]
    a: dict[str, np.ndarray] = {}
    points: dict[str, np.ndarray] = {}
    for name in ALLOCATIONS:
        a[name], points[name] = subgroup_distribution(stacks[name], idx)
    return SubgroupFit(
        a["random"],
        a["dispersed"],
        points["random"],
        points["dispersed"],
        a["dispersed"] - a["random"],
    )


def fit(ctx: Context) -> dict[str, SubgroupFit]:
    """Per-allocation recall stacks and, per subgroup, the selection-gain contrast."""
    n_classes = len(ctx.class_names)
    stacks = {
        name: recall_stack(
            allocation_dirs(ctx.paths10, name), ctx.ctx_l, ctx.perms, n_classes
        )
        for name in ALLOCATIONS
    }
    return {
        subgroup: _subgroup_fit(ctx, subgroup, stacks) for subgroup in ctx.subgroup_idx
    }


def _subgroup_payload(results: dict[str, SubgroupFit]) -> dict[str, Any]:
    """Per-subgroup, per-allocation accuracy and the split-draw contrast spread."""
    payload: dict[str, Any] = {}
    for subgroup, result in results.items():
        points = {"random": result.points_random, "dispersed": result.points_dispersed}
        dists = {"random": result.a_random, "dispersed": result.a_dispersed}
        sel_draws = points["dispersed"] - points["random"]
        payload[subgroup] = {
            "accuracy": {
                name: {
                    **pack_estimate(dists[name]),
                    "draw_dispersion": float(np.std(points[name])),
                }
                for name in dists
            },
            "draw_contrasts": {
                "delta_sel": {
                    "values": sel_draws.tolist(),
                    "mean": float(sel_draws.mean()),
                    "sd": float(np.std(sel_draws)),
                    "n_positive": int((sel_draws > 0).sum()),
                },
            },
            "delta_sel": pack_estimate(result.delta_sel),
        }
    return payload


def _census_summary(census: dict[str, Any]) -> dict[str, Any]:
    """Coverage ratios and site shares, unpacked from the signed census for the report."""
    return {
        s: {key: value for key, value in split.items() if key != "classes"}
        for s, split in census["splits"].items()
    }


def _census_by_subgroup(census: dict[str, Any], ctx: Context) -> dict[str, Any]:
    """Coverage ratios, spreads, and site shares averaged over each subgroup's classes."""
    out: dict[str, Any] = {}
    for subgroup, idx in ctx.subgroup_idx.items():
        names = {ctx.class_names[i] for i in idx}
        rows = [
            census["splits"][str(s)]["classes"][name]
            for s in range(N_SPLITS)
            for name in names
        ]
        summary = _mean_leaves(rows)
        summary.update(_ratios(summary))
        out[subgroup] = summary
    return out


def _write_distributions(
    config: dict[str, Any], results: dict[str, SubgroupFit]
) -> None:
    """Write the raw paired bootstrap distributions behind analysis.json."""
    arrays: dict[str, Any] = {}
    for subgroup, result in results.items():
        arrays[f"{subgroup}_a_random"] = result.a_random
        arrays[f"{subgroup}_a_dispersed"] = result.a_dispersed
        arrays[f"{subgroup}_delta_sel"] = result.delta_sel
    np.savez(output_root(config) / "data" / "distributions.npz", **arrays)


def _compute_secondary(
    config: dict[str, Any],
    ctx: Context,
    results: dict[str, SubgroupFit],
    quantities: dict[str, Any],
) -> dict[str, Any]:
    """Run the four secondary analyses and pack them for analysis.json."""
    delta_sel_by_subgroup = {s: r.delta_sel for s, r in results.items()}
    recovered, points_5_32_all = recovered_share(config, ctx, delta_sel_by_subgroup)
    log_split0_check(results["all"].points_random, points_5_32_all)
    tertiles = secondary_analysis(
        config, ctx.paths10, ctx.class_names, ctx.subgroup_idx
    )
    relation = relation_secondary(config, quantities, results["all"].delta_sel, ctx)
    return {
        "recovered_share": recovered,
        "tertiles": tertiles["tertiles"],
        "relation": relation,
    }


def _build_output(
    ctx: Context,
    results: dict[str, SubgroupFit],
    census: dict[str, Any],
    secondary: dict[str, Any],
    sel_est: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    """Assemble the analysis.json payload."""
    return {
        "allocations": list(ALLOCATIONS),
        "grid_reference": {"cell": "G5_m32"},
        "census": _census_summary(census),
        "census_by_subgroup": _census_by_subgroup(census, ctx),
        "subgroups": _subgroup_payload(results),
        "delta_sel": sel_est,
        "secondary": secondary,
        "interpretation": {"label": label, "threshold_pp": THRESHOLD_PP},
    }


def run_analyze(config: dict[str, Any]) -> Path:
    """Compute the selection gain, its interpretation, and the four secondary analyses."""
    ctx = prepare(config)
    results = fit(ctx)
    sel_est = pack_estimate(results["all"].delta_sel)
    label = classify((sel_est["ci_2_5"], sel_est["ci_97_5"]))

    census = load_census(config)
    quantities = load_quantities(config)
    secondary = _compute_secondary(config, ctx, results, quantities)

    out = _build_output(ctx, results, census, secondary, sel_est, label)
    out_p = output_root(config) / "data" / "analysis.json"
    write_json(out_p, out)
    _write_distributions(config, results)
    logger.info("Shortage-selection analysis complete: %s", out_p)
    return out_p
