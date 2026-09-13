"""Analyze stage: concentration damage, selection gain, and their interpretation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from decodability.evidence import load_freeze_meta
from imbalance_benchmark.common import ensure_dirs, output_root, split_paths, write_json

from breadth import N_SPLITS, exp2_split_paths
from breadth.analyze.diagnostics import _mean_leaves
from breadth.analyze.secondary import pack_estimate

from sites.recall import PathsBySplit, allocation_dirs, contexts, ctx_list
from sites.stages import load_census as load_site_census

from neighbours.accuracy import _subgroup_indices, recall_stack, subgroup_distribution
from neighbours.analyze import _above_threshold, _within_threshold
from neighbours.census import load_census

from composition import ALLOCATIONS, THRESHOLD_PP, exp7_config
from composition.census import _ratios
from composition.secondary import secondary_analysis

__all__ = ["classify", "run_analyze"]

logger = logging.getLogger(__name__)


class Context(NamedTuple):
    """Everything the accuracy fit and the secondary analyses both need."""

    class_names: list[str]
    site_classes: list[str]
    subgroup_idx: dict[str, np.ndarray]
    ctx_l: list[Any]
    paths10: PathsBySplit


class SubgroupFit(NamedTuple):
    """One subgroup's allocation accuracy and the two primary contrasts."""

    a_clustered: np.ndarray
    a_random: np.ndarray
    a_dispersed: np.ndarray
    points_clustered: np.ndarray
    points_random: np.ndarray
    points_dispersed: np.ndarray
    delta_con: np.ndarray
    delta_sel: np.ndarray


def classify(con_ci: tuple[float, float], sel_ci: tuple[float, float]) -> str:
    """Interpretation label (report Table "readings")."""
    if _above_threshold(con_ci) and _above_threshold(sel_ci):
        return "selection_beyond_sampling"
    if _above_threshold(con_ci) and not _above_threshold(sel_ci):
        return "concentration_damage"
    if _within_threshold(con_ci) and not _above_threshold(sel_ci):
        return "no_composition_effect"
    return "inconclusive"


def prepare(config: dict[str, Any]) -> Context:
    """Load context, subgroup class sets, and per-split bootstrap contexts."""
    class_names = list(load_freeze_meta(exp2_split_paths(config, 0))["class_names"])
    site_classes = list(load_site_census(exp7_config(config))["site_classes"])
    subgroup_idx = _subgroup_indices(class_names, site_classes)
    ctx_l = ctx_list(contexts(config))
    paths10 = {s: split_paths(ensure_dirs(config), s) for s in range(N_SPLITS)}
    return Context(class_names, site_classes, subgroup_idx, ctx_l, paths10)


def _subgroup_fit(
    ctx: Context, subgroup: str, stacks: dict[str, np.ndarray]
) -> SubgroupFit:
    idx = ctx.subgroup_idx[subgroup]
    a: dict[str, np.ndarray] = {}
    points: dict[str, np.ndarray] = {}
    for name in ALLOCATIONS:
        a[name], points[name] = subgroup_distribution(stacks[name], idx)
    return SubgroupFit(
        a["clustered"],
        a["random"],
        a["dispersed"],
        points["clustered"],
        points["random"],
        points["dispersed"],
        a["random"] - a["clustered"],
        a["dispersed"] - a["random"],
    )


def fit(config: dict[str, Any], ctx: Context) -> dict[str, SubgroupFit]:
    """Per-allocation recall stacks and, per subgroup, the two primary contrasts."""
    stacks = {
        name: recall_stack(
            allocation_dirs(ctx.paths10, name), ctx.class_names, ctx.ctx_l
        )
        for name in ALLOCATIONS
    }
    return {
        subgroup: _subgroup_fit(ctx, subgroup, stacks) for subgroup in ctx.subgroup_idx
    }


def _allocation_payload(results: dict[str, SubgroupFit]) -> dict[str, Any]:
    """Per-subgroup, per-allocation accuracy and the split-draw contrast spread."""
    payload: dict[str, Any] = {}
    for subgroup, result in results.items():
        points = {
            "clustered": result.points_clustered,
            "random": result.points_random,
            "dispersed": result.points_dispersed,
        }
        dists = {
            "clustered": result.a_clustered,
            "random": result.a_random,
            "dispersed": result.a_dispersed,
        }
        con_draws = points["random"] - points["clustered"]
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
                "delta_con": {
                    "values": con_draws.tolist(),
                    "mean": float(con_draws.mean()),
                    "sd": float(np.std(con_draws)),
                    "n_positive": int((con_draws > 0).sum()),
                },
                "delta_sel": {
                    "values": sel_draws.tolist(),
                    "mean": float(sel_draws.mean()),
                    "sd": float(np.std(sel_draws)),
                    "n_positive": int((sel_draws > 0).sum()),
                },
            },
            "delta_con": pack_estimate(result.delta_con),
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
        arrays[f"{subgroup}_a_clustered"] = result.a_clustered
        arrays[f"{subgroup}_a_random"] = result.a_random
        arrays[f"{subgroup}_a_dispersed"] = result.a_dispersed
        arrays[f"{subgroup}_delta_con"] = result.delta_con
        arrays[f"{subgroup}_delta_sel"] = result.delta_sel
    np.savez(output_root(config) / "data" / "distributions.npz", **arrays)


def _build_results(
    results: dict[str, SubgroupFit],
    census: dict[str, Any],
    ctx: Context,
    secondary: dict[str, Any],
    con_est: dict[str, float],
    sel_est: dict[str, float],
    label: str,
) -> dict[str, Any]:
    """Assemble the analysis.json payload."""
    return {
        "allocations": list(ALLOCATIONS),
        "grid_reference": {"cell": "G20_m8"},
        "census": _census_summary(census),
        "census_by_subgroup": _census_by_subgroup(census, ctx),
        "subgroups": _allocation_payload(results),
        "delta_con": con_est,
        "delta_sel": sel_est,
        "secondary": secondary,
        "interpretation": {"label": label, "threshold_pp": THRESHOLD_PP},
    }


def run_analyze(config: dict[str, Any]) -> Path:
    """Compute the concentration damage, selection gain, and their interpretation."""
    ctx = prepare(config)
    results = fit(config, ctx)
    all_result = results["all"]
    con_est = pack_estimate(all_result.delta_con)
    sel_est = pack_estimate(all_result.delta_sel)
    label = classify(
        (con_est["ci_2_5"], con_est["ci_97_5"]), (sel_est["ci_2_5"], sel_est["ci_97_5"])
    )

    census = load_census(config)
    secondary = secondary_analysis(
        config, ctx.paths10, ctx.class_names, ctx.subgroup_idx
    )
    out = _build_results(results, census, ctx, secondary, con_est, sel_est, label)

    out_p = output_root(config) / "data" / "analysis.json"
    write_json(out_p, out)
    _write_distributions(config, results)
    logger.info("Cohort-composition analysis complete: %s", out_p)
    return out_p
