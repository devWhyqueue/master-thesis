"""Analyze stage (only after gates 0-2 pass): pool B and every S arm's fixed-lambda test accuracy,
D_S per arm, the coverage rescue R_d = D_S(random) - D_S(coverage), and -- BRACS only, once TCGA-UT
is ready -- the two primary estimates and the pre-registered label (PLAN.md "Primary analysis",
"Labels").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
from imbalance_benchmark.common import ensure_dirs, output_root, split_paths, write_json

from breadth import BOOTSTRAP_SEED
from breadth.analyze.canonical import canonical_class_names, canonical_permutation
from breadth.analyze.secondary import pack_estimate

from sites import allocation_dir
from sites.recall import contexts as bootstrap_contexts

from neighbours.accuracy import recall_stack

from decomposition.model import draw_weights

from support import LABEL_RESCUE_HALF, MAIN_DRAWS, N_SPLITS, S_ARMS

__all__ = ["condition_accuracy", "combine", "simultaneous_interval", "run_analyze"]

SIMULTANEOUS_ALPHA: float = (
    0.05  # Bonferroni-adjusted CI for R_BRACS and R_BRACS - R_TCGA
)

_KEYS: tuple[str, ...] = ("B", *(f"{arm}_fixedlambda" for arm in S_ARMS))


def condition_accuracy(
    config: dict[str, Any], names: list[str]
) -> dict[str, np.ndarray]:
    """Per arm, (F, R) class-mean patient-macro recall in percent, F = splits x draws."""
    ctxs = bootstrap_contexts(config)
    paths = {s: split_paths(ensure_dirs(config), s) for s in range(N_SPLITS)}
    perms = {s: canonical_permutation(config, s, names) for s in range(N_SPLITS)}
    keys = [(s, d) for s in range(N_SPLITS) for d in MAIN_DRAWS]
    ctx_l = [ctxs[s] for s, _ in keys]
    perm_l = [perms[s] for s, _ in keys]
    out: dict[str, np.ndarray] = {}
    for key in _KEYS:
        dirs = [allocation_dir(paths[s], key, d) for s, d in keys]
        out[key] = recall_stack(dirs, ctx_l, perm_l, len(names)).mean(axis=1) * 100.0
    return out


def _pooled_ba(config: dict[str, Any], names: list[str]) -> dict[str, np.ndarray]:
    """Draw-weighted pooled B/S-arm accuracy, over this dataset's fixed-lambda main-phase fits."""
    acc = condition_accuracy(config, names)
    n_replicates = next(iter(acc.values())).shape[-1]
    n_draws = len(MAIN_DRAWS)
    fit_split = np.repeat(np.arange(N_SPLITS), n_draws)
    w = draw_weights(
        fit_split, n_draws, n_replicates, np.random.default_rng(BOOTSTRAP_SEED)
    )
    return {name: (w * a).sum(axis=0) / w.sum(axis=0) for name, a in acc.items()}


def combine(ba: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """B, D_S per S arm, and the coverage rescue R_d = D_S(random) - D_S(coverage)."""
    b = ba["B"]
    dists = {"arm_B": b}
    for arm in S_ARMS:
        s = ba[f"{arm}_fixedlambda"]
        dists[f"arm_{arm}"] = s
        dists[f"D_S_{arm}"] = b - s
    dists["rescue_R"] = dists["D_S_random"] - dists["D_S_coverage"]
    return dists


def simultaneous_interval(dist: np.ndarray, n_estimates: int = 2) -> dict[str, float]:
    """Bonferroni-adjusted percentile CI for one of ``n_estimates`` simultaneous estimates."""
    half_alpha_pct = 100.0 * (SIMULTANEOUS_ALPHA / (2 * n_estimates))
    replicates = dist[1:] if len(dist) > 1 else dist
    return {
        "point": float(dist[0]),
        "ci_lower": float(np.nanpercentile(replicates, half_alpha_pct)),
        "ci_upper": float(np.nanpercentile(replicates, 100.0 - half_alpha_pct)),
    }


def _positive(estimate: dict[str, float]) -> bool:
    return estimate["point"] > 0 and estimate["ci_lower"] > 0


def _peer_dists(config: dict[str, Any]) -> dict[str, np.ndarray] | None:
    """TCGA-UT's own D_S/rescue distributions, once its main analysis has run (PLAN.md "Labels")."""
    if config["dataset"]["name"] != "bracs" or not config.get("slurm", {}).get(
        "peer_outputs"
    ):
        return None
    peer_config = {**config, "paths": {"outputs": str(config["slurm"]["peer_outputs"])}}
    npz_path = output_root(peer_config) / "data" / "distributions.npz"
    if not npz_path.exists():
        return None
    with np.load(npz_path) as npz:
        return {k: np.asarray(npz[k]) for k in npz.files}


def _add_primary(
    config: dict[str, Any],
    dists: dict[str, np.ndarray],
    estimates: dict[str, dict[str, float]],
) -> str | None:
    """BRACS only: the two primary estimates (simultaneous CI) and the pre-registered label."""
    if config["dataset"]["name"] != "bracs":
        return None
    estimates["primary_rescue_bracs"] = simultaneous_interval(dists["rescue_R"])
    peer = _peer_dists(config)
    if peer is None:
        return None
    g_random = dists["D_S_random"] - peer["D_S_random"]
    g_coverage = dists["D_S_coverage"] - peer["D_S_coverage"]
    rescue_diff = dists["rescue_R"] - peer["rescue_R"]
    dists["primary_rescue_diff_bracs_minus_tcga"] = rescue_diff
    dists["gap_random"] = g_random
    dists["gap_coverage"] = g_coverage
    estimates["primary_rescue_diff_bracs_minus_tcga"] = simultaneous_interval(
        rescue_diff
    )
    estimates["gap_random"] = pack_estimate(g_random)
    estimates["gap_coverage"] = pack_estimate(g_coverage)
    r_bracs_positive = _positive(estimates["primary_rescue_bracs"])
    r_diff_positive = _positive(estimates["primary_rescue_diff_bracs_minus_tcga"])
    if not (r_bracs_positive and r_diff_positive):
        return "coverage_not_supported"
    if float(g_coverage[0]) <= LABEL_RESCUE_HALF * float(g_random[0]):
        return "coverage_explains"
    return "coverage_contributes"


def run_analyze(config: dict[str, Any]) -> Path:
    """Pool this dataset's B/S-arm accuracy, compute D_S/rescue and any primary estimates/label."""
    names = canonical_class_names(config)
    dists = combine(_pooled_ba(config, names))
    estimates: dict[str, dict[str, float]] = {
        k: pack_estimate(v) for k, v in dists.items()
    }
    label = _add_primary(config, dists, estimates)

    out_p = output_root(config) / "data" / "analysis.json"
    write_json(out_p, {"estimates": estimates, "label": label})
    np.savez(output_root(config) / "data" / "distributions.npz", **cast(Any, dists))
    return out_p
