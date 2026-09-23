"""Analyze stage (main phase): pool every setting's B/P/S/R test accuracy, the D_P/D_S/D_R/I
channel decomposition, the two primary estimates, and their simultaneous 95% intervals.

Primary 2 ("Reduction in the matched-budget cross-dataset damage gap when BRACS receives the
joint intervention and TCGA-UT remains native", PLAN.md line 82) telescopes to primary 1 by
construction -- TCGA-UT's own native D_R distribution appears unchanged in both the pre- and
post-intervention gap, at the same paired replicate, so it cancels exactly. That is expected, not
a bug: it is a within-dataset/cross-dataset consistency cross-check on the same rescue, not an
independent second effect.
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

from joint import ARMS, MAIN_DRAWS, N_SPLITS, SETTINGS, SIMULTANEOUS_ALPHA

__all__ = ["condition_accuracy", "combine", "simultaneous_interval", "run_analyze"]

_KEYS: tuple[str, ...] = tuple(
    f"{setting}_{arm}" for setting in SETTINGS for arm in ARMS
)


def condition_accuracy(
    config: dict[str, Any], names: list[str]
) -> dict[str, np.ndarray]:
    """Per (setting, arm), (F, R) class-mean patient-macro recall in percent, F = splits x draws."""
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
    """Draw-weighted pooled B/P/S/R accuracy per setting, over this dataset's main-phase fits."""
    acc = condition_accuracy(config, names)
    n_replicates = next(iter(acc.values())).shape[-1]
    n_draws = len(MAIN_DRAWS)
    fit_split = np.repeat(np.arange(N_SPLITS), n_draws)
    w = draw_weights(
        fit_split, n_draws, n_replicates, np.random.default_rng(BOOTSTRAP_SEED)
    )
    return {name: (w * a).sum(axis=0) / w.sum(axis=0) for name, a in acc.items()}


def combine(ba: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """D_P, D_S, D_R, and the interaction I = D_R - D_P - D_S, per setting."""
    dists = {f"arm_{k}": v for k, v in ba.items()}
    for setting in SETTINGS:
        b, p, s, r = (ba[f"{setting}_{arm}"] for arm in ARMS)
        d_p, d_s, d_r = b - p, b - s, b - r
        dists[f"D_P_{setting}"], dists[f"D_S_{setting}"], dists[f"D_R_{setting}"] = (
            d_p,
            d_s,
            d_r,
        )
        dists[f"I_{setting}"] = d_r - d_p - d_s
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


def _peer_native_d_r(config: dict[str, Any]) -> np.ndarray | None:
    """TCGA-UT's own native D_R distribution, once its main analysis has run (PLAN.md primary 2)."""
    if config["dataset"]["name"] != "bracs" or not config.get("slurm", {}).get(
        "peer_outputs"
    ):
        return None
    peer_config = {**config, "paths": {"outputs": str(config["slurm"]["peer_outputs"])}}
    npz_path = output_root(peer_config) / "data" / "distributions.npz"
    if not npz_path.exists():
        return None
    with np.load(npz_path) as npz:
        if "D_R_native" not in npz:
            return None
        return np.asarray(npz["D_R_native"])


def _add_primary_estimates(
    config: dict[str, Any],
    dists: dict[str, np.ndarray],
    estimates: dict[str, dict[str, float]],
) -> None:
    """BRACS only: the joint rescue, and the cross-dataset gap reduction once TCGA-UT is ready."""
    if config["dataset"]["name"] != "bracs":
        return
    dists["primary_bracs_rescue"] = dists["D_R_native"] - dists["D_R_joint"]
    estimates["primary_bracs_rescue"] = simultaneous_interval(
        dists["primary_bracs_rescue"]
    )
    peer_native = _peer_native_d_r(config)
    if peer_native is not None:
        gap_before = dists["D_R_native"] - peer_native
        gap_after = dists["D_R_joint"] - peer_native
        dists["primary_gap_reduction"] = gap_before - gap_after
        estimates["primary_gap_reduction"] = simultaneous_interval(
            dists["primary_gap_reduction"]
        )


def run_analyze(config: dict[str, Any]) -> Path:
    """Pool this dataset's settings, compute channel decomposition and any primary estimates."""
    names = canonical_class_names(config)
    dists = combine(_pooled_ba(config, names))
    estimates: dict[str, dict[str, float]] = {
        k: pack_estimate(v) for k, v in dists.items()
    }
    _add_primary_estimates(config, dists, estimates)

    out_p = output_root(config) / "data" / "analysis.json"
    write_json(out_p, {"estimates": estimates})
    np.savez(output_root(config) / "data" / "distributions.npz", **cast(Any, dists))
    return out_p
