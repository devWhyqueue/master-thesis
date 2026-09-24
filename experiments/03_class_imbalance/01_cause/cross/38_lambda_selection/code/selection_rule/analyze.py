"""Shared analyze math: pooled per-(rule, arm) accuracy, D_P (primary channel) and D_S (secondary,
out of scope) per rule, and the cross-dataset gap/Delta/closure that both the phase-1 diagnostic
(draws 0-1) and the phase-2 confirmatory analysis (draws 2-9) compute identically.
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

from selection_rule import (
    ARMS,
    CLOSURE_GATE,
    LABEL_NO_PRIOR_GAP,
    LABEL_PRIOR_GAP_INTRINSIC,
    LABEL_SELECTION_CONTRIBUTES,
    LABEL_SELECTION_EXPLAINS,
    MAIN_DRAWS,
    N_SPLITS,
    RULES,
)

__all__ = [
    "condition_accuracy",
    "pooled_ba",
    "combine",
    "simultaneous_interval",
    "gap_closure",
    "run_analyze",
]

SIMULTANEOUS_ALPHA: float = 0.05

_KEYS: tuple[str, ...] = tuple(f"{rule}_{arm}" for rule in RULES for arm in ARMS)


def condition_accuracy(
    config: dict[str, Any], names: list[str], draws: tuple[int, ...]
) -> dict[str, np.ndarray]:
    """Per (rule, arm), (F, R) class-mean patient-macro recall in percent, F = splits x draws."""
    ctxs = bootstrap_contexts(config)
    paths = {s: split_paths(ensure_dirs(config), s) for s in range(N_SPLITS)}
    perms = {s: canonical_permutation(config, s, names) for s in range(N_SPLITS)}
    keys = [(s, d) for s in range(N_SPLITS) for d in draws]
    ctx_l = [ctxs[s] for s, _ in keys]
    perm_l = [perms[s] for s, _ in keys]
    out: dict[str, np.ndarray] = {}
    for key in _KEYS:
        dirs = [allocation_dir(paths[s], key, d) for s, d in keys]
        out[key] = recall_stack(dirs, ctx_l, perm_l, len(names)).mean(axis=1) * 100.0
    return out


def pooled_ba(
    config: dict[str, Any], names: list[str], draws: tuple[int, ...]
) -> dict[str, np.ndarray]:
    """Draw-weighted pooled per-(rule, arm) accuracy, over ``draws``'s fits."""
    acc = condition_accuracy(config, names, draws)
    n_replicates = next(iter(acc.values())).shape[-1]
    fit_split = np.repeat(np.arange(N_SPLITS), len(draws))
    w = draw_weights(
        fit_split, len(draws), n_replicates, np.random.default_rng(BOOTSTRAP_SEED)
    )
    return {k: (w * a).sum(axis=0) / w.sum(axis=0) for k, a in acc.items()}


def combine(ba: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """D_P (primary) and D_S (secondary) per rule, plus the P-only-oracle descriptive variant."""
    dists = dict(ba)
    for rule in RULES:
        dists[f"D_P_{rule}"] = ba[f"{rule}_B"] - ba[f"{rule}_P"]
        dists[f"D_S_{rule}"] = ba[f"{rule}_B"] - ba[f"{rule}_S"]
    dists["D_P_oracle_p_only"] = ba["tuned_B"] - ba["oracle_P"]
    return dists


def simultaneous_interval(dist: np.ndarray, n_estimates: int = 1) -> dict[str, float]:
    """Bonferroni-adjusted percentile CI for one of ``n_estimates`` simultaneous estimates."""
    half_alpha_pct = 100.0 * (SIMULTANEOUS_ALPHA / (2 * n_estimates))
    replicates = dist[1:] if len(dist) > 1 else dist
    return {
        "point": float(dist[0]),
        "ci_lower": float(np.nanpercentile(replicates, half_alpha_pct)),
        "ci_upper": float(np.nanpercentile(replicates, 100.0 - half_alpha_pct)),
    }


def _label(gap_tuned: dict[str, float], delta: dict[str, float], closure: float) -> str:
    """Precondition labels: closure is meaningless without a tuned gap (PLAN.md "Labels")."""
    if gap_tuned["ci_lower"] <= 0.0 <= gap_tuned["ci_upper"]:
        return LABEL_NO_PRIOR_GAP
    if delta["ci_lower"] <= 0.0 <= delta["ci_upper"]:
        return LABEL_PRIOR_GAP_INTRINSIC
    return (
        LABEL_SELECTION_EXPLAINS
        if closure >= CLOSURE_GATE
        else LABEL_SELECTION_CONTRIBUTES
    )


def gap_closure(
    dists: dict[str, np.ndarray], peer: dict[str, np.ndarray], rule: str
) -> dict[str, Any]:
    """Gap = D_P(this) - D_P(peer) under ``tuned``; Delta = tuned gap - ``rule`` gap; closure."""
    gap_tuned = dists["D_P_tuned"] - peer["D_P_tuned"]
    gap_rule = dists[f"D_P_{rule}"] - peer[f"D_P_{rule}"]
    delta = gap_tuned - gap_rule
    gap_tuned_est = simultaneous_interval(gap_tuned)
    delta_est = simultaneous_interval(delta)
    closure = (
        float(delta_est["point"] / gap_tuned_est["point"])
        if gap_tuned_est["point"]
        else 0.0
    )
    return {
        "gap_tuned": gap_tuned,
        f"gap_{rule}": gap_rule,
        "delta": delta,
        "gap_tuned_estimate": gap_tuned_est,
        "delta_estimate": delta_est,
        "closure": closure,
        "closure_descriptive_ci": pack_estimate(
            np.divide(
                delta, gap_tuned, out=np.full_like(delta, np.nan), where=gap_tuned != 0
            )
        ),
        "label": _label(gap_tuned_est, delta_est, closure),
    }


def _peer_dists(config: dict[str, Any]) -> dict[str, np.ndarray] | None:
    """TCGA-UT's own main-phase D_P/D_S distributions, once its analyze stage has run."""
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
    dists: dict[str, np.ndarray], estimates: dict[str, Any], peer: dict[str, np.ndarray]
) -> str:
    """BRACS only: gap/Delta/closure under both oracle rules, and the pre-registered label."""
    cross_fit = gap_closure(dists, peer, "oracle")
    naive = gap_closure(dists, peer, "naive")
    estimates["gap_tuned"] = cross_fit["gap_tuned_estimate"]
    estimates["delta_cross_fit"] = cross_fit["delta_estimate"]
    estimates["closure_cross_fit"] = cross_fit["closure"]
    estimates["closure_cross_fit_descriptive_ci"] = cross_fit["closure_descriptive_ci"]
    estimates["delta_naive"] = naive["delta_estimate"]
    estimates["closure_naive"] = naive["closure"]
    dists["gap_tuned"] = cross_fit["gap_tuned"]
    dists["delta_cross_fit"] = cross_fit["delta"]
    dists["delta_naive"] = naive["delta"]
    return cross_fit["label"]


def run_analyze(config: dict[str, Any]) -> Path:
    """Pool this dataset's per-rule B/P/S/R accuracy; BRACS additionally computes gap/Delta/closure
    and the pre-registered label, once TCGA-UT's own main-phase analysis is ready."""
    names = canonical_class_names(config)
    dists = combine(pooled_ba(config, names, MAIN_DRAWS))
    estimates: dict[str, Any] = {k: pack_estimate(v) for k, v in dists.items()}
    peer = _peer_dists(config)
    label = (
        _add_primary(dists, estimates, peer)
        if config["dataset"]["name"] == "bracs" and peer is not None
        else None
    )

    out_p = output_root(config) / "data" / "analysis.json"
    write_json(out_p, {"estimates": estimates, "label": label})
    np.savez(output_root(config) / "data" / "distributions.npz", **cast(Any, dists))
    return out_p
