"""Analyze stage (main phase): pool B/P/S/R test accuracy per setting, the D_P/D_S/D_R/I
decomposition, the two primary rescue/susceptibility effects, and their Holm-adjusted
significance once both datasets have finished.
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

from centre import N_DRAWS as NATIVE_DRAWS
from centre.analyze import pooled

from spectrum import baseline_config

from imbalance_benchmark.analysis.inference.confirmatory.holm import holm_adjust_pvalues

from separation import ARMS, MAIN_DRAWS, N_SPLITS, SEVERITY, main_settings

__all__ = ["condition_accuracy", "native_accuracy", "combine", "run_analyze"]

_PRIMARY_KEY = {
    "bracs": "primary_bracs_rescue",
    "tcga_ut": "primary_tcga_susceptibility",
}


def condition_accuracy(
    config: dict[str, Any],
    names: list[str],
    settings: tuple[str, ...],
    draws: tuple[int, ...],
) -> dict[str, np.ndarray]:
    """Per (setting, arm), (F, R) class-mean patient-macro recall in percent, F = splits x draws."""
    ctxs = bootstrap_contexts(config)
    paths = {s: split_paths(ensure_dirs(config), s) for s in range(N_SPLITS)}
    perms = {s: canonical_permutation(config, s, names) for s in range(N_SPLITS)}
    keys = [(s, d) for s in range(N_SPLITS) for d in draws]
    ctx_l = [ctxs[s] for s, _ in keys]
    perm_l = [perms[s] for s, _ in keys]
    out: dict[str, np.ndarray] = {}
    for setting in settings:
        for arm in ARMS:
            dirs = [allocation_dir(paths[s], f"{setting}_{arm}", d) for s, d in keys]
            out[f"{setting}_{arm}"] = (
                recall_stack(dirs, ctx_l, perm_l, len(names)).mean(axis=1) * 100.0
            )
    return out


def native_accuracy(config: dict[str, Any], names: list[str]) -> dict[str, np.ndarray]:
    """This dataset's own reused native B/R (exp-25/26) and P/S (exp-27/28) arms, 0 new fits."""
    prevalence_cfg = baseline_config(config, "native_prevalence_outputs")
    cause_cfg = baseline_config(config, "native_cause_outputs")
    ctxs = bootstrap_contexts(config)
    perms = {s: canonical_permutation(config, s, names) for s in range(N_SPLITS)}
    keys = [(s, d) for s in range(N_SPLITS) for d in range(NATIVE_DRAWS)]
    ctx_l = [ctxs[s] for s, _ in keys]
    perm_l = [perms[s] for s, _ in keys]

    def _read(cfg: dict[str, Any], arm: str) -> np.ndarray:
        paths = {s: split_paths(ensure_dirs(cfg), s) for s in range(N_SPLITS)}
        dirs = [allocation_dir(paths[s], arm, d) for s, d in keys]
        return recall_stack(dirs, ctx_l, perm_l, len(names)).mean(axis=1) * 100.0

    return {
        "native_B": _read(prevalence_cfg, "r1"),
        "native_R": _read(prevalence_cfg, f"r{SEVERITY}"),
        "native_P": _read(cause_cfg, f"P{SEVERITY}"),
        "native_S": _read(cause_cfg, f"S{SEVERITY}"),
    }


def combine(
    ba: dict[str, np.ndarray], settings: tuple[str, ...]
) -> dict[str, np.ndarray]:
    """D_P, D_S, D_R, and the interaction I = D_R - D_P - D_S, per setting."""
    dists = {f"arm_{k}": v for k, v in ba.items()}
    for setting in settings:
        b, p, s, r = (ba[f"{setting}_{arm}"] for arm in ARMS)
        d_p, d_s, d_r = b - p, b - s, b - r
        dists[f"D_P_{setting}"], dists[f"D_S_{setting}"], dists[f"D_R_{setting}"] = (
            d_p,
            d_s,
            d_r,
        )
        dists[f"I_{setting}"] = d_r - d_p - d_s
    return dists


def _one_sided_p(dist: np.ndarray) -> float:
    """One-sided bootstrap p-value: replicate share at or below zero (effect predicted positive)."""
    return float(np.mean(dist[1:] <= 0.0))


def _peer_primary(config: dict[str, Any]) -> np.ndarray | None:
    """The other dataset's own primary-effect distribution, once its main analysis has run."""
    if not config.get("slurm", {}).get("peer_outputs"):
        return None
    peer_config = baseline_config(config, "peer_outputs")
    npz_path = output_root(peer_config) / "data" / "distributions.npz"
    if not npz_path.exists():
        return None
    key = _PRIMARY_KEY["tcga_ut" if config["dataset"]["name"] == "bracs" else "bracs"]
    with np.load(npz_path) as npz:
        if key not in npz:
            return None
        return np.asarray(npz[key])


def _pooled_ba(
    config: dict[str, Any], names: list[str], settings: tuple[str, ...]
) -> dict[str, np.ndarray]:
    """Draw-weighted pooled B/P/S/R accuracy per setting, over this dataset's main-phase fits."""
    acc = condition_accuracy(config, names, settings, MAIN_DRAWS)
    n_replicates = next(iter(acc.values())).shape[-1]
    n_draws = len(MAIN_DRAWS)
    fit_split = np.repeat(np.arange(N_SPLITS), n_draws)
    w = draw_weights(
        fit_split, n_draws, n_replicates, np.random.default_rng(BOOTSTRAP_SEED)
    )
    return {name: pooled(a, w) for name, a in acc.items()}


def _primary_effect(
    dataset: str, dists: dict[str, np.ndarray], native: dict[str, np.ndarray]
) -> np.ndarray:
    """BRACS: native minus expanded D_R (rescue). TCGA-UT: contracted minus native D_R."""
    if dataset == "bracs":
        return (native["native_B"] - native["native_R"]) - dists["D_R_expanded_bracs"]
    return dists["D_R_tcga_contracted_10"] - dists["D_R_tcga_native_10"]


def _holm_diagnostics(config: dict[str, Any], own_dist: np.ndarray) -> dict[str, Any]:
    """Holm-adjusted one-sided p-values for the two primary effects, once the peer is ready."""
    peer_primary = _peer_primary(config)
    if peer_primary is None:
        return {}
    own_p, peer_p = _one_sided_p(own_dist), _one_sided_p(peer_primary)
    adj = holm_adjust_pvalues([own_p, peer_p])
    return {
        "holm": {
            "own_one_sided_p": own_p,
            "peer_one_sided_p": peer_p,
            "own_adjusted_p": adj[0],
            "peer_adjusted_p": adj[1],
        }
    }


def run_analyze(config: dict[str, Any]) -> Path:
    """Pool this dataset's new settings, compute its primary effect, and Holm-correct once paired."""
    names = canonical_class_names(config)
    dataset = config["dataset"]["name"]
    settings = main_settings(dataset)
    ba = _pooled_ba(config, names, settings)
    dists = combine(ba, settings)
    primary_key = _PRIMARY_KEY[dataset]
    dists[primary_key] = _primary_effect(dataset, dists, native_accuracy(config, names))
    diagnostics = _holm_diagnostics(config, dists[primary_key])

    out_p = output_root(config) / "data" / "analysis.json"
    write_json(
        out_p,
        {"estimates": {k: pack_estimate(v) for k, v in dists.items()}, **diagnostics},
    )
    np.savez(output_root(config) / "data" / "distributions.npz", **cast(Any, dists))
    return out_p
