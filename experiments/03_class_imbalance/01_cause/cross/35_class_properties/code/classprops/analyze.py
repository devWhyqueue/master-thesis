"""Analyze stage: pool both datasets' extracted observations, fit M0/M1, and write the shrink
share, dataset residual G, and outcome label.

Reads the peer dataset's own ``extract.npz`` through ``slurm.peer_outputs`` (exp-33's idiom): the
two per-dataset submissions can run in either order, but only the one that runs second finds the
peer ready and actually writes the joint result (into its own output tree; both write the same
numbers, so either dataset's ``analysis.json`` is authoritative once both have run).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.common import ensure_dirs, write_json

from classprops import POOL_KINDS, RATIOS
from classprops.model import (
    DatasetCovariates,
    fit_pool,
    gather_covariates,
    label_outcome,
)
from classprops.pool import Observations, observations_from_arrays
from classprops.sensitivities import sensitivity_a, sensitivity_b, sensitivity_c

__all__ = ["run_analyze"]

logger = logging.getLogger(__name__)

PoolFits = tuple[Observations, DatasetCovariates, Observations, DatasetCovariates]
DatasetBundle = tuple[dict[str, np.ndarray], dict[str, Any]]

# exp-25/26/29's own stored analysis.json point estimates (BA %, split-0 canonical order),
# reproduced here as fixed sanity targets rather than re-read from another experiment's output
# tree at analyze time (this stage's own r1/r100 fits are read fresh from the same stored runs).
_SANITY_D_R100 = {"tcga_ut": 2.914964923, "bracs": 7.649701816}
_SANITY_SHIFT_BA_R100 = {
    "a1_r100": 31.163426361843204,
    "a2_r100": 31.84955125121279,
    "a3_r100": 32.095520314468146,
    "a4_r100": 32.213020280428864,
    "a5_r100": 31.646607941496548,
    "a6_r100": 30.993426843820654,
}
_SANITY_TOL_PP = 0.05
_SANITY_SHIFT_TOL_PP = 0.01


def _peer_extract_path(config: dict[str, Any]) -> Path | None:
    """The peer dataset's own ``data/extract.npz``, if that side has already run ``extract``."""
    peer_key = config.get("slurm", {}).get("peer_outputs")
    if not peer_key:
        return None
    path = Path(peer_key) / "data" / "extract.npz"
    return path if path.exists() else None


def _load_extract(data_dir: Path) -> DatasetBundle:
    with np.load(data_dir / "extract.npz") as npz:
        arrays = {k: npz[k] for k in npz.files}
    meta = json.loads((data_dir / "extract_meta.json").read_text(encoding="utf-8"))
    return arrays, meta


def _load_both_datasets(
    config: dict[str, Any], data_dir: Path
) -> tuple[DatasetBundle, DatasetBundle] | None:
    """(tcga, bracs) extract bundles, this dataset's own plus the peer's; None if peer not ready."""
    own = _load_extract(data_dir)
    peer_path = _peer_extract_path(config)
    if peer_path is None:
        return None
    peer = _load_extract(peer_path.parent)
    return (own, peer) if config["dataset"]["name"] == "tcga_ut" else (peer, own)


def _check_sanity(
    tcga_meta: dict[str, Any], bracs_meta: dict[str, Any]
) -> dict[str, Any]:
    """Assert this run's own r1/r100 BA reproduces exp-25/26/29's already-published values."""
    out: dict[str, Any] = {}
    for dataset, meta in (("tcga_ut", tcga_meta), ("bracs", bracs_meta)):
        sanity = meta["sanity"]
        d_r100 = sanity["ba_r1"] - sanity["ba_by_arm_r100"]["r100"]
        out[f"D_r100_{dataset}"] = d_r100
        logger.info(
            "Sanity: %s D(r100) = %.4f (published %.4f)",
            dataset,
            d_r100,
            _SANITY_D_R100[dataset],
        )
        assert abs(d_r100 - _SANITY_D_R100[dataset]) < _SANITY_TOL_PP, (
            f"{dataset} D(r100) sanity check failed: {d_r100} vs {_SANITY_D_R100[dataset]}"
        )
    for arm, published in _SANITY_SHIFT_BA_R100.items():
        observed = bracs_meta["sanity"]["ba_by_arm_r100"][arm]
        logger.info(
            "Sanity: bracs %s BA = %.4f (published %.4f)", arm, observed, published
        )
        assert abs(observed - published) < _SANITY_SHIFT_TOL_PP, (
            f"bracs {arm} sanity check failed: {observed} vs {published}"
        )
    return out


def _fit_all_pools(
    tcga_arrays: dict[str, np.ndarray], bracs_arrays: dict[str, np.ndarray]
) -> tuple[dict[str, Any], dict[str, PoolFits]]:
    """Fit M0/M1 for every (pool, rho) combination."""
    results: dict[str, Any] = {}
    covariates_by_pool: dict[str, PoolFits] = {}
    for pool in POOL_KINDS:
        for rho in RATIOS:
            prefix = f"{pool}_{rho}"
            tcga_obs = observations_from_arrays(tcga_arrays, prefix)
            bracs_obs = observations_from_arrays(bracs_arrays, prefix)
            tcga_cov = gather_covariates(tcga_obs, tcga_arrays["h"], tcga_arrays["m"])
            bracs_cov = gather_covariates(
                bracs_obs, bracs_arrays["h"], bracs_arrays["m"]
            )
            covariates_by_pool[prefix] = (tcga_obs, tcga_cov, bracs_obs, bracs_cov)
            results[prefix] = fit_pool(tcga_obs, tcga_cov, bracs_obs, bracs_cov)
    return results, covariates_by_pool


def _all_sensitivities(
    tcga_arrays: dict[str, np.ndarray], covariates_by_pool: dict[str, PoolFits]
) -> dict[str, Any]:
    """Pre-registered sensitivities (a)/(b)/(c), evaluated on the primary total_100 pool."""
    t_obs, t_cov, b_obs, b_cov = covariates_by_pool["total_100"]
    return {
        "a_covariate_only": sensitivity_a(t_obs, t_cov, b_obs, b_cov),
        "b_absolute_allocation": sensitivity_b(t_obs, t_cov, b_obs, b_cov),
        "c_random_orders_only": sensitivity_c(tcga_arrays, b_obs, b_cov),
    }


def _write_outputs(
    data_dir: Path,
    label: str,
    results: dict[str, Any],
    sanity: dict[str, Any],
    sensitivities: dict[str, Any],
) -> Path:
    """Write analysis.json (published estimates) and diagnostics.json (full detail)."""
    path = data_dir / "analysis.json"
    write_json(
        path, {"label": label, "primary_pool": "total_100", "estimates": results}
    )
    write_json(
        data_dir / "diagnostics.json",
        {
            "label": label,
            "sanity": sanity,
            "pools": results,
            "sensitivities": sensitivities,
        },
    )
    return path


def run_analyze(config: dict[str, Any]) -> Path | None:
    """Pool both datasets' extracted observations, fit M0/M1 per pool, and write analysis.json."""
    data_dir = ensure_dirs(config)["data"]
    loaded = _load_both_datasets(config, data_dir)
    if loaded is None:
        logger.info(
            "Peer extract not ready yet; nothing to analyze until both datasets finish."
        )
        return None
    (tcga_arrays, tcga_meta), (bracs_arrays, bracs_meta) = loaded

    sanity = _check_sanity(tcga_meta, bracs_meta)
    results, covariates_by_pool = _fit_all_pools(tcga_arrays, bracs_arrays)
    primary = results["total_100"]
    label = label_outcome(
        primary["shrink_share"], primary["G0"]["point"], primary["G1"]
    )
    sensitivities = _all_sensitivities(tcga_arrays, covariates_by_pool)
    return _write_outputs(data_dir, label, results, sanity, sensitivities)
