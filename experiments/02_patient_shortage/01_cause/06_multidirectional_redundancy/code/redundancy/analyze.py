"""Analyze stage: refit the support surface under each redundancy measure."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.common import output_root, verify_signed_file, write_json

from breadth import GRID_CELLS, N_REPLICATES
from breadth.analyze.contrasts import collect_cell_distributions
from breadth.analyze.secondary import pack_estimate

from redundancy import MEASURES, exp5_config
from redundancy.estimator import cell_effective_support
from redundancy.surfaces import LN2, fit_all_surfaces, paired_differences

__all__ = ["classify", "run_analyze"]

logger = logging.getLogger(__name__)

_EXP5_TOLERANCE = 1e-6
_UNDERESTIMATED_BOUND = 1.0


def classify(b_ci: tuple[float, float], res_full: float, res_single: float) -> str:
    """Interpretation label (report Sec. "Interpretation criteria")."""
    lo, hi = b_ci
    within_bound = lo >= -_UNDERESTIMATED_BOUND and hi <= _UNDERESTIMATED_BOUND
    if within_bound and res_full <= res_single:
        return "underestimated_redundancy"
    if lo > _UNDERESTIMATED_BOUND:
        return "breadth_beyond_redundancy"
    return "inconclusive"


def _load_correlations(
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], list[str]]:
    npz_p = output_root(config) / "data" / "correlations.npz"
    verify_signed_file(npz_p)
    with np.load(npz_p, allow_pickle=False) as data:
        class_names = [str(c) for c in data["class_names"]]
        raw = {measure: np.asarray(data[measure]) for measure in MEASURES}
    return raw, class_names


def _clipped_split_mean(raw_measure: np.ndarray) -> np.ndarray:
    """``(R, S, C)`` raw ICC to ``(R, C)`` clipped, split-averaged correlations."""
    return np.mean(np.clip(raw_measure, 0.0, 1.0), axis=1)


def _correlation_summaries(
    raw: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    rho_mean = {measure: _clipped_split_mean(raw[measure]) for measure in MEASURES}
    neff_all = {
        measure: cell_effective_support(rho_mean[measure], GRID_CELLS)
        for measure in MEASURES
    }
    neff_point = {measure: neff_all[measure][0] for measure in MEASURES}
    return rho_mean, neff_all, neff_point


def _class_correlations(
    raw_measure: np.ndarray, rho_mean: np.ndarray, class_names: list[str]
) -> dict[str, Any]:
    raw_point = raw_measure[0]  # (S, C): true unweighted sample, unclipped
    return {
        "raw_per_split": {
            name: raw_point[:, idx].tolist() for idx, name in enumerate(class_names)
        },
        "clipped_mean": {
            name: float(rho_mean[0, idx]) for idx, name in enumerate(class_names)
        },
        "training_ci": {
            name: pack_estimate(rho_mean[:, idx])
            for idx, name in enumerate(class_names)
        },
    }


def _cell_effective_support_report(
    neff_point: np.ndarray, neff_all: np.ndarray
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for idx, (g, m) in enumerate(GRID_CELLS):
        est = pack_estimate(neff_all[:, idx])
        out[f"G{g}_m{m}"] = {
            "point": float(neff_point[idx]),
            "training_ci": [est["ci_2_5"], est["ci_97_5"]],
        }
    return out


def _cell_accuracy_report(
    pooled_dists: dict[tuple[int, int], np.ndarray],
    dispersions: dict[tuple[int, int], float],
) -> dict[str, Any]:
    return {
        f"G{g}_m{m}": {
            **pack_estimate(pooled_dists[(g, m)]),
            "draw_dispersion": dispersions[(g, m)],
        }
        for g, m in GRID_CELLS
    }


def _interpretation(surfaces: dict[str, Any]) -> dict[str, Any]:
    full_b = surfaces["full"]["b"]
    res_full_point = surfaces["full"]["res_std_neff"]["point"]
    res_single_point = surfaces["single"]["res_std_neff"]["point"]
    label_test = classify(tuple(full_b["test_ci"]), res_full_point, res_single_point)
    label_training = classify(
        tuple(full_b["training_ci"]), res_full_point, res_single_point
    )
    return {
        "label_test": label_test,
        "label_training": label_training,
        "fragile": label_test != label_training,
        "threshold_pp": _UNDERESTIMATED_BOUND,
    }


def _build_results(
    raw: dict[str, np.ndarray],
    rho_mean: dict[str, np.ndarray],
    neff_point: dict[str, np.ndarray],
    neff_all: dict[str, np.ndarray],
    class_names: list[str],
    pooled_dists: dict[tuple[int, int], np.ndarray],
    dispersions: dict[tuple[int, int], float],
    surfaces: dict[str, Any],
) -> dict[str, Any]:
    return {
        "class_correlations": {
            measure: _class_correlations(raw[measure], rho_mean[measure], class_names)
            for measure in MEASURES
        },
        "cell_effective_support": {
            measure: _cell_effective_support_report(
                neff_point[measure], neff_all[measure]
            )
            for measure in MEASURES
        },
        "cell_accuracies": _cell_accuracy_report(pooled_dists, dispersions),
        "surfaces": surfaces,
        "interpretation": _interpretation(surfaces),
    }


def _verify_against_exp5_analysis(
    config: dict[str, Any], single_point_b: float, single_point_res_std: float
) -> None:
    """Guard: the single-measure surface must reproduce exp-5's own fit exactly."""
    analysis_p = output_root(exp5_config(config)) / "data" / "analysis.json"
    exp5_analysis = json.loads(analysis_p.read_text(encoding="utf-8"))
    surf = exp5_analysis["surface_parameters"]
    expected_b = surf["gamma_e"]["point"] * LN2
    expected_res_std = surf["res_std_neff"]["point"]
    if abs(single_point_b - expected_b) > _EXP5_TOLERANCE:
        raise ValueError(
            f"Single-measure b={single_point_b} does not reproduce exp-5's "
            f"b={expected_b}"
        )
    if abs(single_point_res_std - expected_res_std) > _EXP5_TOLERANCE:
        raise ValueError(
            f"Single-measure res_std_neff={single_point_res_std} does not "
            f"reproduce exp-5's res_std_neff={expected_res_std}"
        )


def _prepare_accuracies(
    config: dict[str, Any], class_names: list[str]
) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], float], np.ndarray]:
    """Exp-5's stored bootstrap accuracy distributions, plus the (R, 9) grid matrix."""
    logger.info("Collecting exp-5's stored bootstrap accuracy distributions...")
    pooled_dists, dispersions, _secondaries = collect_cell_distributions(
        exp5_config(config), class_names
    )
    accs_matrix = np.array(
        [[pooled_dists[c][b] for c in GRID_CELLS] for b in range(N_REPLICATES)]
    )
    return pooled_dists, dispersions, accs_matrix


def _fit_and_verify(
    config: dict[str, Any],
    accs_matrix: np.ndarray,
    neff_point: dict[str, np.ndarray],
    neff_all: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Fit both measures' surfaces, guard the single measure against exp-5, and diff."""
    surfaces, b_dists, res_dists = fit_all_surfaces(
        accs_matrix, accs_matrix[0], neff_point, neff_all
    )
    _verify_against_exp5_analysis(
        config,
        surfaces["single"]["b"]["point"],
        surfaces["single"]["res_std_neff"]["point"],
    )
    surfaces["differences_full_minus_single"] = paired_differences(b_dists, res_dists)
    return surfaces


def run_analyze(config: dict[str, Any]) -> Path:
    """Refit the residual breadth surface under both redundancy measures."""
    raw, class_names = _load_correlations(config)
    rho_mean, neff_all, neff_point = _correlation_summaries(raw)
    pooled_dists, dispersions, accs_matrix = _prepare_accuracies(config, class_names)
    surfaces = _fit_and_verify(config, accs_matrix, neff_point, neff_all)

    results = _build_results(
        raw,
        rho_mean,
        neff_point,
        neff_all,
        class_names,
        pooled_dists,
        dispersions,
        surfaces,
    )
    out_p = output_root(config) / "data" / "analysis.json"
    write_json(out_p, results)
    logger.info("Multidirectional-redundancy analysis complete: %s", out_p)
    return out_p
