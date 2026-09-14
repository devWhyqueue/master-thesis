"""Analyze stage: residual breadth benefit, composition prediction, and label."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from imbalance_benchmark.common import ensure_dirs, output_root, split_paths, write_json

from breadth import N_SPLITS
from breadth.analyze.canonical import canonical_class_names
from breadth.analyze.secondary import pack_estimate

from sites.recall import contexts, ctx_list, perm_list

from neighbours.analyze import _above_threshold, _within_threshold

from coverage_redundancy import LN2, THRESHOLD_PP, exp5_config, exp10_config
from coverage_redundancy.census import load_quantities
from coverage_redundancy.models import descriptives, model_result
from coverage_redundancy.prediction import (
    FittedModels,
    Prediction,
    fit_models,
    predict_composition,
)
from coverage_redundancy.rows import (
    GridRows,
    CompositionRows,
    cohort_index,
    composition_rows,
    grid_rows,
)

__all__ = ["classify", "run_analyze"]

logger = logging.getLogger(__name__)


class RecallContext(NamedTuple):
    """The canonical class order and shared bootstrap contexts of every fit."""

    class_names: list[str]
    n_classes: int
    ctx_l: list[Any]
    perms: list[np.ndarray]


def _load_recall_context(config: dict[str, Any]) -> RecallContext:
    """Canonical class order plus the shared per-fit bootstrap contexts and perms."""
    class_names = canonical_class_names(config)
    return RecallContext(
        class_names,
        len(class_names),
        ctx_list(contexts(config)),
        perm_list(config, class_names),
    )


def _load_paths(config: dict[str, Any]) -> tuple[dict[int, Any], dict[int, Any]]:
    """Split-scoped output paths for exp-5's grid and exp-10's composition arms."""
    paths5 = {
        s: split_paths(ensure_dirs(exp5_config(config)), s) for s in range(N_SPLITS)
    }
    paths10 = {
        s: split_paths(ensure_dirs(exp10_config(config)), s) for s in range(N_SPLITS)
    }
    return paths5, paths10


def _entirely_outside(ci: tuple[float, float]) -> bool:
    """Whether an interval sits entirely outside +/- the practical threshold."""
    return ci[1] < -THRESHOLD_PP or ci[0] > THRESHOLD_PP


def classify(
    b_ci: tuple[float, float],
    con_err_ci: tuple[float, float],
    sel_err_ci: tuple[float, float],
) -> str:
    """Interpretation label (report Table "readings")."""
    if (
        _within_threshold(b_ci)
        and _within_threshold(con_err_ci)
        and _within_threshold(sel_err_ci)
    ):
        return "coverage_and_similarity"
    if _within_threshold(b_ci) and _entirely_outside(con_err_ci):
        return "random_cohorts_only"
    if _above_threshold(b_ci):
        return "breadth_beyond_both"
    return "inconclusive"


def _model_summaries(models: FittedModels) -> dict[str, Any]:
    """Pack models (a), (b), (c), and (c) without log G for analysis.json."""
    model_c = {
        **model_result(models.theta_c, models.res_c, beta_idx=3, gamma_idx=5),
        "delta": pack_estimate(models.theta_c[:, 4]),
    }
    return {
        "a": model_result(models.theta_a, models.res_a, beta_idx=3, gamma_idx=4),
        "b": model_result(models.theta_b, models.res_b, beta_idx=3, gamma_idx=4),
        "c": model_c,
        "c_minus_log_g": {"res_std": pack_estimate(models.res_cm)},
    }


def _label_for(
    model_c: dict[str, Any], prediction: Prediction
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Interpretation label from model (c)'s b and the two prediction errors."""
    con_est = pack_estimate(prediction.con_error)
    sel_est = pack_estimate(prediction.sel_error)
    label = classify(
        (model_c["b"]["ci_2_5"], model_c["b"]["ci_97_5"]),
        (con_est["ci_2_5"], con_est["ci_97_5"]),
        (sel_est["ci_2_5"], sel_est["ci_97_5"]),
    )
    return label, con_est, sel_est


def _prediction_summary(
    prediction: Prediction, con_est: dict[str, Any], sel_est: dict[str, Any]
) -> dict[str, Any]:
    """Pack the composition prediction, its errors, and the random-arm calibration."""
    return {
        "delta_con_hat": pack_estimate(prediction.delta_con_hat),
        "delta_sel_hat": pack_estimate(prediction.delta_sel_hat),
        "con_error": con_est,
        "sel_error": sel_est,
        "calibration_random": {
            "predicted": pack_estimate(prediction.a_ran_hat),
            "observed": pack_estimate(prediction.a_ran_obs),
        },
    }


def _write_distributions(
    config: dict[str, Any], models: FittedModels, prediction: Prediction
) -> None:
    """Write the raw per-replicate distributions behind analysis.json."""
    np.savez(
        output_root(config) / "data" / "distributions.npz",
        b_a=models.theta_a[:, 4] * LN2,
        b_b=models.theta_b[:, 4] * LN2,
        b_c=models.theta_c[:, 5] * LN2,
        res_a=models.res_a,
        res_b=models.res_b,
        res_c=models.res_c,
        res_c_minus_log_g=models.res_cm,
        delta_con_hat=prediction.delta_con_hat,
        delta_sel_hat=prediction.delta_sel_hat,
        con_error=prediction.con_error,
        sel_error=prediction.sel_error,
    )


def _build_output(
    grid: GridRows,
    comp: CompositionRows,
    quantities: dict[str, Any],
    model_summaries: dict[str, Any],
    prediction: Prediction,
    con_est: dict[str, Any],
    sel_est: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    """Assemble the analysis.json payload."""
    return {
        "grid_reference": {
            "random_cohorts": int(grid.accuracy.shape[0]),
            "composition_cohorts": int(comp.accuracy.shape[0]),
        },
        "models": model_summaries,
        "prediction": _prediction_summary(prediction, con_est, sel_est),
        "descriptives": descriptives(quantities),
        "interpretation": {"label": label, "threshold_pp": THRESHOLD_PP},
    }


def run_analyze(config: dict[str, Any]) -> Path:
    """Fit the residual breadth surface, predict cohort composition, and label it."""
    quantities = load_quantities(config)
    index = cohort_index(quantities)
    rc = _load_recall_context(config)
    paths5, paths10 = _load_paths(config)

    grid = grid_rows(config, paths5, rc.ctx_l, rc.perms, rc.n_classes, index)
    comp, accuracy_by_allocation = composition_rows(
        config, paths10, rc.ctx_l, rc.perms, rc.n_classes, index
    )

    models = fit_models(grid)
    prediction = predict_composition(models.theta_cm, comp, accuracy_by_allocation)
    model_summaries = _model_summaries(models)
    label, con_est, sel_est = _label_for(model_summaries["c"], prediction)

    out = _build_output(
        grid, comp, quantities, model_summaries, prediction, con_est, sel_est, label
    )
    out_p = output_root(config) / "data" / "analysis.json"
    write_json(out_p, out)
    _write_distributions(config, models, prediction)
    logger.info("Coverage-redundancy analysis complete: %s", out_p)
    return out_p
