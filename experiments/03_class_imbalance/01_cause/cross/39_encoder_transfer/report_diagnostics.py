"""Phase-06 report diagnostics that the locked analyze stage does not write.

Lives outside ``code/`` because the fit lock hashes every file there. Reads the accepted
fit records only and writes ``data/report_diagnostics.json``: per-split and per-draw test
BA, pooled per-class test recall, paired encoder contrasts of raw/scaled NLL and ECE and
of their imbalance damage, and validation-only probe diagnostics (feature norms, validation
balanced recall, logit margin, confusion).
"""

from __future__ import annotations

import argparse
import importlib
import logging
from typing import Any

import numpy as np

import _bootstrap  # noqa: F401
from imbalance_benchmark.common import (
    N_PATIENT_SPLITS,
    load_config,
    output_root,
    write_json,
)
from imbalance_benchmark.datasets.features.cache import reset_feature_bank

logger = logging.getLogger(__name__)

_PROB_ARMS = ("B", "R10", "R100")
_MARGIN_ARMS = ("B", "R100")
_TOP_PAIRS = 10


def _test_recalls(mods: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Observed per-fit BA and per-class recall (percent) for every encoder/arm."""
    secondary, common, transfer = mods["secondary"], mods["common"], mods["transfer"]
    paths = common.paths_by_split(config)
    ctxs = mods["bootstrap"].contexts(config)
    names = {s: mods["schedule"].load_train_identity(config, s)[1] for s in paths}
    per_fit: dict[str, Any] = {}
    per_class: dict[str, Any] = {}
    for m in transfer.ENCODERS:
        per_fit[m], per_class[m] = {}, {}
        for arm in transfer.ARMS:
            fits, classes = [], {}
            for s, d, result_dir in common.fit_dirs(paths, m, arm):
                test = common.require_record(
                    result_dir, splits=("test",), array_fields=("labels", "preds")
                )["splits"]["test"]
                recalls = (
                    secondary._patient_macro_recalls(
                        ctxs[s],
                        np.asarray(test["labels"]),
                        np.asarray(test["preds"]),
                        len(names[s]),
                    )[:, 0]
                    * 100.0
                )
                fits.append({"split": s, "draw": d, "ba": float(recalls.mean())})
                for name, value in zip(names[s], recalls):
                    classes.setdefault(name, []).append(float(value))
            per_fit[m][arm] = fits
            per_class[m][arm] = {k: float(np.mean(v)) for k, v in classes.items()}
    return {"per_fit_ba": per_fit, "per_class_recall": per_class}


def _probability_contrasts(
    mods: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Paired encoder differences in NLL/ECE and in their R-minus-B damage."""
    accuracy, common, transfer = mods["accuracy"], mods["common"], mods["transfer"]
    bootstrap, secondary = mods["bootstrap"], mods["secondary"]
    ctxs = bootstrap.contexts(config)
    w = bootstrap.encoder_draw_weights(common.N_DRAWS, ctxs[0].n_replicates)
    n_classes = len(mods["canonical"].canonical_class_names(config))
    _, prob = accuracy.pooled_arms(
        common.paths_by_split(config), ctxs, n_classes, (), _PROB_ARMS, w
    )
    v2, uni = transfer.ENCODERS
    out: dict[str, Any] = {}
    for key in ("nll", "nll_ts", "ece", "ece_ts"):
        for arm in _PROB_ARMS:
            out[f"delta_{arm}_{key}"] = prob[uni][arm][key] - prob[v2][arm][key]
        for arm in _PROB_ARMS[1:]:
            damage = {m: prob[m][arm][key] - prob[m]["B"][key] for m in (v2, uni)}
            out[f"damage_{v2}_{arm}_{key}"] = damage[v2]
            out[f"damage_{uni}_{arm}_{key}"] = damage[uni]
            out[f"delta_damage_{arm}_{key}"] = damage[uni] - damage[v2]
    return {k: secondary.pack_estimate(v) for k, v in out.items()}


def _margin_summary(margins: np.ndarray) -> dict[str, Any]:
    sd = float(margins.std())
    mean = float(margins.mean())
    return {
        "raw_mean": mean,
        "raw_median": float(np.median(margins)),
        "raw_sd": sd,
        "standardized_mean": mean / sd if sd > 0 else None,
        "zero_variance": sd == 0,
    }


def _validation_fit(
    mods: dict[str, Any], result_dir: Any, val_x: np.ndarray, val_y: np.ndarray
) -> tuple[float, dict[str, Any], np.ndarray]:
    """Selected-candidate validation score, logit margin summary, and predictions."""
    record = mods["common"].require_record(result_dir)
    lam = float(record["selected_lambda"])
    with np.load(result_dir / mods["fit"].CANDIDATES_NAME) as data:
        idx = int(np.argmin(np.abs(data["lambdas"] - lam)))
        logits = val_x @ data["coef"][idx].T + data["intercept"][idx]
    true = logits[np.arange(len(val_y)), val_y]
    rival = logits.copy()
    rival[np.arange(len(val_y)), val_y] = -np.inf
    margins = true - rival.max(axis=1)
    score = float(record["candidates"][idx]["validation_score"]) * 100.0
    return score, _margin_summary(margins), logits.argmax(axis=1)


def _encoder_validation(
    mods: dict[str, Any], config: dict[str, Any], encoder: str
) -> dict[str, Any]:
    common = mods["common"]
    paths = common.paths_by_split(config)
    norms: list[np.ndarray] = []
    dim = 0
    arms: dict[str, Any] = {a: {"scores": [], "margins": []} for a in _MARGIN_ARMS}
    confusion: dict[tuple[str, str], int] = {}
    class_totals: dict[str, int] = {}
    for s in range(N_PATIENT_SPLITS):
        _, names, evals, _ = mods["fit"].init_shard(config, s, encoder)
        val_x = np.asarray(evals.val_x, dtype=np.float64)
        val_y = np.asarray(evals.val_y)
        dim = val_x.shape[1]
        norms.append(np.linalg.norm(val_x, axis=1))
        for arm in _MARGIN_ARMS:
            for split, _, result_dir in common.fit_dirs(paths, encoder, arm):
                if split != s:
                    continue
                score, margin, preds = _validation_fit(mods, result_dir, val_x, val_y)
                arms[arm]["scores"].append(score)
                arms[arm]["margins"].append(margin)
                if arm != "B":
                    continue
                for t, p in zip(val_y, preds):
                    class_totals[names[t]] = class_totals.get(names[t], 0) + 1
                    if t != p:
                        pair = (names[t], names[p])
                        confusion[pair] = confusion.get(pair, 0) + 1
    all_norms = np.concatenate(norms)
    rates = sorted(
        ((n / class_totals[t], t, p) for (t, p), n in confusion.items()), reverse=True
    )
    return {
        "dimension": dim,
        "feature_norm": {
            "mean": float(all_norms.mean()),
            "sd": float(all_norms.std()),
            "median": float(np.median(all_norms)),
        },
        "arms": {
            a: {
                "validation_ba_mean": float(np.mean(v["scores"])),
                "validation_ba_sd": float(np.std(v["scores"])),
                "margin_raw_mean": float(
                    np.mean([m["raw_mean"] for m in v["margins"]])
                ),
                "margin_raw_sd": float(np.mean([m["raw_sd"] for m in v["margins"]])),
                "margin_standardized_mean": _mean_or_none(
                    [m["standardized_mean"] for m in v["margins"]]
                ),
                "zero_variance_fits": sum(m["zero_variance"] for m in v["margins"]),
            }
            for a, v in arms.items()
        },
        "b_top_confusions": [
            {"true": t, "predicted": p, "patch_rate": r}
            for r, t, p in rates[:_TOP_PAIRS]
        ],
    }


def _mean_or_none(values: list[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return float(np.mean(kept)) if kept else None


def _validation(mods: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for encoder in mods["transfer"].ENCODERS:
        reset_feature_bank()
        out[encoder] = _encoder_validation(mods, config, encoder)
    return out


def _modules() -> dict[str, Any]:
    return {
        "transfer": importlib.import_module("transfer"),
        "fit": importlib.import_module("transfer.fit"),
        "common": importlib.import_module("transfer.analyze.common"),
        "accuracy": importlib.import_module("transfer.analyze.accuracy"),
        "bootstrap": importlib.import_module("transfer.analyze.bootstrap"),
        "schedule": importlib.import_module("transfer.schedule"),
        "secondary": importlib.import_module("breadth.analyze.secondary"),
        "canonical": importlib.import_module("breadth.analyze.canonical"),
    }


def main() -> None:
    """Write data/report_diagnostics.json for one dataset config."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    config = load_config(parser.parse_args().config)
    mods = _modules()
    mods["fit"].audit_fits(config)
    payload = {
        **_test_recalls(mods, config),
        "probability_contrasts": _probability_contrasts(mods, config),
        "validation": _validation(mods, config),
    }
    out_path = output_root(config) / "data" / "report_diagnostics.json"
    write_json(out_path, payload)
    logger.info(out_path)


if __name__ == "__main__":
    main()
