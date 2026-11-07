"""Analyze stage: patient-weighted confusion, margins, and transitions over every stored fit.

Reads r1 (balanced), r100, P100 (prior only), and S100 (support only) for 3 splits x 10 draws,
weights each fit equally, and checks every identity the report relies on: stored per-run BA,
the pooled r1/r100 means against the source experiments' own estimates, positive-margin mass
against BA, and lost - gained = damage. Geometry summaries and example selections written by the
geometry stage are merged in, so the local report needs only this one JSON and examples.csv.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imbalance_benchmark.common import ensure_dirs, write_json

from centre import N_DRAWS, N_SPLITS

from spectrum import baseline_config

from gap.embedding import geometry_path
from gap.metrics import (
    confusion,
    lost_by_margin,
    margins,
    transitions,
    weighted_cdf,
    weighted_median,
)
from gap.records import ARM_SOURCES, load_fit, split_identity

__all__ = ["run_analyze", "run_smoke"]

logger = logging.getLogger(__name__)

_IMBALANCED = ("r100", "P100", "S100")
_GRID = np.round(np.arange(-40.0, 40.0001, 0.05), 4)
_IDENTITY_TOLERANCE = 1e-12
_SOURCE_TOLERANCE_PP = 0.02
_MARGIN_EDGES = (0.0, 0.5, 1.0, 2.0, 4.0, np.inf)


def _margin_summary(fit, labels: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    """Raw and validation-temperature-scaled margin CDFs plus the BA/tie consistency check."""
    m, ties = margins(fit.probs, labels)
    mass = weights / fit.probs.shape[1]
    preds, temperature = fit.preds, fit.temperature
    mismatch = ((m > 0) != (preds == labels)) & ~ties
    if mismatch.any():
        raise RuntimeError(
            f"{int(mismatch.sum())} untied rows where margin sign != correctness"
        )
    return {
        "cdf_raw": weighted_cdf(m, mass, _GRID),
        "cdf_scaled": weighted_cdf(m / temperature, mass, _GRID),
        "positive_mass": float(mass[m > 0].sum()),
        "tie_mass": float(mass[ties].sum()),
        "median_weighted": weighted_median(m, mass),
        "median_pooled": float(np.median(m)),
        "median_weighted_scaled": weighted_median(m / temperature, mass),
    }


def _arm_transitions(r1, fit, ident, r1_margin: np.ndarray) -> dict[str, Any]:
    """Lost/gained mass against r1, overall, per class, and per r1-margin bin."""
    t = transitions(r1.preds, fit.preds, ident.labels, ident.weights)
    if abs(t["lost"] - t["gained"] - (r1.ba - fit.ba)) > _IDENTITY_TOLERANCE:
        raise RuntimeError("lost - gained != balanced-accuracy damage")
    t["correct_by_margin"], t["lost_by_margin"] = lost_by_margin(
        r1_margin,
        r1.preds == ident.labels,
        fit.preds == ident.labels,
        ident.weights,
        _MARGIN_EDGES,
        len(ident.names),
    )
    return t


def _one_fit(config: dict[str, Any], ident, split: int, draw: int) -> dict[str, Any]:
    """Every per-fit quantity for one (split, draw)."""
    r1 = load_fit(config, ident, "r1", split, draw)
    out: dict[str, Any] = {
        "ba": {"r1": r1.ba},
        "confusion": confusion(ident.labels, r1.preds, ident.weights, len(ident.names)),
        "margin": _margin_summary(r1, ident.labels, ident.weights),
        "temperature": r1.temperature,
        "r1_correct": r1.preds == ident.labels if split == 0 else None,
        "r1_preds": r1.preds if split == 0 else None,
    }
    if r1.probs is None:
        raise RuntimeError(f"r1 probabilities missing at s{split} d{draw}")
    r1_margin, _ = margins(r1.probs, ident.labels)
    for arm in _IMBALANCED:
        fit = load_fit(config, ident, arm, split, draw, with_probs=False)
        out["ba"][arm] = fit.ba
        out[arm] = _arm_transitions(r1, fit, ident, r1_margin)
    margin = out["margin"]
    if abs(margin["positive_mass"] - r1.ba) > margin["tie_mass"] + _IDENTITY_TOLERANCE:
        raise RuntimeError(f"positive-margin mass != BA at s{split} d{draw}")
    return out


def _source_check(config: dict[str, Any], ba: dict[str, np.ndarray]) -> dict[str, Any]:
    """Pooled means against the source experiments' stored point estimates (0.02 pp)."""
    out = {}
    for arm, values in ba.items():
        path = Path(baseline_config(config, ARM_SOURCES[arm])["paths"]["outputs"])
        estimates = json.loads((path / "data/analysis.json").read_text())["estimates"]
        source = estimates.get(f"arm_{arm}", {}).get("point")
        ours = float(values.mean() * 100.0)
        if source is not None and abs(ours - source) > _SOURCE_TOLERANCE_PP:
            raise RuntimeError(f"{arm} mean BA {ours:.4f} vs source {source:.4f}")
        out[arm] = {"reproduced": ours, "source": source}
    return out


def _examples(
    config: dict[str, Any], split0: list[dict[str, Any]], ident0
) -> pd.DataFrame:
    """Attach split-0 r1 correctness rate and modal prediction to each example query patch."""
    path = geometry_path(config, 0).with_name("examples.csv")
    examples = pd.read_csv(path, dtype={"patch_id": str, "case_id": str})
    row_of = {p: i for i, p in enumerate(ident0.frame["patch_id"])}
    correct = np.stack([f["r1_correct"] for f in split0])
    preds = np.stack([f["r1_preds"] for f in split0])
    queries = examples["role"] == "query"
    rows = [row_of[p] for p in examples.loc[queries, "patch_id"]]
    modal = [ident0.names[np.bincount(preds[:, i]).argmax()] for i in rows]
    examples.loc[queries, "r1_correct_rate"] = correct[:, rows].mean(axis=0)
    examples.loc[queries, "r1_modal_prediction"] = modal
    return examples


def _pool(
    fits: list[dict[str, Any]], names: list[str]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Equal-weight means over every fit, plus per-split BA."""
    ba = {arm: np.array([f["ba"][arm] for f in fits]) for arm in fits[0]["ba"]}
    out: dict[str, Any] = {
        "ba_per_fit": {arm: v.tolist() for arm, v in ba.items()},
        "ba_per_split": {
            arm: v.reshape(N_SPLITS, N_DRAWS).mean(axis=1).tolist()
            for arm, v in ba.items()
        },
        "confusion_r1": np.mean([f["confusion"] for f in fits], axis=0).tolist(),
        "temperature": [f["temperature"] for f in fits],
        "margin": {
            "grid": _GRID.tolist(),
            **{
                key: np.mean([f["margin"][key] for f in fits], axis=0).tolist()
                for key in fits[0]["margin"]
            },
        },
    }
    for arm in _IMBALANCED:
        out[arm] = {
            key: np.mean([f[arm][key] for f in fits], axis=0).tolist()
            for key in (
                "lost",
                "gained",
                "lost_per_class",
                "gained_per_class",
                "correct_by_margin",
                "lost_by_margin",
            )
        }
    out["class_names"] = names
    return out, ba


def _all_fits(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], Any, dict[str, list[int]]]:
    """Every (split, draw) fit summary, split 0's identity, and test rows per split."""
    fits, ident0, n_rows = [], None, {"patches": [], "patients": []}
    for split in range(N_SPLITS):
        ident = split_identity(config, split)
        ident0 = ident if split == 0 else ident0
        n_rows["patches"].append(len(ident.labels))
        n_rows["patients"].append(int(ident.frame["case_id"].nunique()))
        for draw in range(N_DRAWS):
            logger.info("split %d draw %d", split, draw)
            fits.append(_one_fit(config, ident, split, draw))
    return fits, ident0, n_rows


def _provenance(config: dict[str, Any], n_rows: dict[str, list[int]]) -> dict[str, Any]:
    """Source experiments, design, and sampling parameters behind analysis.json."""
    return {
        "dataset": config["dataset"]["name"],
        "arms": {arm: ARM_SOURCES[arm] for arm in ("r1", *_IMBALANCED)},
        "sources": {key: config["slurm"][key] for key in set(ARM_SOURCES.values())},
        "exp2_outputs": config["slurm"]["exp2_outputs"],
        "n_splits": N_SPLITS,
        "n_draws": N_DRAWS,
        "sampling": config["analysis"],
        "test_rows_per_split": n_rows["patches"],
        "test_patients_per_split": n_rows["patients"],
        "margin_bin_edges": list(_MARGIN_EDGES[:-1]),
    }


def run_analyze(config: dict[str, Any]) -> Path:
    """Pool all stored fits, verify identities, merge geometry, and write analysis.json."""
    fits, ident0, n_rows = _all_fits(config)
    result, ba = _pool(fits, ident0.names)
    result["source_check"] = _source_check(config, ba)
    result["geometry"] = [
        json.loads(geometry_path(config, s).read_text()) for s in range(N_SPLITS)
    ]
    result["provenance"] = _provenance(config, n_rows)
    data = ensure_dirs(config)["data"]
    examples = _examples(config, fits[:N_DRAWS], ident0)
    examples.to_csv(data / "examples_annotated.csv", index=False)
    path = data / "analysis.json"
    write_json(path, result)
    logger.info("analysis written to %s", path)
    return path


def run_smoke(config: dict[str, Any]) -> None:
    """Check record alignment and every identity on split 0, draw 0 only; writes nothing."""
    ident = split_identity(config, 0)
    fit = _one_fit(config, ident, 0, 0)
    margin = fit["margin"]
    logger.info(
        "smoke s0 d0: rows %d, BA %s, positive-margin mass %.6f, tie mass %.2e, T %.4f",
        len(ident.labels),
        {arm: round(v * 100.0, 4) for arm, v in fit["ba"].items()},
        margin["positive_mass"],
        margin["tie_mass"],
        fit["temperature"],
    )
