"""Derived numbers quoted in the report, computed only from exported summaries.

Writes ``report/summary.json``: pooled balanced accuracy, neighbour agreement (raw and chance
adjusted, per metric and k), within-patient variance share, margin masses near zero, the
transition shares, the strongest rival's share of each class's errors, the per-class agreement-recall rank correlation, and the class-centre
separation index read from the separation-intervention study's stored precheck.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gap.plots.style import DATASETS
from gap.plots.codes import short_name

__all__ = ["write_summary"]

_SEPARATION = Path("34_separation_intervention/report/precheck_{ds}.json")


def _agreement(geo: list[dict[str, Any]], k_classes: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for metric in ("cosine", "euclidean"):
        for k in geo[0][metric]:
            out[f"{metric}_{k}"] = {
                "raw": float(np.mean([g[metric][k]["mean"] for g in geo])),
                "adjusted": float(
                    np.mean([g[metric][k]["chance_adjusted"] for g in geo])
                ),
                "adjusted_per_split": [g[metric][k]["chance_adjusted"] for g in geo],
            }
    out["chance"] = 1.0 / k_classes
    return out


def _cdf_quantile(grid: np.ndarray, cdf: np.ndarray, q: float) -> float:
    """Smallest grid value where the fit-averaged CDF reaches ``q``."""
    return float(grid[np.searchsorted(cdf, q)])


def _margin(a: dict[str, Any]) -> dict[str, Any]:
    """Medians of the fit-averaged CDFs, mass near a tie, and the fitted temperatures."""
    m = a["margin"]
    grid, raw = np.asarray(m["grid"]), np.asarray(m["cdf_raw"])
    temperature = np.asarray(a["temperature"])
    return {
        "median_pooled_cdf": _cdf_quantile(grid, raw, 0.5),
        "median_pooled_cdf_scaled": _cdf_quantile(
            grid, np.asarray(m["cdf_scaled"]), 0.5
        ),
        "median_patch_pooled_mean_over_fits": m["median_pooled"],
        "mass_within_1_nat": float(
            np.interp(1.0, grid, raw) - np.interp(-1.0, grid, raw)
        ),
        "tie_mass": m["tie_mass"],
        "temperature": {
            "mean": float(temperature.mean()),
            "min": float(temperature.min()),
            "max": float(temperature.max()),
        },
    }


def _transitions(a: dict[str, Any], ba_r1: float) -> dict[str, Any]:
    """Lost/gained shares per arm, and loss rate per balanced-margin bin."""
    out = {}
    for arm in ("P100", "S100", "r100"):
        correct = np.asarray(a[arm]["correct_by_margin"])
        lost = np.asarray(a[arm]["lost_by_margin"])
        out[arm] = {
            "lost": a[arm]["lost"] * 100,
            "gained": a[arm]["gained"] * 100,
            "lost_share_of_correct": a[arm]["lost"] * 100 / ba_r1,
            "gained_share_of_errors": a[arm]["gained"] * 100 / (100 - ba_r1),
            "correct_mass_by_margin_bin": (correct * 100).tolist(),
            "lost_rate_by_margin_bin": (lost / correct).tolist(),
            "lost_share_within_1_nat": float(lost[:2].sum() / lost.sum()),
        }
    correct = np.asarray(a["r100"]["correct_by_margin"])
    out["correct_share_within_1_nat"] = float(correct[:2].sum() / correct.sum())
    out["margin_bin_edges"] = a["provenance"]["margin_bin_edges"]
    return out


def _confusion(a: dict[str, Any]) -> dict[str, Any]:
    """Per-class recall and the strongest rival's share of each class's errors."""
    matrix = np.asarray(a["confusion_r1"])
    recall = np.diag(matrix)
    off = matrix - np.diag(recall)
    return {
        "recall": {
            short_name(n): float(r) * 100 for n, r in zip(a["class_names"], recall)
        },
        "top_rival_error_share": {
            "mean": float(np.mean(off.max(axis=1) / off.sum(axis=1))),
            "uniform": 1.0 / (len(recall) - 1),
        },
    }


def _geometry(a: dict[str, Any]) -> dict[str, Any]:
    """Agreement, its rank correlation with recall, variance shares, and PCA variance."""
    geo, names = a["geometry"], a["class_names"]
    per_class = np.mean([g["cosine"]["k15"]["per_class"] for g in geo], axis=0)
    recall = pd.Series(np.diag(np.asarray(a["confusion_r1"])))
    rho = pd.Series(per_class).corr(recall, method="spearman")
    return {
        "agreement": _agreement(geo, len(names)),
        "agreement_per_class_k15": {
            short_name(n): float(v) for n, v in zip(names, per_class)
        },
        "agreement_recall_spearman": {"rho": float(rho), "n": len(names)},
        "within_patient_share": float(
            np.mean([np.mean(g["within_patient_share"]) for g in geo])
        ),
        "pca_explained_variance_pc12": [g["pca_explained_variance"][:2] for g in geo],
        "samples": {
            "reference_patients_per_class": [
                sorted(set(g["reference"]["patients"].values())) for g in geo
            ],
            "query_patients": [sum(g["query"]["patients"].values()) for g in geo],
            "query_patches": [sum(g["query"]["patches"].values()) for g in geo],
        },
    }


def _dataset(a: dict[str, Any], cross_dir: Path, ds: str) -> dict[str, Any]:
    ba = {arm: float(np.mean(v)) * 100 for arm, v in a["ba_per_fit"].items()}
    separation = json.loads((cross_dir / str(_SEPARATION).format(ds=ds)).read_text())
    return {
        "ba": ba,
        "ba_per_split": {k: [x * 100 for x in v] for k, v in a["ba_per_split"].items()},
        **_confusion(a),
        **_geometry(a),
        "margin": _margin(a),
        "transitions": _transitions(a, ba["r1"]),
        "separation_index_patient_means": separation["separation_index"],
        "test_rows_per_split": a["provenance"]["test_rows_per_split"],
        "test_patients_per_split": a["provenance"]["test_patients_per_split"],
        "source_check": a["source_check"],
    }


def write_summary(report_dir: Path) -> Path:
    """Write report/summary.json from report/analysis_<dataset>.json."""
    cross_dir = report_dir.parents[1]
    out = {
        ds: _dataset(
            json.loads(
                (report_dir / f"analysis_{ds}.json").read_text(encoding="utf-8")
            ),
            cross_dir,
            ds,
        )
        for ds in DATASETS
    }
    path = report_dir / "summary.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return path
