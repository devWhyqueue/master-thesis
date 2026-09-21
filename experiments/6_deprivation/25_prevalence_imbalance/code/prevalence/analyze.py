"""Analyze stage: BA/NLL/ECE per arm (raw + TS), rank diagnostics, and the prevalence curve figure."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from imbalance_benchmark.common import (
    ensure_dirs,
    output_root,
    read_run_record,
    split_paths,
    write_json,
)

from breadth import BOOTSTRAP_SEED
from breadth.analyze.canonical import canonical_class_names
from breadth.analyze.secondary import _patient_macro_recalls, _probability_quality
from breadth.calibrate import scaled_test_probabilities

from sites import allocation_dir
from sites.recall import contexts

from decomposition.model import draw_weights

from centre import N_DRAWS, N_SPLITS
from centre.analyze import arm_accuracy, pooled

from directions.analyze import _write_analysis

from prevalence import ARMS, RATIOS
from prevalence.fit import class_permutation

__all__ = ["run_analyze"]


def _paths(config: dict[str, Any]) -> dict[int, dict[str, Path]]:
    """Result directories for every split."""
    return {s: split_paths(ensure_dirs(config), s) for s in range(N_SPLITS)}


def _require_record(
    result_dir: Path,
    splits: tuple[str, ...] | None = None,
    array_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Read one stored run record, raising if it is missing."""
    rec = read_run_record(result_dir, splits=splits, array_fields=array_fields)
    if rec is None:
        raise RuntimeError(f"Missing run record at {result_dir}")
    return rec


def _arm_dist(
    config: dict[str, Any], arms: tuple[str, ...], n_classes: int
) -> dict[str, np.ndarray]:
    """Raw and TS macro-NLL/ECE (F, R) distributions per arm, stacked over every (split, draw) fit."""
    ctxs = contexts(config)
    paths = _paths(config)
    keys = [(s, d) for s in range(N_SPLITS) for d in range(N_DRAWS)]
    out: dict[str, np.ndarray] = {}
    for arm in arms:
        cols: dict[str, list[np.ndarray]] = {
            "nll": [],
            "ece": [],
            "nll_ts": [],
            "ece_ts": [],
        }
        for s, d in keys:
            result_dir = allocation_dir(paths[s], arm, d)
            rec = _require_record(
                result_dir, splits=("test",), array_fields=("labels", "probabilities")
            )
            test = rec["splits"]["test"]
            labels = np.asarray(test["labels"])
            probs = np.asarray(test["probabilities"])
            scaled = scaled_test_probabilities(result_dir, probs)
            nll, ece = _probability_quality(ctxs[s], labels, probs, n_classes)
            nll_ts, ece_ts = _probability_quality(ctxs[s], labels, scaled, n_classes)
            cols["nll"].append(nll)
            cols["ece"].append(ece)
            cols["nll_ts"].append(nll_ts)
            cols["ece_ts"].append(ece_ts)
        for key, vals in cols.items():
            out[f"{key}_{arm}"] = np.stack(vals)
    return out


def _realized_rho(config: dict[str, Any], arms: tuple[str, ...]) -> dict[str, float]:
    """Mean realized rho per arm over every stored fit."""
    paths = _paths(config)
    return {
        arm: float(
            np.mean(
                [
                    _require_record(allocation_dir(paths[s], arm, d))["realized_rho"]
                    for s in range(N_SPLITS)
                    for d in range(N_DRAWS)
                ]
            )
        )
        for arm in arms
    }


def _native_placement(
    ba: dict[str, np.ndarray], rho: dict[str, float]
) -> dict[str, float]:
    """BA(N) - BA(r1), and the gap to the ratio curve interpolated at N's realized rho."""
    xs = np.array([rho[f"r{r}"] for r in RATIOS])
    ys = np.array([float(ba[f"r{r}"][0]) for r in RATIOS])
    order = np.argsort(xs)
    interpolated = float(np.interp(rho["N"], xs[order], ys[order]))
    return {
        "ba_N_minus_r1": float(ba["N"][0] - ba["r1"][0]),
        "ba_N_minus_interpolated_curve": float(ba["N"][0]) - interpolated,
    }


def _thirds(
    config: dict[str, Any], arms: tuple[str, ...], names: list[str]
) -> dict[str, dict[str, float]]:
    """Mean observed patient-macro recall (%) of head/body/tail classes per arm, by assigned rank."""
    ctxs = contexts(config)
    paths = _paths(config)
    n_classes = len(names)
    out: dict[str, dict[str, float]] = {"head": {}, "body": {}, "tail": {}}
    for arm in arms:
        vals: dict[str, list[float]] = {"head": [], "body": [], "tail": []}
        for s in range(N_SPLITS):
            for d in range(N_DRAWS):
                groups = np.array_split(class_permutation(s, d, n_classes), 3)
                result_dir = allocation_dir(paths[s], arm, d)
                rec = _require_record(
                    result_dir, splits=("test",), array_fields=("labels", "preds")
                )
                test = rec["splits"]["test"]
                labels, preds = np.asarray(test["labels"]), np.asarray(test["preds"])
                recalls = (
                    _patient_macro_recalls(ctxs[s], labels, preds, n_classes)[:, 0]
                    * 100.0
                )
                for key, idx in zip(("head", "body", "tail"), groups):
                    vals[key].append(float(recalls[idx].mean()))
        for key in out:
            out[key][arm] = float(np.mean(vals[key]))
    return out


def _figure(dists: dict[str, np.ndarray], rho: dict[str, float], dest: Path) -> None:
    """3-panel BA/NLL/ECE vs log2(rho) curve; native arm as an off-axis marker, raw solid / TS dashed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    xs = [np.log2(r) for r in RATIOS]
    native_x = xs[-1] + 1.5
    panels = (
        ("BA (%)", "arm", None),
        ("Macro NLL (nats)", "nll", "nll_ts"),
        ("ECE (pp)", "ece", "ece_ts"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=200)
    for ax, (label, raw_key, ts_key) in zip(axes, panels):
        ax.plot(
            xs,
            [dists[f"{raw_key}_r{r}"][0] for r in RATIOS],
            "o-",
            color="tab:blue",
            label="raw",
        )
        ax.scatter(
            [native_x],
            [dists[f"{raw_key}_N"][0]],
            color="tab:red",
            marker="D",
            zorder=5,
            label="N",
        )
        if ts_key is not None:
            ax.plot(
                xs,
                [dists[f"{ts_key}_r{r}"][0] for r in RATIOS],
                "o--",
                color="tab:blue",
                alpha=0.6,
                label="TS",
            )
            ax.scatter(
                [native_x],
                [dists[f"{ts_key}_N"][0]],
                facecolors="none",
                edgecolors="tab:red",
                marker="D",
            )
        ax.axvline(native_x - 0.75, linestyle=":", color="gray", linewidth=0.8)
        ax.set_xlabel(r"$\log_2(\rho)$")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
    fig.suptitle(f"Prevalence curve (realized $\\rho_N$ = {rho['N']:.1f})")
    fig.tight_layout()
    fig.savefig(dest)
    plt.close(fig)


def _combine(
    ba: dict[str, np.ndarray], secondary: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Arm-prefixed BA distributions, secondary metrics, and r-vs-r1 contrasts."""
    dists = {f"arm_{arm}": d for arm, d in ba.items()} | secondary
    for r in RATIOS[1:]:
        dists[f"arm_r{r}_minus_r1"] = ba[f"r{r}"] - ba["r1"]
    dists["arm_N_minus_r1"] = ba["N"] - ba["r1"]
    return dists


def run_analyze(config: dict[str, Any]) -> Path:
    """Pool BA/NLL/ECE per arm, write analysis and rank diagnostics, and plot the prevalence curve."""
    names = canonical_class_names(config)
    acc = arm_accuracy(config, names, arms=ARMS)
    dist = _arm_dist(config, ARMS, len(names))
    n_replicates = next(iter(acc.values())).shape[-1]
    fit_split = np.repeat(np.arange(N_SPLITS), N_DRAWS)
    w = draw_weights(
        fit_split, N_DRAWS, n_replicates, np.random.default_rng(BOOTSTRAP_SEED)
    )
    ba = {arm: pooled(a, w) for arm, a in acc.items()}
    dists = _combine(ba, {key: pooled(a, w) for key, a in dist.items()})
    path = _write_analysis(config, acc, fit_split, dists)
    rho = _realized_rho(config, ARMS)
    write_json(
        path.with_name("diagnostics.json"),
        {
            "realized_rho": rho,
            **_native_placement(ba, rho),
            "rank_recall": _thirds(config, ARMS, names),
        },
    )
    _figure(dists, rho, output_root(config) / "figures" / "prevalence_curve.pdf")
    return path
