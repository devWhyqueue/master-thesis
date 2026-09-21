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


def _slope(ba: dict[str, np.ndarray], ratios: tuple[int, ...]) -> np.ndarray:
    """OLS slope of BA on log2(r) per replicate, in pp per doubling of the nominal ratio."""
    x = np.log2(ratios) - np.log2(ratios).mean()
    y = np.stack([ba[f"r{r}"] for r in ratios])
    return (x[:, None] * (y - y.mean(0))).sum(0) / (x**2).sum()


def _native_gap(ba: dict[str, np.ndarray], rho: dict[str, float]) -> np.ndarray:
    """BA(N) minus the ratio curve linearly interpolated at N's realized rho, per replicate."""
    xs = np.array([rho[f"r{r}"] for r in RATIOS])
    ys = np.stack([ba[f"r{r}"] for r in RATIOS])
    order = np.argsort(xs)
    curve = np.array([np.interp(rho["N"], xs[order], col) for col in ys[order].T])
    return ba["N"] - curve


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


def _band(ax: Any, xs: list[float], d: list[np.ndarray], fmt: str, **kw: Any) -> None:
    """Point estimates joined by a line, with a shaded 95% percentile band over replicates."""
    lo, hi = zip(*(np.percentile(v[1:], [2.5, 97.5]) for v in d))
    ax.plot(xs, [v[0] for v in d], fmt, color="tab:blue", **kw)
    ax.fill_between(xs, lo, hi, color="tab:blue", alpha=0.15, linewidth=0)


def _figure(dists: dict[str, np.ndarray], rho: dict[str, float], dest: Path) -> None:
    """3-panel BA damage/NLL/ECE vs realized rho (log2 axis), 95% bands, native at its realized rho."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    dists = dists | {f"ba_{a}": dists[f"arm_{a}"] - dists["arm_r1"] for a in ARMS}
    xs = [float(np.log2(rho[f"r{r}"])) for r in RATIOS]
    native_x = float(np.log2(rho["N"]))
    panels = (
        (r"BA change vs. $\rho$ = 1 (pp)", "ba", None),
        ("Macro NLL (nats)", "nll", "nll_ts"),
        ("ECE (pp)", "ece", "ece_ts"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), dpi=200)
    for ax, (label, raw_key, ts_key) in zip(axes, panels):
        keys = (raw_key,) if ts_key is None else (raw_key, ts_key)
        for key, fmt, fill in zip(keys, ("o-", "o--"), ("tab:red", "none")):
            name = "raw" if key == raw_key else "temperature-scaled"
            _band(ax, xs, [dists[f"{key}_r{r}"] for r in RATIOS], fmt, label=name)
            native = dists[f"{key}_N"]
            ax.errorbar(
                [native_x],
                [native[0]],
                yerr=[
                    [native[0] - np.percentile(native[1:], 2.5)],
                    [np.percentile(native[1:], 97.5) - native[0]],
                ],
                fmt="D",
                color="tab:red",
                mfc=fill,
                zorder=5,
                label=f"native ({name})" if ts_key else "native",
            )
        ax.set_xticks(xs, [str(r) for r in RATIOS])
        ax.set_xlabel(r"Imbalance ratio $\rho$ (log scale)")
        ax.set_ylabel(label)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(dest)
    plt.close(fig)


def _combine(
    ba: dict[str, np.ndarray], secondary: dict[str, np.ndarray], rho: dict[str, float]
) -> dict[str, np.ndarray]:
    """Arm-prefixed BA, secondary metrics, r-vs-r1 contrasts, curve slopes, native gap."""
    dists = {f"arm_{arm}": d for arm, d in ba.items()} | secondary
    for r in RATIOS[1:]:
        dists[f"arm_r{r}_minus_r1"] = ba[f"r{r}"] - ba["r1"]
    dists["arm_N_minus_r1"] = ba["N"] - ba["r1"]
    dists["arm_N_minus_curve"] = _native_gap(ba, rho)
    dists["slope_all"] = _slope(ba, RATIOS)
    dists["slope_r1_r10"] = _slope(ba, RATIOS[:4])
    dists["slope_r10_r100"] = _slope(ba, RATIOS[3:])
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
    rho = _realized_rho(config, ARMS)
    dists = _combine(ba, {key: pooled(a, w) for key, a in dist.items()}, rho)
    path = _write_analysis(config, acc, fit_split, dists)
    write_json(
        path.with_name("diagnostics.json"),
        {"realized_rho": rho, "rank_recall": _thirds(config, ARMS, names)},
    )
    _figure(dists, rho, output_root(config) / "figures" / "prevalence_curve.pdf")
    return path
