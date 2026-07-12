"""Per-rank (head/body/tail) discrimination and support-vs-recall analysis."""

import json
import re
from pathlib import Path
from typing import cast

import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd

from analysis.plotting import (
    PATCH_ORDER,
    SEVERITY_COLORS,
    SPLIT_PATTERN,
    WSI_ORDER,
    _method_label,
    _write_unavailable,
    _write_wide_table,
)

TAIL_PARAMS = [0.5, 1.0, 1.5]
TAIL_TIERS = ["head", "body", "tail"]


def tail_class_frame(
    results_dir: Path, output_dir: Path, constructed_dir: Path, native: pd.Series
) -> pd.DataFrame:
    """Join per-class test recall/F1 with native support tier and per-lambda support."""
    selection_path = output_dir / "tuning_selection.json"
    if native.empty or not selection_path.exists():
        return pd.DataFrame()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not selection:
        return pd.DataFrame()
    tier = _native_tiers(native)
    support = _train_support_by_param(constructed_dir)
    rows = [
        row
        for entry in selection
        for row in _tail_entry_rows(entry, results_dir, support, tier)
    ]
    return pd.DataFrame(rows)


def _tail_entry_rows(
    entry: dict,
    results_dir: Path,
    support: dict[tuple[float, int], dict[str, int]],
    tier: dict[str, str],
) -> list[dict[str, object]]:
    param = _regime_param(str(entry["regime"]))
    if param is None:
        return []
    # tuning_selection uses "wsi"/"patch"; the rest of the report uses "wsi_bag".
    benchmark = "wsi_bag" if entry["benchmark"] == "wsi" else entry["benchmark"]
    rows: list[dict[str, object]] = []
    for seed in range(3):
        run = (
            results_dir
            / "tuning"
            / entry["benchmark"]
            / entry["regime"]
            / entry["method"]
            / entry["variant"]
            / f"seed={seed}"
        )
        per_class = _read_per_class(run / "test_results.json")
        if per_class is None:
            continue
        seed_support = support.get((param, seed), {})
        for name, (recall, f1) in per_class.items():
            rows.append(
                {
                    "benchmark": benchmark,
                    "method": entry["method"],
                    "parameter": param,
                    "seed": seed,
                    "class_name": name,
                    "tier": tier.get(name),
                    "train_support": seed_support.get(name),
                    "recall": recall,
                    "f1": f1,
                }
            )
    return rows


def _native_tiers(native: pd.Series) -> dict[str, str]:
    """Assign head/body/tail tiers by native support (top-8 / middle / bottom-8)."""
    names = [str(name) for name in native.index]
    head, tail = set(names[:8]), set(names[-8:])
    return {
        name: "head" if name in head else "tail" if name in tail else "body"
        for name in names
    }


def _train_support_by_param(root: Path) -> dict[tuple[float, int], dict[str, int]]:
    out: dict[tuple[float, int], dict[str, int]] = {}
    for path in sorted(
        root.glob("constructed_order=native_prevalence_parameter=*_seed=*")
    ):
        match = SPLIT_PATTERN.fullmatch(path.name)
        counts_path = path / "target_counts.json"
        if match is None or not counts_path.exists():
            continue
        out[(float(match.group("parameter")), int(match.group("seed")))] = json.loads(
            counts_path.read_text(encoding="utf-8")
        )
    return out


def _read_per_class(path: Path) -> dict[str, tuple[float, float]] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = payload.get("class_names")
    recall = payload.get("recall_per_class")
    f1 = payload.get("f1_per_class")
    if not names or recall is None or f1 is None:
        return None
    return {str(n): (float(recall[i]), float(f1[i])) for i, n in enumerate(names)}


def _regime_param(regime: str) -> float | None:
    match = re.search(r"param=([\d.]+)", regime)
    return float(match.group(1)) if match else None


def write_tail_class_tables(frame: pd.DataFrame, tables_dir: Path) -> None:
    """Write head/body/tail macro-F1 tables for the patch and WSI-bag regimes."""
    for benchmark, stem, order in (
        ("patch", "result_tail_class_patch", PATCH_ORDER),
        ("wsi_bag", "result_tail_class_wsi_bag", WSI_ORDER),
    ):
        part = (
            cast(pd.DataFrame, frame[frame["benchmark"] == benchmark])
            if not frame.empty
            else frame
        )
        _write_tail_class_table(part, tables_dir / f"{stem}.tex", order)


def _write_tail_class_table(part: pd.DataFrame, path: Path, order: list[str]) -> None:
    if part.empty:
        _write_unavailable(
            path, "Method & " + " & ".join(["Head & Body & Tail"] * len(TAIL_PARAMS))
        )
        return
    stats = _tier_stats(part)
    present = set(part["method"])
    methods = [m for m in order if m in present] + [
        m for m in part["method"].unique() if m not in order
    ]
    colspec = "l" + "rrr" * len(TAIL_PARAMS)
    top = "Method & " + " & ".join(
        f"\\multicolumn{{3}}{{c}}{{$\\lambda={p:.1f}$}}" for p in TAIL_PARAMS
    )
    cmidrule = " ".join(
        f"\\cmidrule(lr){{{2 + i * 3}--{4 + i * 3}}}" for i in range(len(TAIL_PARAMS))
    )
    sub = " & ".join([""] + ["Head & Body & Tail"] * len(TAIL_PARAMS))
    rows = [_tail_class_row(method, stats) for method in methods]
    _write_wide_table(path, colspec, [top + f" \\\\ {cmidrule}", sub], rows)


def _tail_class_row(
    method: str, stats: dict[tuple[str, float, str], tuple[float, float]]
) -> str:
    cells = [_method_label(method)]
    for param in TAIL_PARAMS:
        for tier in TAIL_TIERS:
            value = stats.get((method, param, tier))
            cells.append(
                "--"
                if value is None
                else f"\\num{{{value[0]:.3f}}} $\\pm$ \\num{{{value[1]:.3f}}}"
            )
    return " & ".join(cells) + "\\\\"


def _tier_stats(
    part: pd.DataFrame,
) -> dict[tuple[str, float, str], tuple[float, float]]:
    keys = ["method", "parameter", "tier"]
    valid = part.dropna(subset=["tier"])
    seed_mean = valid.groupby([*keys, "seed"])["f1"].mean().reset_index()
    agg = seed_mean.groupby(keys)["f1"].agg(["mean", "std"]).reset_index()
    return {
        (str(r["method"]), float(r["parameter"]), str(r["tier"])): (
            float(r["mean"]),
            0.0 if pd.isna(r["std"]) else float(r["std"]),
        )
        for r in agg.to_dict("records")
    }


def plot_support_vs_recall(frame: pd.DataFrame, path: Path) -> None:
    """Plot per-class test recall against log training support, one panel per regime."""
    if frame.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (benchmark, title) in zip(
        axes, [("patch", "Patch features"), ("wsi_bag", "WSI bag")]
    ):
        _support_recall_panel(ax, frame, benchmark, title)
    axes[0].set_ylabel("Per-class test recall")
    axes[0].legend(fontsize=8, title="Severity")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _support_recall_panel(
    ax: matplotlib.axes.Axes, frame: pd.DataFrame, benchmark: str, title: str
) -> None:
    part = frame[
        (frame["benchmark"] == benchmark)
        & frame["train_support"].notna()
        & frame["recall"].notna()
    ]
    ax.set_xlabel("Training slides per class (log scale)")
    ax.set_title(title)
    if part.empty:  # nothing to plot; avoid log-scaling an empty axis
        return
    agg = (
        part.groupby(["parameter", "class_name"])
        .agg(recall=("recall", "mean"), support=("train_support", "mean"))
        .reset_index()
    )
    for param, color in zip(TAIL_PARAMS, SEVERITY_COLORS):
        sub = agg[agg["parameter"] == param]
        ax.scatter(
            sub["support"],
            sub["recall"],
            s=18,
            alpha=0.6,
            color=color,
            label=f"$\\lambda={param:.1f}$",
        )
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(True, alpha=0.3)
