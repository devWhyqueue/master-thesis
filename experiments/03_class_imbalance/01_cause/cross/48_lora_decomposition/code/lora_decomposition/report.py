"""Report table and figures rendered from ``analysis.json`` only (no refits, no selection).

The component figure plots each signed contrast with its interval, so a negative interaction is
shown rather than stacked away. The recovery figure sets the mitigation studies' recoveries
(descriptive, earlier runs) beside this experiment's recovery benchmark ``G``; the benchmark band
appears only where the share is usable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from imbalance_benchmark.common import REPO_ROOT

__all__ = ["benchmark_band", "render_report", "table_body"]

_DATASETS = {"bracs": "BRACS", "tcga_ut": "TCGA-UT"}
_ARM_ROWS = (
    ("B", "Balanced reference $B$"),
    ("F", "Frequency only $F$"),
    ("C", "Coverage only $C$"),
    ("R", "Both $R$"),
)
_CONTRAST_ROWS = (
    ("D", "Total damage $D$"),
    ("D_F", "Frequency damage $D_F$"),
    ("D_C", "Coverage damage $D_C$"),
    ("I", "Interaction $I$"),
    ("G", "Recovery $G = C - R$"),
)
_MITIGATION = {
    "bracs": "experiments/03_class_imbalance/02_mitigation/41_mitigation_bracs/report/analysis.json",
    "tcga_ut": "experiments/03_class_imbalance/02_mitigation/43_mitigation_tcga_ut/report/analysis.json",
}
_METHODS = (
    ("la", "LA (train)"),
    ("posthoc_la", "LA (post hoc)"),
    ("cuda", "CUDA"),
    ("bs", "Balanced sampling"),
    ("gcl", "GCL"),
    ("gcl2", "GCL (two-stage)"),
    ("crt", "cRT"),
    ("disalign", "DisAlign"),
)
_BLUE, _ORANGE, _INK, _BAND = "#2a78d6", "#eb6834", "#3d3d3a", "#e4e3df"


def _cell(summary: dict[str, float]) -> str:
    return f"{summary['estimate']:.2f} [{summary['lower']:.2f}, {summary['upper']:.2f}]"


def _share_cell(share: dict[str, Any]) -> str:
    if not share["usable_for_hypothesis"] or share["interval"] is None:
        return "unavailable"
    return _cell(share["interval"])


def table_body(analysis: dict[str, Any]) -> str:
    """Tabular body: condition accuracies (%), contrasts (pp), and the share h per dataset."""
    data = [analysis["datasets"][name] for name in _DATASETS]
    lines = ["Quantity & " + " & ".join(_DATASETS.values()) + r" \\", r"\midrule"]
    lines += [
        f"{label} & " + " & ".join(_cell(d["accuracy"][key]) for d in data) + r" \\"
        for key, label in _ARM_ROWS
    ]
    lines.append(r"\midrule")
    lines += [
        f"{label} & " + " & ".join(_cell(d["contrasts"][key]) for d in data) + r" \\"
        for key, label in _CONTRAST_ROWS
    ]
    lines.append(
        r"Share $h = G / D$ & "
        + " & ".join(_share_cell(d["share"]) for d in data)
        + r" \\"
    )
    return "\n".join(lines) + "\n"


def benchmark_band(result: dict[str, Any]) -> tuple[float, float, float] | None:
    """(estimate, lower, upper) of G when the share is usable, else ``None``."""
    if not result["share"]["usable_for_hypothesis"]:
        return None
    g = result["contrasts"]["G"]
    return g["estimate"], g["lower"], g["upper"]


def _errorbar(
    ax: Any, y: float, summary: dict[str, float], color: str, marker: str
) -> None:
    low, high = (
        summary["estimate"] - summary["lower"],
        summary["upper"] - summary["estimate"],
    )
    ax.errorbar(
        summary["estimate"],
        y,
        xerr=[[low], [high]],
        fmt=marker,
        color=color,
        ms=6,
        lw=1.5,
        capsize=0,
        mfc=color if marker == "o" else "white",
    )


def _component_figure(analysis: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, (name, label) in zip(axes, _DATASETS.items()):
        result = analysis["datasets"][name]
        for y, (key, _) in enumerate(_CONTRAST_ROWS):
            _errorbar(ax, -y, result["contrasts"][key], _BLUE, "o")
        ax.axvline(0, color=_INK, lw=0.8)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Percentage points", fontsize=9)
        ax.grid(axis="x", color=_BAND, lw=0.6)
    axes[0].set_yticks(
        [-y for y in range(len(_CONTRAST_ROWS))],
        [r[1] for r in _CONTRAST_ROWS],
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _mitigation_estimates(name: str) -> dict[str, Any] | None:
    path = REPO_ROOT / _MITIGATION[name]
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["arms"]["r100"]["estimates"]


def _to_summary(entry: dict[str, float]) -> dict[str, float]:
    return {
        "estimate": entry["point"],
        "lower": entry["ci_2_5"],
        "upper": entry["ci_97_5"],
    }


def _recovery_panel(ax: Any, name: str, result: dict[str, Any]) -> None:
    band = benchmark_band(result)
    if band is not None:
        ax.axvspan(band[1], band[2], color=_BAND, lw=0)
        ax.axvline(band[0], color=_INK, lw=1.2)
        h = result["share"]["interval"]["estimate"]
        ax.text(
            band[0],
            0.6,
            f" benchmark $G$ (h = {h:.2f})",
            fontsize=7,
            color=_INK,
            va="bottom",
        )
    else:
        ax.text(
            0.02,
            0.6,
            "benchmark unavailable",
            fontsize=7,
            color=_INK,
            transform=ax.get_yaxis_transform(),
        )
    estimates = _mitigation_estimates(name) or {}
    for y, (method, _) in enumerate(_METHODS):
        if f"recovery_{method}" in estimates:
            _errorbar(
                ax, -y + 0.12, _to_summary(estimates[f"recovery_{method}"]), _BLUE, "o"
            )
        if f"specific_recovery_{method}" in estimates:
            _errorbar(
                ax,
                -y - 0.12,
                _to_summary(estimates[f"specific_recovery_{method}"]),
                _ORANGE,
                "s",
            )
    ax.axvline(0, color=_INK, lw=0.8)
    ax.set_title(_DATASETS[name], fontsize=10)
    ax.set_xlabel("Recovery at $\\rho=100$ (pp)", fontsize=9)


def _recovery_figure(analysis: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharey=True)
    for ax, name in zip(axes, _DATASETS):
        _recovery_panel(ax, name, analysis["datasets"][name])
    axes[0].set_yticks(
        [-y for y in range(len(_METHODS))], [m[1] for m in _METHODS], fontsize=8
    )
    axes[0].set_ylim(-len(_METHODS) + 0.4, 1.0)
    handles = [
        Line2D([], [], color=_BLUE, marker="o", ls="", label="Recovery"),
        Line2D(
            [],
            [],
            color=_ORANGE,
            marker="s",
            mfc="white",
            ls="",
            label="Minus balanced-data gain",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def render_report(analysis_path: Path, report_dir: Path) -> None:
    """Write ``arm_table.tex``, ``component_figure.pdf``, and ``recovery_figure.pdf``."""
    plt.switch_backend("Agg")
    analysis = json.loads(Path(analysis_path).read_text(encoding="utf-8"))
    report_dir = Path(report_dir)
    (report_dir / "arm_table.tex").write_text(table_body(analysis), encoding="utf-8")
    _component_figure(analysis, report_dir / "component_figure.pdf")
    _recovery_figure(analysis, report_dir / "recovery_figure.pdf")
