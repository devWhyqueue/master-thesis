"""Reporting, LaTeX table generation, and plotting for decodability analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from imbalance_benchmark.common import output_root

__all__ = ["generate_figure", "generate_latex_tables", "write_report"]

_NAMES = {
    "mlp": "Original MLP",
    "logreg": "Logistic regression",
    "knn": "Cosine $k$-NN",
}


def _format_cell(point: float, low: float, high: float) -> str:
    """Format point and 95% CI for LaTeX table."""
    return f"{point:.2f} [{low:.2f}, {high:.2f}]"


def _build_accuracy_table(contrasts: dict[str, Any]) -> str:
    """Build LaTeX accuracy and coverage benefit table content."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Readout & $A_{h,\mathrm{C}}$ & $A_{h,\mathrm{S}}$ & Benefit $B_h$ \\",
        r"\midrule",
    ]
    pooled, benefits = contrasts["pooled_accuracies"], contrasts["coverage_benefits"]
    for r in ("mlp", "logreg", "knn"):
        ac, aspr, b = pooled["balanced"][r], pooled["balanced_spread"][r], benefits[r]
        row = (
            f"{_NAMES[r]} & "
            f"{_format_cell(ac['point'], ac['ci_2_5'], ac['ci_97_5'])} & "
            f"{_format_cell(aspr['point'], aspr['ci_2_5'], aspr['ci_97_5'])} & "
            f"{_format_cell(b['point'], b['ci_2_5'], b['ci_97_5'])} \\\\"
        )
        lines.append(row)
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Patient-macro balanced accuracy (\%) and coverage benefit by readout.}",
            r"\label{tab:decodability_accuracies}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_contrast_table(contrasts: dict[str, Any]) -> str:
    """Build LaTeX contrast and interaction table content."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Probe & Shortage gain $G_{h,\mathrm{C}}$ & Spread gain $G_{h,\mathrm{S}}$ & Interaction $I_h$ \\",
        r"\midrule",
    ]
    gains, interactions = contrasts["classifier_gains"], contrasts["interactions"]
    for r in ("logreg", "knn"):
        gc, gspr, ih = (
            gains["balanced"][r],
            gains["balanced_spread"][r],
            interactions[r],
        )
        row = (
            f"{_NAMES[r]} & "
            f"{_format_cell(gc['point'], gc['ci_2_5'], gc['ci_97_5'])} & "
            f"{_format_cell(gspr['point'], gspr['ci_2_5'], gspr['ci_97_5'])} & "
            f"{_format_cell(ih['point'], ih['ci_2_5'], ih['ci_97_5'])} \\\\"
        )
        lines.append(row)
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Classifier gains over MLP and interaction effects (percentage points).}",
            r"\label{tab:decodability_contrasts}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def generate_latex_tables(contrasts: dict[str, Any], out_dir: Path) -> None:
    """Generate LaTeX tables for main decodability results and contrasts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "table_accuracies.tex").write_text(
        _build_accuracy_table(contrasts), encoding="utf-8"
    )
    (out_dir / "table_contrasts.tex").write_text(
        _build_contrast_table(contrasts), encoding="utf-8"
    )


def _plot_readout(
    ax: Any, r: str, label: str, fmt: str, col: str, pooled: dict[str, Any]
) -> None:
    """Plot error bar line for one readout."""
    pt_c, pt_s = pooled["balanced"][r]["point"], pooled["balanced_spread"][r]["point"]
    err_c = [
        [pt_c - pooled["balanced"][r]["ci_2_5"]],
        [pooled["balanced"][r]["ci_97_5"] - pt_c],
    ]
    err_s = [
        [pt_s - pooled["balanced_spread"][r]["ci_2_5"]],
        [pooled["balanced_spread"][r]["ci_97_5"] - pt_s],
    ]
    errs = [[err_c[0][0], err_s[0][0]], [err_c[1][0], err_s[1][0]]]
    ax.errorbar(
        [0, 1], [pt_c, pt_s], yerr=errs, fmt=fmt, color=col, label=label, capsize=4
    )


def generate_figure(
    contrasts: dict[str, Any], dataset_name: str, out_path: Path
) -> None:
    """Plot balanced accuracy with support condition on x, one line per readout."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pooled = contrasts["pooled_accuracies"]
    readouts = [
        ("mlp", "MLP (CE)", "o-", "black"),
        ("logreg", "Logistic probe", "s--", "tab:blue"),
        ("knn", "Cosine k-NN", "^:", "tab:green"),
    ]
    fig, ax = plt.subplots(figsize=(6, 5))
    for r, label, fmt, col in readouts:
        _plot_readout(ax, r, label, fmt, col, pooled)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Concentrated (C)", "Spread (S)"])
    ax.set_ylabel("Patient-macro balanced accuracy (%)")
    ax.set_title(f"Class Decodability Under Patient Shortage ({dataset_name.upper()})")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def write_report(
    config: dict[str, Any],
    contrasts: dict[str, Any],
    selection: dict[str, Any],
    split_endpoints: dict[str, Any],
) -> Path:
    """Write comprehensive json report, tables, and figure."""
    root_p = output_root(config)
    dataset_name = config.get("dataset", {}).get("name", "unknown")
    tables_dir, figures_dir = root_p / "tables", root_p / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    json_path = tables_dir / "decodability.json"
    json_path.write_text(
        json.dumps(
            {
                "dataset": dataset_name,
                "contrasts": contrasts,
                "selection": selection,
                "split_endpoints": split_endpoints,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    generate_latex_tables(contrasts, tables_dir)
    generate_figure(
        contrasts, dataset_name, figures_dir / f"decodability_{dataset_name}.png"
    )
    return json_path
