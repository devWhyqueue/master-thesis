"""Reporting, LaTeX table generation, and plotting for the influence analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from imbalance_benchmark.common import output_root

from influence import OBJECTIVES, SUPPORTS

__all__ = ["generate_figure", "generate_latex_tables", "write_report"]

_NAMES = {"patch": "Patch-average", "patient": "Patient-average"}
_SUPPORT_NAMES = {"balanced": "Concentrated (C)", "balanced_spread": "Spread (S)"}
_SUPPORT_LETTERS = {"balanced": "C", "balanced_spread": "S"}


def _format_cell(point: float, low: float, high: float) -> str:
    """Format point and 95% CI for LaTeX table."""
    return f"{point:.2f} [{low:.2f}, {high:.2f}]"


def _build_accuracy_table(contrasts: dict[str, Any]) -> str:
    """Build the four accuracies and both coverage benefits B_o."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Objective & $A_{o,\mathrm{C}}$ & $A_{o,\mathrm{S}}$ & Coverage benefit $B_o$ \\",
        r"\midrule",
    ]
    pooled, benefits = contrasts["pooled_accuracies"], contrasts["coverage_benefits"]
    for o in OBJECTIVES:
        ac, aspr, b = pooled["balanced"][o], pooled["balanced_spread"][o], benefits[o]
        lines.append(
            f"{_NAMES[o]} & "
            f"{_format_cell(ac['point'], ac['ci_2_5'], ac['ci_97_5'])} & "
            f"{_format_cell(aspr['point'], aspr['ci_2_5'], aspr['ci_97_5'])} & "
            f"{_format_cell(b['point'], b['ci_2_5'], b['ci_97_5'])} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Patient-macro balanced accuracy (\%) by objective and support, "
            r"with coverage benefit.}",
            r"\label{tab:influence_accuracies}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_contrast_table(contrasts: dict[str, Any]) -> str:
    """Build the weighting gains W_C, W_S, and interaction I."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Contrast & Value \\",
        r"\midrule",
    ]
    gains, ih = contrasts["weighting_gains"], contrasts["interaction"]
    for support in SUPPORTS:
        g = gains[support]
        label = f"Weighting gain $W_{{{_SUPPORT_LETTERS[support]}}}$"
        lines.append(
            f"{label} & {_format_cell(g['point'], g['ci_2_5'], g['ci_97_5'])} \\\\"
        )
    lines.append(r"\midrule")
    lines.append(
        f"Interaction $I$ & {_format_cell(ih['point'], ih['ci_2_5'], ih['ci_97_5'])} \\\\"
    )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Weighting gains by support condition and their interaction "
            r"(percentage points).}",
            r"\label{tab:influence_contrasts}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def _average_audit_rows(
    contribution_audit: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, list[dict[str, Any]]]:
    """Average per-class contribution stats over the three splits, per support."""
    averaged: dict[str, list[dict[str, Any]]] = {}
    for support in SUPPORTS:
        rows_by_class: dict[str, list[dict[str, Any]]] = {}
        for split_rows in contribution_audit.values():
            for row in split_rows[support]:
                rows_by_class.setdefault(row["class"], []).append(row)
        averaged[support] = [
            {
                "class": cls,
                "G_c": float(np.mean([r["G_c"] for r in rows])),
                "max_share": float(np.mean([r["max_share"] for r in rows])),
                "D_c": float(np.mean([r["D_c"] for r in rows])),
            }
            for cls, rows in rows_by_class.items()
        ]
    return averaged


def _build_audit_table(
    contribution_audit: dict[str, dict[str, list[dict[str, Any]]]],
) -> str:
    """Build the per-class contribution audit table (split-averaged)."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Support & Class & $G_c$ & Max share & $D_c$ \\",
        r"\midrule",
    ]
    for support, rows in _average_audit_rows(contribution_audit).items():
        for row in rows:
            lines.append(
                f"{_SUPPORT_NAMES[support]} & {row['class']} & {row['G_c']:.1f} & "
                f"{row['max_share']:.3f} & {row['D_c']:.3f} \\\\"
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Split-averaged per-class contribution audit "
            r"(report eq.~7): contributing patients $G_c$, largest patient share, "
            r"and departure from equal influence $D_c$.}",
            r"\label{tab:influence_audit}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def generate_latex_tables(
    contrasts: dict[str, Any],
    contribution_audit: dict[str, dict[str, list[dict[str, Any]]]],
    out_dir: Path,
) -> None:
    """Generate LaTeX tables for accuracies, contrasts, and the contribution audit."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "table_accuracies.tex").write_text(
        _build_accuracy_table(contrasts), encoding="utf-8"
    )
    (out_dir / "table_contrasts.tex").write_text(
        _build_contrast_table(contrasts), encoding="utf-8"
    )
    (out_dir / "table_audit.tex").write_text(
        _build_audit_table(contribution_audit), encoding="utf-8"
    )


def _plot_objective(
    ax: Any, o: str, label: str, fmt: str, col: str, pooled: dict[str, Any]
) -> None:
    """Plot error bar line for one objective."""
    pt_c, pt_s = pooled["balanced"][o]["point"], pooled["balanced_spread"][o]["point"]
    err_c = [
        [pt_c - pooled["balanced"][o]["ci_2_5"]],
        [pooled["balanced"][o]["ci_97_5"] - pt_c],
    ]
    err_s = [
        [pt_s - pooled["balanced_spread"][o]["ci_2_5"]],
        [pooled["balanced_spread"][o]["ci_97_5"] - pt_s],
    ]
    errs = [[err_c[0][0], err_s[0][0]], [err_c[1][0], err_s[1][0]]]
    ax.errorbar(
        [0, 1], [pt_c, pt_s], yerr=errs, fmt=fmt, color=col, label=label, capsize=4
    )


def generate_figure(
    contrasts: dict[str, Any], dataset_name: str, out_path: Path
) -> None:
    """Plot balanced accuracy with support condition on x, one line per objective."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pooled = contrasts["pooled_accuracies"]
    objectives = [
        ("patch", "Patch-average", "s--", "tab:blue"),
        ("patient", "Patient-average", "o-", "tab:orange"),
    ]
    fig, ax = plt.subplots(figsize=(6, 5))
    for o, label, fmt, col in objectives:
        _plot_objective(ax, o, label, fmt, col, pooled)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Concentrated (C)", "Spread (S)"])
    ax.set_ylabel("Patient-macro balanced accuracy (%)")
    ax.set_title(f"Patient Influence Under Patient Shortage ({dataset_name.upper()})")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _write_json_report(json_path: Path, dataset_name: str, **sections: Any) -> None:
    """Serialize the report sections to ``json_path``."""
    json_path.write_text(
        json.dumps({"dataset": dataset_name, **sections}, indent=2), encoding="utf-8"
    )


def write_report(
    config: dict[str, Any],
    contrasts: dict[str, Any],
    regularization: dict[str, Any],
    contribution_audit: dict[str, dict[str, list[dict[str, Any]]]],
    split_endpoints: dict[str, Any],
) -> Path:
    """Write comprehensive json report, tables, and figure."""
    root_p = output_root(config)
    dataset_name = config.get("dataset", {}).get("name", "unknown")
    tables_dir, figures_dir = root_p / "tables", root_p / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    json_path = tables_dir / "influence.json"
    _write_json_report(
        json_path,
        dataset_name,
        contrasts=contrasts,
        regularization=regularization,
        contribution_audit=contribution_audit,
        split_endpoints=split_endpoints,
    )
    generate_latex_tables(contrasts, contribution_audit, tables_dir)
    generate_figure(
        contrasts, dataset_name, figures_dir / f"influence_{dataset_name}.png"
    )
    return json_path
