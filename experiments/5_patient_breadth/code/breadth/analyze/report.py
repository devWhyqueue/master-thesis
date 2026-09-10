"""LaTeX tables and report writing for the patient-breadth experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.common import output_root

from breadth import BREADTH_LADDER, DEPTH_LADDER
from breadth.analyze.figures import generate_figures

__all__ = ["generate_latex_tables", "generate_figures", "write_report"]


def _format_cell(point: float, low: float, high: float) -> str:
    """Format point and 95% CI for LaTeX table."""
    return f"{point:.2f} [{low:.2f}, {high:.2f}]"


def _build_grid_table(cell_accs: dict[str, dict[str, Any]]) -> str:
    """Build LaTeX table of cell accuracies across the 3x3 grid."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Patches per patient $m$} \\",
        r"\cmidrule(l){2-4}",
        r"Patients per class $G$ & 8 & 16 & 32 \\",
        r"\midrule",
    ]
    for g in BREADTH_LADDER:
        row_cells = []
        for m in DEPTH_LADDER:
            c = cell_accs[f"G{g}_m{m}"]
            row_cells.append(_format_cell(c["point"], c["ci_2_5"], c["ci_97_5"]))
        lines.append(f"{g} & " + " & ".join(row_cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Patient-macro balanced accuracy (\%) on the $3\times 3$ allocation grid with 95\% bootstrap intervals.}",
            r"\label{tab:breadth_grid_accuracies}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_contrast_table(contrasts: dict[str, Any]) -> str:
    """Build LaTeX table of depth gains, breadth gains, and equal-budget contrast."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Contrast & Estimate [95\% CI] (pp) \\",
        r"\midrule",
        r"\multicolumn{2}{l}{\textit{Depth gains at fixed breadth $\Delta_m(G) = A(G, 32) - A(G, 8)$}} \\",
    ]
    for g, est in contrasts["delta_m"].items():
        lines.append(f"$\\Delta_m(G={g})$ & {_format_cell(est['point'], est['ci_2_5'], est['ci_97_5'])} \\\\")

    lines.extend([
        r"\midrule",
        r"\multicolumn{2}{l}{\textit{Breadth gains at fixed depth $\Delta_G(m) = A(20, m) - A(5, m)$}} \\",
    ])
    for m, est in contrasts["delta_g"].items():
        lines.append(f"$\\Delta_G(m={m})$ & {_format_cell(est['point'], est['ci_2_5'], est['ci_97_5'])} \\\\")

    lines.append(r"\midrule")
    x_est = contrasts["equal_budget_advantage_X"]
    lines.append(f"Equal-budget advantage $X = A(20, 8) - A(5, 32)$ & {_format_cell(x_est['point'], x_est['ci_2_5'], x_est['ci_97_5'])} \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Key contrasts on the allocation grid (percentage points).}",
            r"\label{tab:breadth_contrasts}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_surface_table(surface_params: dict[str, Any]) -> str:
    """Build LaTeX table of support surface regression parameters."""
    p = surface_params
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & Predictors & Slope $\beta$ & Residual breadth $\gamma$ & Residual SD \\",
        r"\midrule",
        f"Nominal & $\\log n$ & {_format_cell(p['beta_n']['point'], p['beta_n']['ci_2_5'], p['beta_n']['ci_97_5'])} & -- & {p['res_std_n']['point']:.2f} \\\\",
        f"Breadth only & $\\log G$ & {_format_cell(p['beta_g']['point'], p['beta_g']['ci_2_5'], p['beta_g']['ci_97_5'])} & -- & {p['res_std_g']['point']:.2f} \\\\",
        f"Effective & $\\log \\Neff$ & {_format_cell(p['beta_neff']['point'], p['beta_neff']['ci_2_5'], p['beta_neff']['ci_97_5'])} & -- & {p['res_std_neff']['point']:.2f} \\\\",
        r"\midrule",
        f"Augmented nominal & $\\log n + \\gamma_n \\log G$ & {_format_cell(p['beta_n']['point'], p['beta_n']['ci_2_5'], p['beta_n']['ci_97_5'])} & {_format_cell(p['gamma_n']['point'], p['gamma_n']['ci_2_5'], p['gamma_n']['ci_97_5'])} & {p['res_std_aug_nom']['point']:.2f} \\\\",
        f"Augmented effective & $\\log \\Neff + \\gamma_e \\log G$ & {_format_cell(p['beta_neff']['point'], p['beta_neff']['ci_2_5'], p['beta_neff']['ci_97_5'])} & {_format_cell(p['gamma_e']['point'], p['gamma_e']['ci_2_5'], p['gamma_e']['ci_97_5'])} & {p['res_std_aug_eff']['point']:.2f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Support surface regression fits over the nine grid cells.}",
        r"\label{tab:breadth_surface_fits}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def _build_icc_table(cohort_iccs: dict[str, float]) -> str:
    """Build LaTeX table of class-level ICC estimates and design effects."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Class & $\ICC_c$ & $\DE_c(m=8)$ & $\DE_c(m=32)$ \\",
        r"\midrule",
    ]
    for c_name, icc_val in sorted(cohort_iccs.items()):
        de_8 = 1.0 + 7.0 * icc_val
        de_32 = 1.0 + 31.0 * icc_val
        lines.append(f"{c_name.replace('_', ' ')} & {icc_val:.4f} & {de_8:.2f} & {de_32:.2f} \\\\")

    mean_icc = float(np.mean(list(cohort_iccs.values())))
    lines.append(r"\midrule")
    lines.append(f"Mean & {mean_icc:.4f} & {1.0 + 7.0 * mean_icc:.2f} & {1.0 + 31.0 * mean_icc:.2f} \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Estimated intraclass correlation $\ICC_c$ and design effects by class.}",
            r"\label{tab:breadth_class_iccs}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def generate_latex_tables(
    results: dict[str, Any], cohort_iccs: dict[str, float], dest_dir: Path
) -> None:
    """Write all generated LaTeX tables into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "grid_accuracies.tex").write_text(
        _build_grid_table(results["cell_accuracies"]), encoding="utf-8"
    )
    (dest_dir / "contrasts.tex").write_text(
        _build_contrast_table(results["contrasts"]), encoding="utf-8"
    )
    (dest_dir / "surface_fits.tex").write_text(
        _build_surface_table(results["surface_parameters"]), encoding="utf-8"
    )
    (dest_dir / "class_iccs.tex").write_text(
        _build_icc_table(cohort_iccs), encoding="utf-8"
    )


def write_report(
    config: dict[str, Any],
    results: dict[str, Any],
    cohort_iccs: dict[str, float],
) -> Path:
    """Save analysis JSON and LaTeX tables/figures."""
    root_p = output_root(config)
    data_dir = root_p / "data"
    tables_dir = root_p / "report" / "tables"
    figures_dir = root_p / "report" / "figures"

    data_dir.mkdir(parents=True, exist_ok=True)
    report_p = data_dir / "analysis.json"
    report_p.write_text(json.dumps(results, indent=2), encoding="utf-8")

    generate_latex_tables(results, cohort_iccs, tables_dir)
    generate_figures(results, figures_dir)
    return report_p

