"""Render ``report/results.tex`` mechanically from ``analysis.json``; no outcome prose is invented."""

from __future__ import annotations

from typing import Any

__all__ = ["render_results"]

_NAMES = {
    "bracs": "BRACS",
    "tcga_ut": "TCGA-UT",
    "la": "LA",
    "bs": "BS",
    "disalign": "DisAlign",
    "crt": "cRT",
    "gcl": "GCL",
    "gcl2": "GCL2",
    "cuda": r"CUDA (vs CE$_\mathrm{SGD}$)",
}
_GROUPS = (("la", "bs", "disalign"), ("crt", "gcl", "gcl2"), ("cuda",))
_BOLD_COLUMNS = ("R", "share", "R_minus_G")
_MATERIAL = (
    r"$^{*}$ material (point estimate $\geq 1$ percentage point, lower bound $> 0$)."
)
_BASELINE_HEAD = (
    r"\begin{table}[t]\centering\small",
    r"\caption{Cross-entropy (CE) references: patient-macro balanced accuracy (BA, \%) for"
    r" logistic regression (LR) and stochastic gradient descent (SGD). r1 and r100 denote"
    r" class ratios $\rho=1$ and $\rho=100$. Damage $D$ = BA(r1) $-$ BA(r100), with"
    r" 95\,\% intervals; " + _MATERIAL + "}",
    r"\label{tab:baselines}",
    r"\begin{tabular}{lrrlrrl}\toprule",
    r"Dataset & LR r1 & LR r100 & $D$ & SGD r1 & SGD r100 & $D_\mathrm{SGD}$ \\ \midrule",
)
_CONTRAST_HEAD = (
    r"\begin{table}[t]\centering\small",
    r"\caption{Recovery of the class-imbalance damage at $\rho=100$, with 95\,\% intervals."
    r" Contrasts are in percentage points of balanced accuracy; $R/D$ is the share of the"
    r" damage $D$ recovered. LA, BS and DisAlign use frozen features; cRT, GCL and GCL2"
    r" use LoRA; CUDA is compared with its SGD-trained reference and $D_\mathrm{SGD}$."
    r" Bold: largest point estimate per dataset and column. "
    + _MATERIAL
    + r" $^{\dagger}$ zero by construction (equal class counts at $\rho=1$).}",
    r"\label{tab:contrasts}",
    r"\begin{tabular}{@{}lllll@{}}\toprule",
    r" & Recovery & Share of damage & Balanced gain & Imbalance-specific \\",
    r"Method & $R$ & $R/D$ & $G$ & $R-G$ \\ \midrule",
)
_TAIL = "\\bottomrule\\end{tabular}\\end{table}\n"


def _cell(
    summary: dict[str, Any], *, bold: bool = False, mark_material: bool = True
) -> str:
    point = f"{summary['point']:.2f}"
    if bold:
        point = rf"\textbf{{{point}}}"
    material = mark_material and summary.get("verdict") == "material"
    mark = r"$^{*}$" if material else ""
    cell = f"{point} [{summary['lower']:.2f}, {summary['upper']:.2f}]{mark}"
    return cell.replace("-", "$-$")


def _row(method: str, block: dict[str, Any], best: dict[str, str]) -> str:
    g_cell = r"0$^{\dagger}$" if block.get("G_by_construction") else _cell(block["G"])
    cells = [
        _NAMES[method],
        _cell(block["R"], bold=best["R"] == method),
        _cell(block["share"], bold=best["share"] == method, mark_material=False),
        g_cell,
        _cell(block["R_minus_G"], bold=best["R_minus_G"] == method),
    ]
    return " & ".join(cells) + r" \\"


def _dataset_rows(dataset: str, analysis: dict[str, Any]) -> list[str]:
    methods = analysis["methods"]
    best = {
        column: max(methods, key=lambda m, c=column: methods[m][c]["point"])
        for column in _BOLD_COLUMNS
    }
    base = analysis["baseline"]
    damage = rf"$D={base['D']['point']:.2f}$, $D_\mathrm{{SGD}}={base['D_sgd']['point']:.2f}$"
    rows = [rf"\multicolumn{{5}}{{@{{}}l}}{{\emph{{{_NAMES[dataset]}}} ({damage})}} \\"]
    for index, group in enumerate(_GROUPS):
        if index:
            rows.append(r"\addlinespace[2pt]")
        rows.extend(_row(m, methods[m], best) for m in group)
    return rows


def _baseline_row(dataset: str, analysis: dict[str, Any]) -> str:
    base = analysis["baseline"]
    ba = base["ba"]
    cells = [
        _NAMES[dataset],
        f"{ba['ce:r1']['point']:.2f}",
        f"{ba['ce:r100']['point']:.2f}",
        _cell(base["D"]),
        f"{ba['sgd_ce:r1']['point']:.2f}",
        f"{ba['sgd_ce:r100']['point']:.2f}",
        _cell(base["D_sgd"]),
    ]
    return " & ".join(cells) + r" \\"


def render_results(analysis: dict[str, Any]) -> str:
    """Baseline table (logistic-regression and SGD CE) and the per-method contrast table."""
    datasets = analysis["datasets"]
    baseline = [_baseline_row(d, datasets[d]) for d in datasets]
    rows: list[str] = []
    for index, dataset in enumerate(datasets):
        if index:
            rows.append(r"\midrule")
        rows.extend(_dataset_rows(dataset, datasets[dataset]))
    return "\n".join([*_BASELINE_HEAD, *baseline, _TAIL, *_CONTRAST_HEAD, *rows, _TAIL])
