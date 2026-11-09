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
    "cuda": "CUDA",
}
_ORDER = ("la", "bs", "disalign", "crt", "gcl", "gcl2", "cuda")
_BASELINE_HEAD = (
    r"\begin{table}[t]\centering\small",
    r"\caption{CE baselines (patient-macro BA, \%). $D$ = BA(r1) $-$ BA(r100), 95\,\% interval;"
    r" $^{*}$ material (point $\geq 1$ pp, lower bound $> 0$).}",
    r"\label{tab:baselines}",
    r"\begin{tabular}{lrrlrrl}\toprule",
    r"Dataset & LR r1 & LR r100 & $D$ & SGD r1 & SGD r100 & $D_\mathrm{SGD}$ \\ \midrule",
)
_CONTRAST_HEAD = (
    r"\begin{table}[t]\centering\small",
    r"\caption{Recovery $R$, share $R/D$, balanced gain $G$ and imbalance-specific recovery"
    r" $R-G$ (pp, 95\,\% interval; CUDA against CE$_\mathrm{SGD}$). $^{*}$ material.}",
    r"\label{tab:contrasts}",
    r"\begin{tabular}{llllll}\toprule",
    r"Dataset & Method & $R$ & $R/D$ & $G$ & $R-G$ \\ \midrule",
)
_TAIL = "\\bottomrule\\end{tabular}\\end{table}\n"


def _cell(summary: dict[str, Any]) -> str:
    mark = r"$^{*}$" if summary.get("verdict") == "material" else ""
    return (
        f"{summary['point']:.2f} [{summary['lower']:.2f}, {summary['upper']:.2f}]{mark}"
    )


def _row(dataset: str, method: str, block: dict[str, Any]) -> str:
    share = block["share"]
    share_cell = f"{share['point']:.2f} [{share['lower']:.2f}, {share['upper']:.2f}]"
    g_cell = (
        "0 (by construction)" if block.get("G_by_construction") else _cell(block["G"])
    )
    cells = [
        _NAMES[dataset],
        _NAMES[method],
        _cell(block["R"]),
        share_cell,
        g_cell,
        _cell(block["R_minus_G"]),
    ]
    return " & ".join(cells) + r" \\"


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
    rows = [_row(d, m, datasets[d]["methods"][m]) for d in datasets for m in _ORDER]
    return "\n".join([*_BASELINE_HEAD, *baseline, _TAIL, *_CONTRAST_HEAD, *rows, _TAIL])
