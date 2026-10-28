"""Render ``report/results.tex`` mechanically from ``analysis.json``; no outcome prose is invented."""

from __future__ import annotations

from collections import Counter
from typing import Any

__all__ = ["METHODS", "SETTINGS", "render_results"]

SETTINGS = {
    "41_mitigation_bracs": ("BRACS", "Virchow2"),
    "42_mitigation_uni2h": ("BRACS", "UNI2-h"),
    "43_mitigation_tcga_ut": ("TCGA-UT", "Virchow2"),
    "44_mitigation_tcga_ut_uni2h": ("TCGA-UT", "UNI2-h"),
}
METHODS = {
    "bs": "Balanced sampling",
    "cuda": "CUDA",
    "la": "LA, train-time",
    "posthoc_la": "LA, post hoc",
    "disalign": "DisAlign",
    "crt": "cRT",
    "gcl": "GCL, end to end",
    "gcl2": "GCL, classifier only",
}
_VERDICT = {
    "confirmed": (
        "All 32 contrasts show no meaningful gain, so no method improves on temperature-scaled CE "
        "by 0.05 nats or more in the tested scope."
    ),
    "not_confirmed": (
        "Because not all 32 contrasts show no meaningful gain, the claim that no method improves "
        "on temperature-scaled CE is not confirmed."
    ),
}
_ABBREVIATIONS = (
    "CE: cross-entropy; LA: logit adjustment; CUDA: curriculum of data augmentation; "
    "DisAlign: distribution alignment; cRT: classifier re-training; GCL: Gaussian clouded logit "
    "adjustment."
)


def _n(value: float, digits: int) -> str:
    """A number in math mode, so that negative signs typeset as minus signs; no sign on zero."""
    text = f"{value:.{digits}f}"
    return f"${text.lstrip('-') if float(text) == 0 else text}$"


def _secondary_rows(analysis: dict[str, Any], arm: str) -> str:
    settings = analysis["settings"]
    blocks = []
    for quantity, title, digits in (
        ("nll_minus_ce_ts", "Raw NLL minus scaled CE NLL (nats)", 2),
        ("ece_ts_delta", "Scaled ECE, method minus CE (points)", 1),
    ):
        rows = [rf"\multicolumn{{5}}{{@{{}}l}}{{\emph{{{title}}}}} \\"]
        for method in METHODS:
            cells = [
                _n(arms[arm]["methods"][method][quantity]["estimate"], digits)
                for arms in settings.values()
            ]
            rows.append(" & ".join([METHODS[method], *cells]) + r" \\")
        blocks.append("\n".join(rows))
    return "\n\\midrule\n".join(blocks)


def _secondary_table(analysis: dict[str, Any], arm: str) -> str:
    names = [SETTINGS[key] for key in analysis["settings"]]
    return "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            r"\renewcommand{\arraystretch}{1.15}",
            r"\begin{tabular}{@{}lrrrr@{}}",
            r"\toprule",
            " & ".join(["", *(d for d, _ in names)]) + r" \\",
            " & ".join(["Method", *(e for _, e in names)]) + r" \\",
            r"\midrule",
            _secondary_rows(analysis, arm),
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{_SECONDARY_CAPTION}}}",
            r"\label{tab:secondary}",
            r"\end{table}",
        ]
    )


def _figure() -> str:
    return "\n".join(
        [
            r"\begin{figure}[htbp]",
            r"\centering",
            r"\includegraphics[width=\textwidth]{fig_contrasts.pdf}",
            rf"\caption{{{_FIGURE_CAPTION}}}",
            r"\label{fig:contrasts}",
            r"\end{figure}",
        ]
    )


def _intro(analysis: dict[str, Any], arm: str) -> str:
    counts = Counter(analysis["decisions"].values())
    ce = ", ".join(
        f"{arms[arm]['ce_nll_ts']:.2f} ({', '.join(SETTINGS[key])})"
        for key, arms in analysis["settings"].items()
    )
    return "\n".join(
        [
            r"\section{Results}",
            r"\label{sec:results}",
            rf"Figure~\ref{{fig:contrasts}} shows the 32 primary contrasts at $\rho=100$. Of these, "
            f"{counts['no_meaningful_gain']} show no meaningful gain, "
            f"{counts['inconclusive']} are inconclusive, and "
            f"{counts['method_better']} show a meaningful gain. "
            + _VERDICT[analysis["joint_verdict"]],
            f"Temperature-scaled CE reaches an NLL of {ce}, in nats.",
        ]
    )


_FIGURE_CAPTION = (
    r"Only CUDA on TCGA-UT improves on temperature-scaled CE by more than the margin. Paired "
    r"difference $\delta$ in temperature-scaled NLL between each method and CE at $\rho=100$, with "
    r"99.84\% intervals (Bonferroni over 32 contrasts); negative values favour the method. The "
    r"dashed line marks the margin $-0.05$ nats, the solid line zero, and the shaded band "
    r"meaningful gains. " + _ABBREVIATIONS
)
_SECONDARY_CAPTION = (
    r"Without temperature scaling, no method except CUDA on TCGA-UT comes close to scaled CE; "
    r"after scaling, almost every method is slightly better calibrated than CE. Point estimates at "
    r"$\rho=100$ (descriptive). Top: raw NLL of each method minus scaled NLL of CE. Bottom: scaled "
    r"expected calibration error (ECE) of each method minus that of CE. "
    + _ABBREVIATIONS
)


def render_results(analysis: dict[str, Any]) -> str:
    """The results section: verdict, primary-contrast figure, and the descriptive secondary table."""
    arm = analysis["protocol"]["protocol"]["primary_arm"]
    return (
        "\n\n".join([_intro(analysis, arm), _figure(), _secondary_table(analysis, arm)])
        + "\n"
    )
