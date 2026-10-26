"""Render ``report/results.tex`` mechanically from ``analysis.json``; no outcome prose is invented."""

from __future__ import annotations

from typing import Any

__all__ = ["render_results"]

_NAMES = {
    "bracs": "BRACS",
    "tcga_ut": "TCGA-UT",
    "la": "Train-time LA",
    "posthoc_la": "Post-hoc LA",
    "cuda": "CUDA",
    "ce": "CE control",
}
_DECISION = {
    "noninferior": "noninferior",
    "lora_advantage": "LoRA advantage",
    "inconclusive": "inconclusive",
}
_VERDICT = {
    "omission_supported": (
        "All six contrasts are noninferior: omitting LoRA caused no material accuracy loss "
        "for these methods in the tested scope."
    ),
    "not_supported": (
        "At least one contrast is not noninferior, so omitting LoRA is not supported for the "
        "full tested scope; the individual outcomes stand on their own."
    ),
}


def _ba(analysis: dict[str, Any], rep: str, dataset: str, method: str) -> str:
    return (
        f"{analysis['datasets'][dataset]['absolute_ba'][rep][method]['estimate']:.2f}"
    )


def _contrast_row(analysis: dict[str, Any], dataset: str, method: str) -> str:
    primary = analysis["datasets"][dataset]["primary"][method]
    cells = [
        _NAMES[dataset],
        _NAMES[method],
        _ba(analysis, "frozen", dataset, method),
        _ba(analysis, "adapted", dataset, method),
        f"{primary['estimate']:.2f}",
        f"[{primary['lower']:.2f}, {primary['upper']:.2f}]",
        _DECISION[primary["decision"]],
    ]
    return " & ".join(cells) + r" \\"


def _resource_row(analysis: dict[str, Any], dataset: str, method: str) -> str:
    row = analysis["datasets"][dataset]["resources"][method]
    memory = (
        row["peak_gpu_mb"] if row["peak_gpu_mb"] is not None else row["peak_rss_mb"]
    )
    cells = [
        _NAMES[dataset],
        _NAMES[method],
        str(row["fits"]),
        f"{row['seconds_mean']:.1f}",
        "--" if memory is None else f"{memory:.0f}",
    ]
    return " & ".join(cells) + r" \\"


def _table(spec: str, header: str, rows: str, caption: str, label: str) -> str:
    """One booktabs table with a caption and label."""
    return "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            r"\renewcommand{\arraystretch}{1.15}",
            rf"\begin{{tabular}}{{@{{}}{spec}@{{}}}}",
            r"\toprule",
            header + r" \\",
            r"\midrule",
            rows,
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
        ]
    )


_CONTRAST_CAPTION = (
    r"Balanced accuracy (BA, \%) at $\rho=100$ without and with LoRA, and their paired difference "
    r"$\delta$ with a 99.17\% percentile interval (Bonferroni over six contrasts). A lower bound "
    r"above $-1$ is noninferior; an upper bound below $-1$ is a LoRA advantage; anything else is "
    r"inconclusive."
)
_RESOURCE_CAPTION = (
    "Measured cost of the new fits (GPU peak where recorded, otherwise host peak). "
    "Post-hoc LA only scores the stored CE head."
)


def _intro(analysis: dict[str, Any]) -> str:
    datasets = list(analysis["datasets"])
    controls = ", ".join(
        f"{_NAMES[d]} {_ba(analysis, 'frozen', d, 'ce')} (frozen) and {_ba(analysis, 'adapted', d, 'ce')} (adapted)"
        for d in datasets
    )
    return "\n".join(
        [
            r"\section{Results}",
            r"\label{sec:results}",
            rf"Table~\ref{{tab:contrast}} reports the six primary contrasts at $\rho=100$. {_VERDICT[analysis['joint_verdict']]}",
            f"The CE controls reach {controls} balanced accuracy.",
        ]
    )


def render_results(analysis: dict[str, Any]) -> str:
    """The results section: contrast table, joint verdict, and measured resources, from the analysis alone."""
    datasets = list(analysis["datasets"])
    contrasts = "\n".join(
        _contrast_row(analysis, d, m)
        for d in datasets
        for m in ("la", "posthoc_la", "cuda")
    )
    resources = "\n".join(
        _resource_row(analysis, d, m)
        for d in datasets
        for m in ("ce", "la", "posthoc_la", "cuda")
    )
    header = r"Dataset & Method & Without LoRA & With LoRA & $\delta$ (pp) & 99.17\% interval & Decision"
    first = _table("llrrrrl", header, contrasts, _CONTRAST_CAPTION, "tab:contrast")
    second = _table(
        "llrrr",
        r"Dataset & Candidate & Fits & Mean seconds & Peak memory (MB)",
        resources,
        _RESOURCE_CAPTION,
        "tab:resources",
    )
    return "\n\n".join([_intro(analysis), first, second]) + "\n"
