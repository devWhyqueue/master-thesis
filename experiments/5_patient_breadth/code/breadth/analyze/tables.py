"""Shared cell formatting and the prespecified secondary endpoint tables."""

from __future__ import annotations

from typing import Any

from breadth import BREADTH_LADDER, DEPTH_LADDER

__all__ = [
    "SECONDARY_COLUMNS",
    "build_class_recall_table",
    "build_secondary_table",
    "format_cell",
]

SECONDARY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("macro_nll", r"Macro NLL (nats)"),
    ("expected_calibration_error", r"ECE (\%)"),
    ("patch_micro_balanced_accuracy", r"Patch-micro BA (\%)"),
)


def format_cell(point: float, low: float, high: float) -> str:
    """Format point and 95% CI for LaTeX table."""
    return f"{point:.2f} [{low:.2f}, {high:.2f}]"


def _row(label: str, estimates: dict[str, Any]) -> str:
    """Format one table row of the three scalar secondary endpoints."""
    values = [
        format_cell(
            estimates[key]["point"], estimates[key]["ci_2_5"], estimates[key]["ci_97_5"]
        )
        for key, _ in SECONDARY_COLUMNS
    ]
    return f"{label} & " + " & ".join(values) + r" \\"


def _preamble(column_spec: str, header: str) -> list[str]:
    """Opening lines of a booktabs table up to its header rule."""
    return [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{" + column_spec + "}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]


def _closing(caption: str, label: str) -> list[str]:
    """Closing lines of a booktabs table from its bottom rule onwards."""
    return [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\end{table}",
    ]


def build_secondary_table(secondary: dict[str, Any]) -> str:
    """Build LaTeX table of secondary endpoints per cell and their contrast."""
    header = r"Cell $(G, m)$ & " + " & ".join(h for _, h in SECONDARY_COLUMNS)
    lines = _preamble("l" + "c" * len(SECONDARY_COLUMNS), header)
    lines.extend(
        _row(f"({g}, {m})", secondary["cells"][f"G{g}_m{m}"])
        for g in BREADTH_LADDER
        for m in DEPTH_LADDER
    )
    lines.append(r"\midrule")
    lines.append(_row(r"$X$: $(20, 8) - (5, 32)$", secondary["equal_budget_X"]))
    lines.extend(
        _closing(
            r"Secondary endpoints per allocation cell with 95\% bootstrap intervals. "
            r"Negative $X$ favours the broad allocation for NLL and ECE.",
            "tab:breadth_secondary",
        )
    )
    return "\n".join(lines) + "\n"


def build_class_recall_table(secondary: dict[str, Any]) -> str:
    """Build LaTeX table of class-specific patient-macro recalls at the $X$ cells."""
    lines = _preamble("lccc", r"Class & $(5, 32)$ & $(20, 8)$ & $X$")
    for name, est in secondary["class_recalls"].items():
        cells = [
            format_cell(est[k]["point"], est[k]["ci_2_5"], est[k]["ci_97_5"])
            for k in ("G5_m32", "G20_m8", "equal_budget_X")
        ]
        lines.append(f"{name.replace('_', ' ')} & " + " & ".join(cells) + r" \\")
    lines.extend(
        _closing(
            r"Class-specific patient-macro recall (\%) at the two equal-budget cells "
            r"and their difference $X$.",
            "tab:breadth_class_recalls",
        )
    )
    return "\n".join(lines) + "\n"
