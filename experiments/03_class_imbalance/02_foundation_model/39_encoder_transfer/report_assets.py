"""Build the exp-39 report's table fragments from the accepted analysis JSON.

``--sync`` first copies each dataset's ``analysis.json``, ``diagnostics.json`` and
``report_diagnostics.json`` into ``report/`` (``<name>_<dataset>.json``); every number in
the generated tables and figures (``report_figures.py``) is then read from those copies, never typed by hand.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

EXP = Path(__file__).resolve().parent
REPORT = EXP / "report"
DATASETS = {"tcga_ut": "TCGA-UT", "bracs": "BRACS"}
ENCODERS = {"virchow2": "Virchow2", "uni2h": "UNI2-h"}
SOURCES = ("analysis", "diagnostics", "report_diagnostics")
COMPONENTS = (("DP", "$D_P$"), ("DS", "$D_S$"), ("I", "$I$"), ("DR", "$D_R$"))
PROB_METRICS = (
    ("nll", "NLL raw"),
    ("nll_ts", "NLL scaled"),
    ("ece", "ECE raw"),
    ("ece_ts", "ECE scaled"),
)


def sync() -> None:
    """Copy the accepted per-dataset analysis outputs next to the report."""
    for ds in DATASETS:
        data = EXP / "outputs" / ds / "patch" / "data"
        for name in SOURCES:
            shutil.copyfile(data / f"{name}.json", REPORT / f"{name}_{ds}.json")


def load() -> dict[str, dict[str, Any]]:
    """Report-local JSON copies keyed by dataset, then source name."""
    return {
        ds: {
            name: json.loads((REPORT / f"{name}_{ds}.json").read_text(encoding="utf-8"))
            for name in SOURCES
        }
        for ds in DATASETS
    }


def num(x: float | None, digits: int = 2) -> str:
    """Math-mode number with a true minus sign."""
    if x is None:
        return "--"
    return f"${x:.{digits}f}$"


def ci(est: dict[str, float], lo: str = "ci_2_5", hi: str = "ci_97_5") -> str:
    """Point estimate with bracketed interval."""
    return f"{num(est['point'])} [{num(est[lo])}, {num(est[hi])}]"


def _write(name: str, rows: list[str]) -> None:
    (REPORT / name).write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")


def _fixed_b(diag: dict[str, Any], m: str) -> dict[str, float]:
    sens = diag["b_lambda_sensitivity"][m]
    d = {k: sens[f"D_{k[1]}100_b_lambda"] for k in ("DR", "DP", "DS")}
    d["I"] = d["DR"] - d["DP"] - d["DS"]
    return d


def decomposition_table(data: dict[str, dict[str, Any]], rho: int, fixed: bool) -> None:
    """D_P/D_S/I/D_R per encoder and their encoder contrast at one ratio."""
    rows = []
    for ds in DATASETS:
        est = data[ds]["analysis"]["estimates"]
        diag = data[ds]["diagnostics"]
        rows.append(
            f"\\multicolumn{{{7 if fixed else 4}}}{{l}}{{\\emph{{{DATASETS[ds]}}}}} \\\\"
        )
        for key, label in COMPONENTS:
            cells = [
                label,
                ci(est[f"{key}_virchow2_{rho}"]),
                ci(est[f"{key}_uni2h_{rho}"]),
                ci(est[f"delta_{key}_{rho}"]),
            ]
            if fixed:
                v2, uni = _fixed_b(diag, "virchow2")[key], _fixed_b(diag, "uni2h")[key]
                cells += [num(v2), num(uni), num(uni - v2)]
            rows.append(" & ".join(cells) + " \\\\")
        if ds != list(DATASETS)[-1]:
            rows.append("\\addlinespace")
    _write(f"tab_decomposition_{rho}.tex", rows)


def probability_table(data: dict[str, dict[str, Any]], arm: str) -> None:
    """Raw/scaled NLL and ECE at B and one ratio arm, with paired encoder contrasts."""
    rows = []
    for ds in DATASETS:
        pq = data[ds]["analysis"]["probability_quality"]
        pc = data[ds]["report_diagnostics"]["probability_contrasts"]
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{DATASETS[ds]}}}}} \\\\")
        for key, label in PROB_METRICS:
            digits = 2 if key.startswith("nll") else 1
            cells = [label] + [
                num(pq[f"{m}_{a}_{key}"]["point"], digits)
                for a in ("B", arm)
                for m in ENCODERS
            ]
            cells += [ci(pc[f"delta_B_{key}"]), ci(pc[f"delta_damage_{arm}_{key}"])]
            rows.append(" & ".join(cells) + " \\\\")
        if ds != list(DATASETS)[-1]:
            rows.append("\\addlinespace")
    _write(f"tab_probability_{arm}.tex", rows)


def thirds_table(data: dict[str, dict[str, Any]]) -> None:
    """Head/body/tail recall at B and R100 per encoder."""
    rows = []
    for ds in DATASETS:
        rr = data[ds]["diagnostics"]["rank_recall"]
        for group in ("head", "body", "tail"):
            cells = [DATASETS[ds] if group == "head" else "", group]
            for m in ENCODERS:
                b, r = rr[m]["B"][group], rr[m]["R100"][group]
                cells += [num(b, 1), num(r, 1), num(r - b, 1)]
            rows.append(" & ".join(cells) + " \\\\")
        if ds != list(DATASETS)[-1]:
            rows.append("\\addlinespace")
    _write("tab_thirds.tex", rows)


def _split_ba(per_fit: dict[str, Any], m: str, arm: str, split: int) -> np.ndarray:
    return np.array([f["ba"] for f in per_fit[m][arm] if f["split"] == split])


def split_table(data: dict[str, dict[str, Any]]) -> None:
    """Per-split observed D_R(100) per encoder and contrast, with draw SD."""
    rows = []
    for ds in DATASETS:
        per_fit = data[ds]["report_diagnostics"]["per_fit_ba"]
        for s in range(3):
            d = {
                m: _split_ba(per_fit, m, "B", s) - _split_ba(per_fit, m, "R100", s)
                for m in ENCODERS
            }
            b = {m: _split_ba(per_fit, m, "B", s).mean() for m in ENCODERS}
            cells = [DATASETS[ds] if s == 0 else "", str(s)]
            cells += [num(b["virchow2"], 1), num(b["uni2h"], 1)]
            cells += [f"{num(d[m].mean())} ({num(d[m].std(ddof=1))})" for m in ENCODERS]
            cells.append(num(d["uni2h"].mean() - d["virchow2"].mean()))
            rows.append(" & ".join(cells) + " \\\\")
        if ds != list(DATASETS)[-1]:
            rows.append("\\addlinespace")
    _write("tab_splits.tex", rows)


def _boundary(counts: dict[str, int]) -> int:
    return sum(v for k, v in counts.items() if float(k) in (1e-8, 100.0))


def validation_table(data: dict[str, dict[str, Any]]) -> None:
    """Validation-only probe diagnostics and selection summaries per encoder."""
    rows = []
    for ds in DATASETS:
        val = data[ds]["report_diagnostics"]["validation"]
        lt = data[ds]["diagnostics"]["lambda_temperature"]
        for m in ENCODERS:
            v = val[m]
            cells = [DATASETS[ds] if m == "virchow2" else "", ENCODERS[m]]
            cells += [str(v["dimension"]), num(v["feature_norm"]["mean"], 1)]
            for arm in ("B", "R100"):
                a = v["arms"][arm]
                cells += [
                    num(a["validation_ba_mean"], 1),
                    num(a["margin_raw_mean"]),
                    num(a["margin_standardized_mean"]),
                    num(lt[m][arm]["mean_temperature"]),
                ]
            cells.append(
                str(sum(_boundary(x["lambda_counts"]) for x in lt[m].values()))
            )
            rows.append(" & ".join(cells) + " \\\\")
        if ds != list(DATASETS)[-1]:
            rows.append("\\addlinespace")
    _write("tab_validation.tex", rows)


def main() -> None:
    """Optionally sync outputs into report/, then regenerate every report asset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync", action="store_true")
    if parser.parse_args().sync:
        sync()
    data = load()
    decomposition_table(data, 100, fixed=True)
    decomposition_table(data, 10, fixed=False)
    probability_table(data, "R100")
    probability_table(data, "R10")
    thirds_table(data)
    split_table(data)
    validation_table(data)


if __name__ == "__main__":
    main()
