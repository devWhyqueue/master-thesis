import json
import re
from pathlib import Path
from typing import cast

import pandas as pd

from analysis.plotting import (
    PATCH_ORDER,
    WSI_ORDER,
    _benchmark,
    _mean_std,
    _method_label,
    _write_unavailable,
    _write_wide_table,
)

METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
NativeResults = dict[str, dict[str, dict[str, tuple[float, float]]]]

NATIVE_TABLES = Path(__file__).parents[5] / "experiments/class_imbalance/outputs/tables"
# Maps companion (native-benchmark) display labels to this report's display names.
NATIVE_LABEL_MAP = {
    "CFAL": "CFAL",
    "OKO": "OKO",
    "CE + soft F1 (balanced)": "CE + soft F1",
    "Weighted CE": "Weighted CE",
    "Balanced sampler": "Balanced sampling",
    "CE + soft MCC (balanced)": "CE + soft MCC",
    "ProGAN augmentation": "ProGAN augmentation",
    "Focal": "Focal loss",
    "MDE-MIL (ensemble)": "MDE-MIL",
    "RankMix": "RankMix",
    "SC-MIL": "SC-MIL",
    "Weighted MIL": "Weighted CE",
    "Balanced MIL": "Balanced sampling",
    "Focal MIL": "Focal loss",
}


# --- Tuning-selected test result loading ---


def result_summary(results_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Return test results for tuning-selected winners, one row per (method, regime, seed)."""
    selection_path = output_dir / "tuning_selection.json"
    if not selection_path.exists():
        return pd.DataFrame()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not selection:
        return pd.DataFrame()
    return pd.DataFrame(_result_rows(results_dir, selection))


def _result_rows(results_dir: Path, selection: list[dict]) -> list[dict]:
    return [
        row for entry in selection for row in _entry_result_rows(entry, results_dir)
    ]


def _entry_result_rows(entry: dict, results_dir: Path) -> list[dict]:
    m = re.match(r"order=(?P<order>.+)/param=(?P<parameter>[\d.]+)", entry["regime"])
    if m is None:
        return []
    order, parameter = m.group("order"), float(m.group("parameter"))
    rows = []
    for seed in range(3):
        test_path = (
            results_dir
            / "tuning"
            / entry["benchmark"]
            / entry["regime"]
            / entry["method"]
            / entry["variant"]
            / f"seed={seed}"
            / "test_results.json"
        )
        row = _load_result_row(test_path, entry, order, parameter, seed)
        if row:
            rows.append(row)
    return rows


def _load_result_row(
    test_path: Path, entry: dict, order: str, parameter: float, seed: int
) -> dict | None:
    if not test_path.exists():
        return None
    with open(test_path, encoding="utf-8") as f:
        payload = json.load(f)
    return {
        "method": entry["method"],
        "benchmark": _benchmark(entry["method"]),
        "order": order,
        "parameter": parameter,
        "seed": seed,
        **{metric: float(payload[metric]) for metric in METRICS},
    }


# --- Native-benchmark parsing ---


def native_results() -> NativeResults:
    """Parse companion native-benchmark BAcc/F1 per method, keyed by report display name."""
    return {
        "patch": _parse_native(NATIVE_TABLES / "result_summary_patch.tex"),
        "wsi_bag": _parse_native(NATIVE_TABLES / "result_summary_wsi_bag.tex"),
    }


def _parse_native(path: Path) -> dict[str, dict[str, tuple[float, float]]]:
    """Extract (mean, std) for balanced accuracy and macro F1 from a native summary table.

    Columns are Method & Accuracy & Balanced accuracy & Macro F1, each cell two \\num{} values.
    """
    if not path.exists():
        return {}
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "\\num{" not in line or "&" not in line:
            continue
        display = NATIVE_LABEL_MAP.get(line.split("&", 1)[0].strip())
        nums = [float(x) for x in re.findall(r"\\num\{([\d.]+)\}", line)]
        if display is None or len(nums) < 6:
            continue
        out[display] = {
            "balanced_accuracy": (nums[2], nums[3]),
            "macro_f1": (nums[4], nums[5]),
        }
    return out


# --- Result table writers ---


def write_result_tables(
    frame: pd.DataFrame, tables_dir: Path, native: NativeResults | None = None
) -> None:
    """Write patch and WSI result summary tables."""
    native = native or {}
    for benchmark, filename in [
        ("patch", "result_summary_patch.tex"),
        ("wsi_bag", "result_summary_wsi_bag.tex"),
    ]:
        part = (
            cast(pd.DataFrame, frame[frame["benchmark"] == benchmark])
            if not frame.empty
            else frame
        )
        write_result_table(part, tables_dir / filename, native.get(benchmark, {}))


def write_result_table(
    frame: pd.DataFrame,
    path: Path,
    native: dict[str, dict[str, tuple[float, float]]] | None = None,
) -> None:
    """Write one benchmark result table with lambda as columns, BAcc and F1 per cell."""
    native = native or {}
    params = [0.5, 1.0, 1.5]
    fallback_header = "Method & " + " & ".join(
        f"BAcc ($\\lambda={p:.1f}$) & F1 ($\\lambda={p:.1f}$)" for p in params
    )
    if frame.empty:
        _write_unavailable(path, fallback_header)
        return
    by_method = _by_method_map(frame)
    method_order = PATCH_ORDER if any("patch" in m for m in by_method) else WSI_ORDER
    ordered = [m for m in method_order if m in by_method] + [
        m for m in by_method if m not in method_order
    ]
    colspec, header_lines = _result_colspec_headers(params, bool(native))
    rows = [_result_method_row(m, by_method[m], params, native) for m in ordered]
    _write_wide_table(path, colspec, header_lines, rows)


def _by_method_map(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    metrics = ("balanced_accuracy", "macro_f1")
    agg = (
        frame.groupby(["method", "parameter"])
        .agg(**{f"{m}_{s}": (m, s) for m in metrics for s in ("mean", "std")})
        .reset_index()
    )
    return {
        str(m): cast(pd.DataFrame, agg[agg["method"] == m])
        for m in agg["method"].unique()
    }


def _result_colspec_headers(
    params: list[float], with_native: bool = False
) -> tuple[str, list[str]]:
    blocks = [f"$\\lambda={p:.1f}$" for p in params] + (
        ["Native"] if with_native else []
    )
    # Native is contextual, not part of the constructed experiment: rule it off.
    colspec = "l" + "r" * (len(params) * 2) + ("|rr" if with_native else "")
    header_top = "Method & " + " & ".join(
        f"\\multicolumn{{2}}{{c}}{{{block}}}" for block in blocks
    )
    cmidrule = " ".join(
        f"\\cmidrule(lr){{{2 + i * 2}--{3 + i * 2}}}" for i in range(len(blocks))
    )
    header_sub = " & ".join([""] + ["BAcc & F1"] * len(blocks))
    return colspec, [header_top + f" \\\\ {cmidrule}", header_sub]


def _result_method_row(
    method: str,
    part: pd.DataFrame,
    params: list[float],
    native: dict[str, dict[str, tuple[float, float]]] | None = None,
) -> str:
    cells = [_method_label(method)]
    for p in params:
        p_row = part[part["parameter"] == p]
        if p_row.empty:
            cells += ["--", "--"]
        else:
            r = p_row.iloc[0]
            cells += [_mean_std(r, "balanced_accuracy"), _mean_std(r, "macro_f1")]
    if native:
        cells += _native_cells(native.get(_method_label(method)))
    return " & ".join(cells) + "\\\\"


def _native_cells(stats: dict[str, tuple[float, float]] | None) -> list[str]:
    if stats is None:
        return ["--", "--"]
    return [_native_cell(stats["balanced_accuracy"]), _native_cell(stats["macro_f1"])]


def _native_cell(stat: tuple[float, float]) -> str:
    mean, std = stat
    return f"\\num{{{mean:.3f}}} $\\pm$ \\num{{{std:.3f}}}"
