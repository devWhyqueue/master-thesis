"""Report stage: snapshot this dataset's precheck, pilot diagnostics, and main analysis into
one JSON the hand-written protocol/results ``.tex`` cite numbers from (PLAN.md "generated
results"). Every other experiment in this codebase writes its report by hand from
``analysis.json``/``diagnostics.json`` directly; this stage only saves a second look-up from
walking three separate files to one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from imbalance_benchmark.common import REPO_ROOT, output_root, write_json

__all__ = ["run_report"]

_REPORT_DIR = (
    REPO_ROOT
    / "experiments"
    / "03_class_imbalance"
    / "01_cause"
    / "cross"
    / "34_separation_intervention"
    / "report"
)


def _read_if_present(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def run_report(config: dict[str, Any]) -> Path:
    """Collect this dataset's precheck, diagnostics, and analysis JSON into one snapshot."""
    data_dir = output_root(config) / "data"
    dataset = config["dataset"]["name"]
    payload = {
        "dataset": dataset,
        "precheck": _read_if_present(data_dir / "precheck.json"),
        "pilot_diagnostics": _read_if_present(data_dir / "diagnostics.json"),
        "main_analysis": _read_if_present(data_dir / "analysis.json"),
    }
    out_p = _REPORT_DIR / f"generated_results_{dataset}.json"
    write_json(out_p, payload)
    return out_p
