from __future__ import annotations

from pathlib import Path

from imbalance_benchmark.analysis.reporting.ingestion import method_diagnostics_summary
from imbalance_benchmark.common import write_run_record


def _write_run(
    results_root: Path,
    condition: str,
    method: str,
    seed: int,
    method_diagnostics: dict,
) -> None:
    write_run_record(
        results_root / condition / method / f"seed={seed}",
        {
            "condition": condition,
            "method": method,
            "method_diagnostics": method_diagnostics,
            "splits": {},
        },
    )


def test_method_diagnostics_summary_sums_numeric_counters_across_seeds(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    _write_run(results, "severe", "semantic_scale_ce", 0, {"ssb_invalid_draws": 3})
    _write_run(results, "severe", "semantic_scale_ce", 1, {"ssb_invalid_draws": 5})

    rows = method_diagnostics_summary({"results": results})

    assert rows == [
        {
            "condition": "severe",
            "method": "semantic_scale_ce",
            "seeds": 2,
            "ssb_invalid_draws": 8,
        }
    ]


def test_method_diagnostics_summary_counts_list_valued_diagnostics(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    _write_run(
        results,
        "moderate",
        "oko",
        0,
        {"sc_mil_batch_diagnostics": [{"n_pairs": 0}, {"n_pairs": 2}]},
    )

    rows = method_diagnostics_summary({"results": results})

    assert rows == [
        {
            "condition": "moderate",
            "method": "oko",
            "seeds": 1,
            "sc_mil_batch_diagnostics_count": 2,
        }
    ]


def test_method_diagnostics_summary_skips_runs_without_diagnostics(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    _write_run(results, "balanced", "ce", 0, {})

    assert method_diagnostics_summary({"results": results}) == []
