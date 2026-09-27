"""Per-shard, validation-only param selection for one r100 mitigation method.

The winning param is read from ``splits.validation.endpoints.patient_macro_balanced_accuracy``,
written into the run record at fit time (``mitigation.train.eval.evaluate_and_record``); the
test split is never opened here, so selection cannot leak into the reported test-set recovery.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from imbalance_benchmark.common import read_run_record

from analyze.paths import Shard, run_dir, shard_keys

__all__ = ["candidate_params", "select", "selected_frequency"]


def candidate_params(
    config: dict[str, Any], arm: str, stage: str, method: str
) -> list[float | None]:
    """The configured param sweep for one arm/stage/method."""
    raw = config["mitigation"]["grid"][arm][stage][method]
    return [None if v is None else float(v) for v in raw]


def _validation_ba(result_dir: Path) -> float:
    rec = read_run_record(result_dir, splits=("validation",), array_fields=())
    if rec is None:
        raise RuntimeError(f"Missing run record at {result_dir}")
    return float(
        rec["splits"]["validation"]["endpoints"]["patient_macro_balanced_accuracy"]
    )


def select(
    config: dict[str, Any],
    paths: dict[int, dict[str, Path]],
    arm: str,
    stage: str,
    method: str,
) -> dict[Shard, float | None]:
    """Per shard, the configured param with the best validation patient-macro BA."""
    params = candidate_params(config, arm, stage, method)
    return {
        (s, d): max(
            params, key=lambda p: _validation_ba(run_dir(paths[s], arm, method, p, d))
        )
        for s, d in shard_keys()
    }


def selected_frequency(selection: dict[Shard, float | None]) -> dict[str, int]:
    """How often each configured param won, keyed by its string label (``"null"`` if paramless)."""
    freq = Counter("null" if p is None else f"{p:g}" for p in selection.values())
    return dict(freq)
