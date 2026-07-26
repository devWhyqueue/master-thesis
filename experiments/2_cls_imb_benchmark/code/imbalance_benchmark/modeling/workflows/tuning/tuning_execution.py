from __future__ import annotations

from pathlib import Path
from typing import Any

from imbalance_benchmark.modeling.context import CONDITIONS, roster_for_regime
from imbalance_benchmark.modeling.workflows.tuning.tuning_reduction import (
    write_base_selections,
    write_final_selections,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_schedule import phase_methods


def reduce_tuning_shards(
    base: dict[str, Path],
    raw_scopes: list[tuple[dict[str, Path], Any, Any]],
    freeze: dict[str, Any],
    fingerprint: list[str],
    phase: str,
) -> None:
    """Reduce complete base or dependent shards into signed selections."""
    is_mil = raw_scopes[0][1].is_mil
    base_methods = phase_methods(is_mil, "base")
    if phase == "base":
        write_base_selections(
            base,
            freeze,
            fingerprint,
            base_methods,
            roster_for_regime(is_mil),
            CONDITIONS,
        )
        return
    write_final_selections(
        base,
        freeze,
        fingerprint,
        base_methods,
        phase_methods(is_mil, "dependent"),
        CONDITIONS,
    )
