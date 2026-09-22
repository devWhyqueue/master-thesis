from __future__ import annotations

import json
from typing import Any

from imbalance_benchmark.analysis.metrics import assign_tiers
from imbalance_benchmark.modeling.context import get_grid_configs
from imbalance_benchmark.modeling.training.config import resolve_training_config
from imbalance_benchmark.modeling.workflows.run_context import RunContext

__all__ = [
    "_load_record_freeze",
    "_allocated_condition",
    "_condition_tiers",
    "_provenance_payload",
]


def _load_record_freeze(run: RunContext) -> dict[str, Any]:
    """Load the split's verified freeze once for tiers and run-record provenance."""
    fp = run.paths["data"] / "manifest_freeze.json"
    return json.loads(fp.read_text()) if fp.exists() else {}


def _allocated_condition(
    freeze: dict[str, Any], assignment: str, cond: str
) -> dict[str, Any]:
    """The frozen allocation for one condition (native for balanced, else per assignment)."""
    native = freeze.get("conditions", {}).get(cond)
    if native is not None:
        return native
    return freeze.get("assignment_conditions", {}).get(assignment, {}).get(cond, {})


def _condition_tiers(
    run: RunContext, cond: str, freeze: dict[str, Any]
) -> dict[str, str] | None:
    """Head/body/tail tiers for one condition from the frozen allocation and assignment."""
    if cond == "balanced":
        return None
    alloc = _allocated_condition(freeze, run.assignment, cond).get("allocated_counts")
    if not alloc or not run.class_names:
        return None
    return assign_tiers(
        run.class_names,
        alloc,
        freeze.get("tail_assignments", {}).get(run.assignment, run.class_names),
    )


def _provenance_payload(
    run: RunContext, cond: str, method: str, freeze: dict[str, Any]
) -> dict[str, Any]:
    """Appendix A run provenance the record must carry beyond tuning params and hashes.

    Everything here is drawn from the freeze verified before confirmation, except
    the resolved model/optimizer configuration, whose values are source defaults
    absent from the supplied YAML. ``freeze_content_sha256`` binds the full
    signed condition manifest (dataset, seeds, preflight, grids, pilot).
    """
    allocated = _allocated_condition(freeze, run.assignment, cond)
    return {
        "model_optimizer_config": resolve_training_config(run.config, run.is_mil),
        "candidate_grid": freeze.get("method_grids", {}).get(method)
        or get_grid_configs(method, run.n_classes),
        "freeze_content_sha256": freeze.get("content_sha256"),
        "dataset_version": freeze.get("dataset_provenance", {}).get("version"),
        "achieved_T": freeze.get("shared_T"),
        "achieved_rho": allocated.get("achieved_rho"),
        "pilot_min_support": freeze.get("min_support"),
    }
