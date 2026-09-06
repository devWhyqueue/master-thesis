"""The derived exp-3 freeze, exp-2 path resolution, and the per-split build orchestrator."""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

from imbalance_benchmark.common import (
    ensure_dirs,
    sign_file,
    split_paths,
    verify_signed_file,
    write_json,
)
from imbalance_benchmark.manifest.freeze import (
    lock_manifest_freeze,
    verify_manifest_freeze,
)

from diversity.manifests.allocation import build_allocation_levels
from diversity.manifests.constants import ALLOCATIONS, LEVELS

__all__ = [
    "build_derived_freeze",
    "verify_derived_freeze",
    "exp2_base_paths",
    "exp2_split_paths",
    "build_split",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_derived_freeze(
    exp2_freeze: dict[str, Any], level_conditions: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    """Deep-copy exp-2's freeze, replacing ``assignment_conditions``/``tail_assignments``.

    Plan Stage 1, point 7: every other key (``runtime_config``, ``seed_roles``,
    ``exposure_budgets``, ``feature_provenance``, ``difficulty_evidence``,
    ``conditions``, ``natural``, ``pilot_report``, ``prepared_manifest``, ...)
    is carried over unchanged, still pointing at exp-2's own frozen files, so
    :func:`verify_manifest_freeze` keeps validating them unchanged.
    """
    derived = deepcopy(exp2_freeze)
    derived.pop("content_sha256", None)
    native_order = list(exp2_freeze["tail_assignments"]["native"])
    derived["assignment_conditions"] = level_conditions
    derived["tail_assignments"] = {
        level: list(native_order) for level in level_conditions
    }
    return lock_manifest_freeze(derived)


def verify_derived_freeze(meta: dict[str, Any]) -> None:
    """Re-run exp-2's own freeze verification against the derived exp-3 freeze."""
    verify_manifest_freeze(meta)


def exp2_base_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Exp-2's own dataset-root output directories.

    Reuses ``common.ensure_dirs`` via a path shim rather than reimplementing
    it: exp-3's config carries exp-2's root under ``slurm.exp2_outputs``
    (plan "Cluster wiring"), not under ``paths.outputs`` (exp-3's own tree).
    ``ensure_dirs`` is idempotent, so pointing it at exp-2's already-frozen
    tree is harmless.
    """
    return ensure_dirs({"paths": {"outputs": config["slurm"]["exp2_outputs"]}})


def exp2_split_paths(config: dict[str, Any], split_index: int) -> dict[str, Path]:
    """Exp-2's own per-split output directories (data/results/tables/...)."""
    return split_paths(exp2_base_paths(config), split_index)


def _load_verified_exp2_freeze(exp2_paths: dict[str, Path]) -> dict[str, Any]:
    freeze_path = exp2_paths["data"] / "manifest_freeze.json"
    verify_signed_file(freeze_path)
    freeze = _read_json(freeze_path)
    verify_manifest_freeze(freeze)
    return freeze


def _level_conditions(per_allocation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        level: {
            allocation: per_allocation[allocation]["conditions"][level]
            for allocation in ALLOCATIONS
        }
        for level in LEVELS
    }


def _write_derived_freeze(
    exp2_freeze: dict[str, Any], level_conditions: dict[str, Any], exp3_data_dir: Path
) -> dict[str, Any]:
    derived = build_derived_freeze(exp2_freeze, level_conditions)
    freeze_path = exp3_data_dir / "manifest_freeze.json"
    write_json(freeze_path, derived)
    sign_file(freeze_path)
    verify_derived_freeze(_read_json(freeze_path))
    return derived


def build_split(config: dict[str, Any], split_index: int) -> dict[str, Any]:
    """Build all six (allocation x level) manifests for one split and its derived freeze.

    Reads exp-2's signed ``manifest_freeze.json`` and frozen condition
    manifests for one split, writes exp-3's own signed derived freeze and its
    six manifests, and copies exp-2's ``manifest.csv`` locally (plan Stage 1,
    point 8) so later stages can resolve val/test identity without reaching
    back into exp-2's directory tree.
    """
    exp2_paths = exp2_split_paths(config, split_index)
    exp2_freeze = _load_verified_exp2_freeze(exp2_paths)
    class_names = list(exp2_freeze["class_names"])
    exp3_paths = split_paths(ensure_dirs(config), split_index)
    per_allocation = {
        allocation: build_allocation_levels(
            allocation, exp2_paths["data"], exp3_paths["data"], exp2_freeze, class_names
        )
        for allocation in ALLOCATIONS
    }
    derived = _write_derived_freeze(
        exp2_freeze, _level_conditions(per_allocation), exp3_paths["data"]
    )
    shutil.copyfile(
        exp2_paths["data"] / "manifest.csv", exp3_paths["data"] / "manifest.csv"
    )
    return {"freeze": derived, "allocations": per_allocation}
