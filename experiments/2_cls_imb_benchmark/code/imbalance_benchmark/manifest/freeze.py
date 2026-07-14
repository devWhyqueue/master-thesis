from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from imbalance_benchmark.common import (
    compute_data_hash,
    compute_sha256,
    verify_signed_file,
)
from imbalance_benchmark.construction import build_manifest_hash
from imbalance_benchmark.manifest.statistics import (
    achieved_rho,
    normalized_entropy,
    support_statistics,
)

__all__ = [
    "normalized_entropy",
    "achieved_rho",
    "support_statistics",
    "contribution_stats",
    "build_tail_assignments",
    "lock_manifest_freeze",
    "verify_manifest_freeze",
]


def _class_contribution_stats(
    rows: pd.DataFrame, pool_cls: pd.DataFrame, is_mil: bool
) -> dict[str, Any]:
    """Compute one class's unit counts, largest contributions, and pool coverage."""
    n_patients, n_slides = rows["case_id"].nunique(), rows["slide_id"].nunique()
    n_units = n_slides if is_mil else len(rows)
    slide_rows = rows.drop_duplicates("slide_id") if is_mil else rows
    patient_share = slide_rows["case_id"].value_counts().iloc[0] / max(1, n_units)
    slide_share = slide_rows["slide_id"].value_counts().iloc[0] / max(1, n_units)
    pool_denominator = (
        pool_cls["slide_id"].nunique() if is_mil else pool_cls["case_id"].nunique()
    )
    pool_numerator = n_slides if is_mil else n_patients
    return {
        "n_patients": int(n_patients),
        "n_slides": int(n_slides),
        "n_patches": int(len(rows)),
        "max_patient_contribution": float(patient_share),
        "max_slide_contribution": float(slide_share),
        "pool_fraction_retained": float(pool_numerator / max(1, pool_denominator)),
    }


def contribution_stats(
    condition_df: pd.DataFrame, eligible_pool: pd.DataFrame, is_mil: bool
) -> dict[str, dict[str, Any]]:
    """Report per-class unit counts, largest contributions, and pool coverage."""
    return {
        str(cls): _class_contribution_stats(
            rows,
            cast(pd.DataFrame, eligible_pool[eligible_pool["cancer_type"] == cls]),
            is_mil,
        )
        for cls, rows in condition_df.groupby("cancer_type")
    }


def _distinct_random_order(
    native_order: list[str], taken: list[list[str]], seed: int
) -> list[str]:
    """Draw one random permutation distinct from the already-locked assignments.

    A random permutation may coincide with the native or reversed/rotated
    assignment; class identity would then be confounded with tail status.
    Resample deterministically until the draw is distinct, unless every
    permutation is already taken (too few classes for a third assignment).
    """
    rng = np.random.default_rng(seed)
    feasible = math.factorial(len(native_order)) > len(taken)
    for _ in range(1000):
        candidate = list(rng.permutation(native_order))
        if not feasible or candidate not in taken:
            return candidate
    return list(rng.permutation(native_order))


def build_tail_assignments(
    native_order: list[str], seed: int, ordinal: bool
) -> dict[str, list[str]]:
    """Build the three locked tail assignments: native, reversed/rotated, and random."""
    if len(native_order) == 2:
        return {"native": list(native_order), "reversed": list(reversed(native_order))}
    rotated_or_reversed = (
        list(reversed(native_order)) if ordinal else native_order[1:] + native_order[:1]
    )
    random_order = _distinct_random_order(
        list(native_order), [list(native_order), rotated_or_reversed], seed
    )
    return {
        "native": list(native_order),
        "reversed_or_rotated": rotated_or_reversed,
        "random": random_order,
    }


def lock_manifest_freeze(freeze_meta: dict[str, Any]) -> dict[str, Any]:
    """Attach a stable content lock to frozen design metadata."""
    locked = dict(freeze_meta)
    locked.pop("content_sha256", None)
    return {**locked, "content_sha256": compute_data_hash(locked)}


def _collect_artifacts_to_verify(
    meta: dict[str, Any],
) -> list[tuple[str, str, str | None]]:
    """Gather (path, sha256, label) triples for all content-hashed artifacts."""
    to_verify: list[tuple[str, str, str | None]] = []
    for conds in [
        meta.get("conditions", {}),
        *meta.get("assignment_conditions", {}).values(),
    ]:
        for name, info in conds.items():
            to_verify.append((info["path"], info["sha256"], name))
    if pf := meta.get("bootstrap_preflight"):
        to_verify.append((pf["path"], pf["sha256"], None))
    if pilot := meta.get("pilot_report"):
        verify_signed_file(Path(pilot["path"]))
        to_verify.append((pilot["path"], pilot["sha256"], "pilot report"))
    if manifest := meta.get("prepared_manifest"):
        to_verify.append((manifest["path"], manifest["sha256"], "prepared manifest"))
    return to_verify


def verify_manifest_freeze(meta: dict[str, Any]) -> None:
    """Refuse altered frozen design metadata or condition/preflight artifacts."""
    expected = meta.get("content_sha256")
    actual = compute_data_hash({k: v for k, v in meta.items() if k != "content_sha256"})
    if ("shared_T" in meta or expected) and expected != actual:
        raise RuntimeError("Frozen manifest content no longer matches its lock.")
    if p := meta.get("path"):
        verify_signed_file(Path(p))
    for path_str, sha, name in _collect_artifacts_to_verify(meta):
        p = Path(path_str)
        if not p.exists() or compute_sha256(p) != sha:
            if name == "prepared manifest":
                raise RuntimeError("Prepared manifest altered")
            if name == "pilot report":
                raise RuntimeError("Pilot report altered")
            raise RuntimeError(
                f"Manifest '{name}' altered" if name else "Preflight altered"
            )


def _get_constraints(
    name: str, allocated: dict[str, int], available: list[int], minimum: int
) -> tuple[str | None, str | None]:
    values = list(allocated.values())
    limited = next(
        (
            k
            for (k, c), cap in zip(allocated.items(), available, strict=True)
            if c in (minimum, cap)
        ),
        None,
    )
    if name == "balanced":
        return limited, None
    binding = (
        "independent-support floor"
        if min(values) == minimum
        else "unique-support availability"
        if any(c == cap for c, cap in zip(values, available, strict=True))
        else None
    )
    return limited, binding


def write_condition(
    name: str,
    allocated: dict[str, int],
    rows: list[pd.DataFrame],
    pool: pd.DataFrame,
    is_mil: bool,
    seed: int,
    data_dir: Path,
    stem: str | None,
    pool_hash: str | None,
    available: list[int],
    minimum: int,
) -> dict[str, Any]:
    """Write one controlled manifest and its construction metadata."""
    condition = pd.concat(rows, ignore_index=True)
    path = data_dir / f"manifest_{stem or name}.csv"
    condition.to_csv(path, index=False)
    limited, binding = _get_constraints(name, allocated, available, minimum)
    statistics = support_statistics(condition)
    primary = statistics["slide" if is_mil else "patch"]
    return {
        "path": str(path),
        "sha256": compute_sha256(path),
        "requested_rho": {"balanced": 1.0, "moderate": 10.0, "severe": 100.0}.get(
            name, 1.0
        ),
        "achieved_rho": primary["achieved_rho"],
        "normalized_entropy": primary["normalized_entropy"],
        "allocated_counts": primary["counts"],
        "support_statistics": statistics,
        "manifest_hash": build_manifest_hash(condition),
        "contribution_stats": contribution_stats(condition, pool, is_mil),
        "construction_seed": seed,
        "evidence_pool_hash": pool_hash,
        "limiting_class": limited,
        "binding_independent_support_constraint": binding,
    }
