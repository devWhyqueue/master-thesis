from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from imbalance_benchmark.common import (
    compute_data_hash,
    compute_sha256,
    split_paths,
    verify_signed_file,
)
from imbalance_benchmark.manifest.construction_helpers import condition_metadata
from imbalance_benchmark.manifest.statistics import (
    achieved_rho,
    normalized_entropy,
    support_statistics,
)
from imbalance_benchmark.datasets.features.provenance_lock import (
    verify_frozen_feature_provenance,
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


def build_tail_assignments(
    native_order: list[str], difficulty: dict[str, float]
) -> dict[str, list[str]]:
    """Build native and difficulty-directed assignments; native order breaks ties."""
    if set(native_order) != set(difficulty):
        raise ValueError("Difficulty evidence must cover every locked class")
    easiest_to_hardest = sorted(
        native_order, key=lambda name: (difficulty[name], native_order.index(name))
    )
    if len(native_order) == 2:
        return {
            "native": list(native_order),
            "difficulty_reversed": list(reversed(easiest_to_hardest)),
        }
    return {
        "native": list(native_order),
        "difficulty_aligned": easiest_to_hardest,
        "difficulty_reversed": list(reversed(easiest_to_hardest)),
    }


def _pilot_difficulty_reports(base_paths: dict[str, Path]) -> list[dict[str, Any]]:
    """Load signed difficulty evidence from all required splits."""
    reports = []
    for index in range(3):
        path = split_paths(base_paths, index)["data"] / "pilot_report.json"
        if not path.exists():
            raise RuntimeError(
                "All three signed pilot reports are required before freeze"
            )
        verify_signed_file(path)
        evidence = json.loads(path.read_text()).get("difficulty_evidence")
        if not isinstance(evidence, dict) or not isinstance(
            evidence.get("difficulty"), dict
        ):
            raise RuntimeError(
                "Pilot report lacks signed difficulty evidence; re-run pilot"
            )
        reports.append(evidence)
    return reports


def _duplicate_assignment(
    candidates: list[dict[str, list[str]]], key: str, selected: set[str]
) -> int | None:
    """Return first split where a candidate repeats an earlier assignment."""
    return next(
        (
            index
            for index, candidate in enumerate(candidates)
            if key in candidate
            and candidate[key] in [candidate[name] for name in selected]
        ),
        None,
    )


def locked_difficulty_assignments(
    base_paths: dict[str, Path], split_index: int, classes: list[str]
) -> tuple[dict[str, list[str]], dict[str, str], dict[str, Any]]:
    """Keep assignment keys identical across splits and record duplicate omissions."""
    reports = _pilot_difficulty_reports(base_paths)
    candidates = [
        build_tail_assignments(classes, report["difficulty"]) for report in reports
    ]
    selected, omissions = {"native"}, {}
    for key in ("difficulty_aligned", "difficulty_reversed"):
        duplicate = _duplicate_assignment(candidates, key, selected)
        if key in candidates[0] and duplicate is None:
            selected.add(key)
        elif duplicate is not None:
            omissions[key] = f"duplicates an earlier assignment in split {duplicate}"
    current = candidates[split_index]
    return (
        {key: current[key] for key in current if key in selected},
        omissions,
        reports[split_index],
    )


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
    verify_frozen_feature_provenance(meta)


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


def write_condition(spec: dict[str, Any]) -> dict[str, Any]:
    """Write one controlled manifest and its construction metadata."""
    condition = pd.concat(spec["rows"], ignore_index=True)
    path = spec["data_dir"] / f"manifest_{spec['stem'] or spec['name']}.csv"
    condition.to_csv(path, index=False)
    limited, binding = _get_constraints(
        spec["name"], spec["allocated"], spec["available"], spec["minimum"]
    )
    statistics = support_statistics(condition)
    primary = statistics["slide" if spec["is_mil"] else "patch"]
    return condition_metadata(
        path,
        condition,
        statistics,
        primary,
        contribution_stats(condition, spec["pool"], spec["is_mil"]),
        (limited, binding),
        spec,
    )
