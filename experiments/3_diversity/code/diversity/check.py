"""Stage 2 (check, Gate 0): headroom, semantic volume, S_nom/S_ind audit, and the gate.

Runs once per dataset after ``build`` (plan Stage 2) and never fits a model.
Reads only the manifests ``build`` wrote plus exp-2's frozen per-split
``signal_profile.json``, and writes
``outputs/<dataset>/patch/tables/manipulation_check.json``.

The run stops here: no GPU fit is submitted until these numbers are
reviewed (plan "Execution order", step 4).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from imbalance_benchmark.analysis.inference.context import _tail_classes
from imbalance_benchmark.analysis.predictors.rq3_features import (
    _fixed_diversity,
    _independent_shortage,
)

# ``_nominal_shortage`` lives in the signal-profile module, not rq3_features;
# reused here (private, intentional -- plan explicitly allows reaching into
# exp-2's private analysis helpers) purely as an audit that construction did
# not accidentally move nominal or independent support.
from imbalance_benchmark.analysis.predictors.signals.signal_profile import (
    _nominal_shortage,
)
from imbalance_benchmark.common import (
    ensure_dirs,
    split_paths,
    verify_signed_file,
    write_json,
)
from imbalance_benchmark.modeling.training.semantic_scale import EPS_S

from diversity.manifests import (
    ALLOCATIONS,
    LEVELS,
    eligible_pool,
    exp2_split_paths,
    headroom_table,
    slot_table,
)

__all__ = ["run_check"]

logger = logging.getLogger(__name__)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _headroom_stats(headroom: pd.DataFrame, tail_classes: set[str]) -> dict[str, Any]:
    def _summary(frame: pd.DataFrame) -> dict[str, float]:
        if frame.empty:
            return {
                "min": float("nan"),
                "median": float("nan"),
                "max": float("nan"),
                "frac_h_le_1_2": float("nan"),
            }
        return {
            "min": float(frame["h"].min()),
            "median": float(frame["h"].median()),
            "max": float(frame["h"].max()),
            "frac_h_le_1_2": float((frame["h"] <= 1.2).mean()),
        }

    tail_rows = cast(
        pd.DataFrame, headroom[headroom["cancer_type"].isin(list(tail_classes))]
    )
    return {"overall": _summary(headroom), "tail_classes": _summary(tail_rows)}


def _headroom_for_allocation(
    exp2_data_dir: Path, exp3_data_dir: Path, allocation: str
) -> pd.DataFrame:
    """Recompute headroom for one split/allocation purely from files on disk.

    ``check`` runs as a separate job from ``build``, so it re-derives the
    slot table and eligible pool from the written manifests rather than
    sharing in-memory state.
    """
    random_manifest = exp3_data_dir / f"manifest_{allocation}_random.csv"
    train_df = cast(
        pd.DataFrame,
        pd.read_csv(exp2_data_dir / "manifest.csv").pipe(
            lambda df: df[df["split"] == "train"]
        ),
    )
    slots = slot_table(cast(pd.DataFrame, pd.read_csv(random_manifest)))
    pool = eligible_pool(train_df, slots)
    return headroom_table(pool, slots)


def _semantic_volume_ratio(
    narrow_manifest: Path, wide_manifest: Path, class_names: list[str], seed: int
) -> float:
    """S_div(a): Eq. (2) of the report, mean over ALL K classes.

    Deliberately not ``rq3_features._diversity_shortage`` -- that helper's
    denominator is the deprived-classes-only set, whereas every class is
    manipulated here, so the mean must run over all K classes (report
    Sec. "Manipulation Check").
    """
    narrow_volumes = _fixed_diversity(narrow_manifest, False, class_names, seed)
    wide_volumes = _fixed_diversity(wide_manifest, False, class_names, seed)
    ratios = [
        np.log(
            max(wide_volumes.get(index, EPS_S), EPS_S)
            / max(narrow_volumes.get(index, EPS_S), EPS_S)
        )
        for index in range(len(class_names))
    ]
    return float(np.mean(ratios))


def _audit_zero(name: str, value: float) -> None:
    if value != 0.0:
        raise RuntimeError(
            f"{name} is {value} instead of exactly 0.0 on a fixed-support cell; "
            "this is a construction bug, not a finding."
        )


def _split_allocation_report(
    exp2_data_dir: Path,
    exp3_data_dir: Path,
    freeze: dict[str, Any],
    allocation: str,
    max_diversity_shortage: float,
    dataset: str,
) -> dict[str, Any]:
    class_names = list(freeze["class_names"])
    seed = int(freeze["construction_seed"])
    narrow_manifest = exp3_data_dir / f"manifest_{allocation}_narrow.csv"
    wide_manifest = exp3_data_dir / f"manifest_{allocation}_wide.csv"
    narrow_meta = freeze["assignment_conditions"]["narrow"][allocation]
    wide_meta = freeze["assignment_conditions"]["wide"][allocation]

    s_nom = _nominal_shortage(
        wide_meta,
        narrow_meta,
        freeze.get("difficulty_evidence", {}).get("difficulty", {}),
    )
    s_ind = _independent_shortage(wide_meta, narrow_meta, False)
    _audit_zero("S_nom", s_nom)
    _audit_zero("S_ind", s_ind)

    s_div = _semantic_volume_ratio(narrow_manifest, wide_manifest, class_names, seed)
    tail_classes = {
        class_names[i] for i in _tail_classes(freeze, class_names, "random", "severe")
    }
    headroom = _headroom_for_allocation(exp2_data_dir, exp3_data_dir, allocation)
    return {
        "allocation": allocation,
        "dataset": dataset,
        "s_nom": s_nom,
        "s_ind": s_ind,
        "s_div": s_div,
        "gate_threshold": max_diversity_shortage,
        "gate_passed": s_div > max_diversity_shortage,
        "headroom": _headroom_stats(headroom, tail_classes),
    }


def _max_diversity_shortage(exp2_data_dir: Path) -> float:
    profile_path = exp2_data_dir / "signal_profile.json"
    verify_signed_file(profile_path)
    profile = _read_json(profile_path)
    shortages = [c["diversity_shortage"] for c in profile["comparisons"]]
    if not shortages:
        raise RuntimeError(f"{profile_path} has no comparisons to gate against")
    return float(max(shortages))


def _log_report(split_index: int, report: dict[str, Any]) -> None:
    logger.info(
        "check: split=%s allocation=%s S_div=%.4f threshold=%.4f gate_passed=%s",
        split_index,
        report["allocation"],
        report["s_div"],
        report["gate_threshold"],
        report["gate_passed"],
    )


def _split_reports(
    config: dict[str, Any], exp3_base: dict[str, Path], split_index: int, dataset: str
) -> list[dict[str, Any]]:
    """Every allocation's Gate 0 report for one split, or [] if 'build' hasn't run yet."""
    exp2_paths = exp2_split_paths(config, split_index)
    exp3_paths = split_paths(exp3_base, split_index)
    freeze_path = exp3_paths["data"] / "manifest_freeze.json"
    if not freeze_path.exists():
        logger.warning(
            "check: split %s has no derived freeze; run build first", split_index
        )
        return []
    verify_signed_file(freeze_path)
    freeze = _read_json(freeze_path)
    max_shortage = _max_diversity_shortage(exp2_paths["data"])
    reports = []
    for allocation in ALLOCATIONS:
        report = _split_allocation_report(
            exp2_paths["data"],
            exp3_paths["data"],
            freeze,
            allocation,
            max_shortage,
            dataset,
        )
        report["split"] = split_index
        reports.append(report)
        _log_report(split_index, report)
    return reports


def run_check(config: dict[str, Any]) -> dict[str, Any]:
    """Run Gate 0 for every split and allocation of one dataset; write the summary table."""
    dataset = config["dataset"]["name"]
    exp3_base = ensure_dirs(config)
    reports = [
        report
        for split_index in range(3)
        for report in _split_reports(config, exp3_base, split_index, dataset)
    ]
    summary = {"dataset": dataset, "levels": list(LEVELS), "reports": reports}
    out_path = exp3_base["tables"] / "manipulation_check.json"
    write_json(out_path, summary)
    logger.info("check: wrote %s", out_path)
    return summary
