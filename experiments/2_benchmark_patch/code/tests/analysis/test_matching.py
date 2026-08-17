from __future__ import annotations

from pathlib import Path

import pytest

from imbalance_benchmark.analysis.inference.confirmatory.holm import PRIMARY_METHODS
from imbalance_benchmark.analysis.predictors.signals.matching import (
    build_matching_record,
    unit_key,
)
from imbalance_benchmark.common import sign_file, write_json


def _write_split(
    root: Path, index: int, group: str, comparisons: list[dict[str, object]]
) -> None:
    freeze = {
        "dataset_provenance": {"name": group.split(":")[0], "target": group.split(":")[1]},
        "content_sha256": "deadbeef",
    }
    split_dir = root / f"split={index}" / "data"
    write_json(split_dir / "manifest_freeze.json", freeze)
    profile_path = split_dir / "signal_profile.json"
    write_json(
        profile_path,
        {"comparisons": comparisons, "freeze_content_sha256": "deadbeef"},
    )
    sign_file(profile_path)


def _write_root(root: Path, group: str, comparisons: list[dict[str, object]]) -> None:
    for index in range(3):
        _write_split(root, index, group, comparisons)


def test_dominant_shortage_is_the_largest_standardized_score(tmp_path: Path) -> None:
    """Two units pulling opposite ways on nominal vs. independent shortage."""
    unit_a = {
        "assignment": "native",
        "severity": "severe",
        "rho": 10.0,
        "nominal_shortage": 2.0,
        "independent_shortage": 0.0,
        "diversity_shortage": 0.0,
        "support_difficulty_alignment": 0.0,
    }
    unit_b = {
        "assignment": "native",
        "severity": "moderate",
        "rho": 5.0,
        "nominal_shortage": 0.0,
        "independent_shortage": 2.0,
        "diversity_shortage": 0.0,
        "support_difficulty_alignment": 0.0,
    }
    root = tmp_path / "ds"
    _write_root(root, "ds:target", [unit_a, unit_b])

    record = build_matching_record([root])

    a = record["units"][unit_key("ds:target", "native", "severe")]
    assert a["dominant"] == "nominal"
    assert a["ambiguous"] is False
    assert a["matched_methods"] == ["class_balanced_ce", "weighted_ce"]
    assert set(a["unmatched_methods"]) == PRIMARY_METHODS - {
        "weighted_ce",
        "class_balanced_ce",
    }

    b = record["units"][unit_key("ds:target", "native", "moderate")]
    assert b["dominant"] == "independent"
    assert b["matched_methods"] == ["independent_support_ce"]


def test_ambiguous_when_top_two_standardized_scores_tie(tmp_path: Path) -> None:
    """Nominal and independent shortage standardize to an exact tie -> ambiguous."""
    unit_x = {
        "assignment": "native",
        "severity": "severe",
        "rho": 10.0,
        "nominal_shortage": 1.0,
        "independent_shortage": 1.0,
        "diversity_shortage": -5.0,
        "support_difficulty_alignment": 0.0,
    }
    unit_y = {
        "assignment": "native",
        "severity": "moderate",
        "rho": 5.0,
        "nominal_shortage": -1.0,
        "independent_shortage": -1.0,
        "diversity_shortage": 5.0,
        "support_difficulty_alignment": 0.0,
    }
    root = tmp_path / "ds"
    _write_root(root, "ds:target", [unit_x, unit_y])

    record = build_matching_record([root])

    x = record["units"][unit_key("ds:target", "native", "severe")]
    assert x["ambiguous"] is True
    assert x["dominant"] is None
    assert x["matched_methods"] == []
    assert set(x["unmatched_methods"]) == PRIMARY_METHODS


def test_no_deprived_class_zero_scores_standardize_without_crashing(tmp_path: Path) -> None:
    """An all-zero unit pooled against variation elsewhere must not divide by zero."""
    zero_unit = {
        "assignment": "native",
        "severity": "severe",
        "rho": 1.0,
        "nominal_shortage": 0.0,
        "independent_shortage": 0.0,
        "diversity_shortage": 0.0,
        "support_difficulty_alignment": 0.0,
    }
    varied_unit = {
        "assignment": "native",
        "severity": "moderate",
        "rho": 8.0,
        "nominal_shortage": 3.0,
        "independent_shortage": 1.0,
        "diversity_shortage": 1.0,
        "support_difficulty_alignment": 0.5,
    }
    root = tmp_path / "ds"
    _write_root(root, "ds:target", [zero_unit, varied_unit])

    record = build_matching_record([root])

    zero = record["units"][unit_key("ds:target", "native", "severe")]
    assert all(isinstance(v, float) for v in zero["standardized_scores"].values())
    assert zero["dominant"] in {None, "nominal", "independent", "difficulty", "diversity"}


def test_single_unit_pool_has_zero_variance_and_is_ambiguous(tmp_path: Path) -> None:
    """A single pooled unit standardizes to all zeros (std undefined -> zero), hence ambiguous."""
    only_unit = {
        "assignment": "native",
        "severity": "severe",
        "rho": 4.0,
        "nominal_shortage": 1.5,
        "independent_shortage": 0.2,
        "diversity_shortage": 0.1,
        "support_difficulty_alignment": -0.3,
    }
    root = tmp_path / "ds"
    _write_root(root, "ds:target", [only_unit])

    record = build_matching_record([root])

    unit = record["units"][unit_key("ds:target", "native", "severe")]
    assert unit["standardized_scores"] == {
        "nominal": pytest.approx(0.0),
        "independent": pytest.approx(0.0),
        "difficulty": pytest.approx(0.0),
        "diversity": pytest.approx(0.0),
    }
    assert unit["ambiguous"] is True
