from __future__ import annotations

import math
from pathlib import Path

import pytest

from imbalance_benchmark.analysis.inference.confirmatory.holm import PRIMARY_METHODS
from imbalance_benchmark.analysis.predictors.signals.matching import (
    build_matching_record,
    unit_key,
)
from imbalance_benchmark.common import sign_file, write_json


def _write_split(
    root: Path,
    index: int,
    group: str,
    comparisons: list[dict[str, object]],
    class_names: tuple[str, ...] = ("A", "B", "C"),
) -> None:
    freeze = {
        "dataset_provenance": {"name": group.split(":")[0], "target": group.split(":")[1]},
        "content_sha256": "deadbeef",
        "class_names": list(class_names),
    }
    split_dir = root / f"split={index}" / "data"
    write_json(split_dir / "manifest_freeze.json", freeze)
    profile_path = split_dir / "signal_profile.json"
    write_json(
        profile_path,
        {"comparisons": comparisons, "freeze_content_sha256": "deadbeef"},
    )
    sign_file(profile_path)


def _write_root(
    root: Path,
    group: str,
    comparisons: list[dict[str, object]],
    class_names: tuple[str, ...] = ("A", "B", "C"),
) -> None:
    for index in range(3):
        _write_split(root, index, group, comparisons, class_names)


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
    # A shortage that was never created (raw score <= 0) can never be dominant.
    assert zero["dominant"] is None


def test_single_unit_pool_has_zero_variance_and_every_axis_is_degenerate(
    tmp_path: Path,
) -> None:
    """A single pooled unit has undefined variance on every axis: all NaN, none dominant."""
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
    assert all(math.isnan(v) for v in unit["standardized_scores"].values())
    assert set(unit["degenerate_axes"]) == {
        "nominal",
        "independent",
        "difficulty",
        "diversity",
    }
    assert unit["dominant"] is None
    assert unit["ambiguous"] is False


def test_structurally_zero_independent_shortage_is_never_dominant(
    tmp_path: Path,
) -> None:
    """Regression for the published mislabelling: a constant-zero axis must not win the argmax.

    Independent-support shortage is 0.0 for every unit in the pool (structurally
    zero, per the report), while the other three scores standardize negative for
    the unit under test. The old unguarded argmax picked "independent" here
    because a constant column standardized to exactly 0.0, beating the genuinely
    negative axes.
    """
    unit_p = {
        "assignment": "native",
        "severity": "severe",
        "rho": 4.0,
        "nominal_shortage": 0.0,
        "independent_shortage": 0.0,
        "diversity_shortage": 0.0,
        "support_difficulty_alignment": 2.0,
    }
    unit_q = {
        "assignment": "native",
        "severity": "moderate",
        "rho": 8.0,
        "nominal_shortage": 3.0,
        "independent_shortage": 0.0,
        "diversity_shortage": 3.0,
        "support_difficulty_alignment": -2.0,
    }
    root = tmp_path / "ds"
    _write_root(root, "ds:target", [unit_p, unit_q])

    record = build_matching_record([root])

    p = record["units"][unit_key("ds:target", "native", "severe")]
    assert "independent" in p["degenerate_axes"]
    assert p["dominant"] is None


def test_standardization_is_computed_within_a_dataset_root_not_pooled(
    tmp_path: Path,
) -> None:
    """A root's own scale must not be diluted by another root's very different scale.

    Regression for the published bug: TCGA-UT's nominal shortage read -1.00
    because it was small relative to BRACS's, not because TCGA-UT has none.
    """

    def unit(assignment: str, severity: str, nominal: float) -> dict[str, object]:
        return {
            "assignment": assignment,
            "severity": severity,
            "rho": 1.0,
            "nominal_shortage": nominal,
            "independent_shortage": 0.0,
            "diversity_shortage": 0.0,
            "support_difficulty_alignment": 0.0,
        }

    root_a = tmp_path / "a"
    _write_root(
        root_a, "a:target", [unit("native", "moderate", 1.0), unit("native", "severe", 3.0)]
    )
    root_b = tmp_path / "b"
    _write_root(
        root_b,
        "b:target",
        [unit("native", "moderate", 100.0), unit("native", "severe", 300.0)],
    )

    record = build_matching_record([root_a, root_b])

    a_low = record["units"][unit_key("a:target", "native", "moderate")]
    b_low = record["units"][unit_key("b:target", "native", "moderate")]
    # Pooling both roots' raw values ([1, 3, 100, 300]) before standardizing
    # would give the low unit in root A a z-score around -0.82, not -1.0.
    assert a_low["standardized_scores"]["nominal"] == pytest.approx(-1.0)
    assert b_low["standardized_scores"]["nominal"] == pytest.approx(-1.0)


def test_binary_target_alignment_axis_is_degenerate_in_matching(
    tmp_path: Path,
) -> None:
    """Pearson correlation over exactly two points is always +/-1; a two-class
    dataset's alignment score must never be eligible for the dominant argmax."""
    unit_severe = {
        "assignment": "native",
        "severity": "severe",
        "rho": 4.0,
        "nominal_shortage": 0.0,
        "independent_shortage": 0.0,
        "diversity_shortage": 0.0,
        "support_difficulty_alignment": -1.0,
    }
    unit_moderate = {
        "assignment": "native",
        "severity": "moderate",
        "rho": 8.0,
        "nominal_shortage": 0.0,
        "independent_shortage": 0.0,
        "diversity_shortage": 0.0,
        "support_difficulty_alignment": 1.0,
    }
    root = tmp_path / "ds"
    _write_root(root, "ds:target", [unit_severe, unit_moderate], class_names=("A", "B"))

    record = build_matching_record([root])

    severe = record["units"][unit_key("ds:target", "native", "severe")]
    assert "difficulty" in severe["degenerate_axes"]
    assert severe["dominant"] is None


def test_multiclass_alignment_axis_can_still_be_dominant(tmp_path: Path) -> None:
    """The same setup on a three-class dataset must remain eligible."""
    unit_severe = {
        "assignment": "native",
        "severity": "severe",
        "rho": 4.0,
        "nominal_shortage": 0.0,
        "independent_shortage": 0.0,
        "diversity_shortage": 0.0,
        "support_difficulty_alignment": -1.0,
    }
    unit_moderate = {
        "assignment": "native",
        "severity": "moderate",
        "rho": 8.0,
        "nominal_shortage": 0.0,
        "independent_shortage": 0.0,
        "diversity_shortage": 0.0,
        "support_difficulty_alignment": 1.0,
    }
    root = tmp_path / "ds"
    _write_root(root, "ds:target", [unit_severe, unit_moderate])  # default 3 classes

    record = build_matching_record([root])

    severe = record["units"][unit_key("ds:target", "native", "severe")]
    assert "difficulty" not in severe["degenerate_axes"]
    assert severe["dominant"] == "difficulty"
