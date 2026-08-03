from __future__ import annotations

import math

from imbalance_benchmark.modeling.context import (
    NATURAL_ANCHOR_METHODS,
    roster_for_regime,
)
from imbalance_benchmark.modeling.workflows.confirmation_schedule import (
    CONFIRMATION_SEED_COUNT,
    confirm_array_size,
    confirm_group_methods,
    confirm_units_for_group,
    resolve_confirm_bundle,
)


def test_confirm_group_methods_excludes_post_hoc() -> None:
    """Post-hoc rides with its seed's ce unit; it never gets its own unit."""
    for is_mil in (False, True):
        methods = confirm_group_methods(is_mil, "balanced")
        assert "post_hoc_logit_adjustment" not in methods
        assert set(methods) == set(roster_for_regime(is_mil)) - {
            "post_hoc_logit_adjustment"
        }


def test_natural_anchor_schedules_only_the_ce_reference() -> None:
    """The anchor is descriptive, so no mitigation method is fitted on it."""
    for is_mil in (False, True):
        assert confirm_group_methods(is_mil, "natural") == NATURAL_ANCHOR_METHODS


def test_natural_units_cover_one_condition_across_every_split_method_seed() -> None:
    units = confirm_units_for_group("natural", is_mil=False)
    methods = confirm_group_methods(False, "natural")

    assert len(units) == 3 * len(methods) * CONFIRMATION_SEED_COUNT
    assert all(unit.condition == "natural" for unit in units)
    seen = {(u.split_index, u.condition, u.method, u.seed_index) for u in units}
    assert len(seen) == len(units), "no unit is enumerated more than once"
    expected = {
        (split, "natural", method, seed)
        for split in range(3)
        for method in methods
        for seed in range(CONFIRMATION_SEED_COUNT)
    }
    assert seen == expected


def test_controlled_units_cover_three_conditions_across_every_split_method_seed() -> (
    None
):
    units = confirm_units_for_group("controlled", is_mil=True)
    methods = confirm_group_methods(True, "balanced")

    assert len(units) == 3 * 3 * len(methods) * CONFIRMATION_SEED_COUNT
    assert {u.condition for u in units} == {"balanced", "moderate", "severe"}
    seen = {(u.split_index, u.condition, u.method, u.seed_index) for u in units}
    assert len(seen) == len(units)
    expected = {
        (split, condition, method, seed)
        for split in range(3)
        for condition in ("balanced", "moderate", "severe")
        for method in methods
        for seed in range(CONFIRMATION_SEED_COUNT)
    }
    assert seen == expected


def test_confirm_array_size_matches_bundled_unit_count() -> None:
    total = len(confirm_units_for_group("controlled", is_mil=False))
    assert confirm_array_size("controlled", False, 1) == total
    assert confirm_array_size("controlled", False, 20) == math.ceil(total / 20)


def test_resolve_confirm_bundle_partitions_every_unit_exactly_once() -> None:
    """Every array task's bundle, concatenated in order, reconstructs the full list
    with no gaps, overlaps, or duplicates -- the correctness property the sharded
    SLURM array depends on."""
    shards_per_task = 7
    units = confirm_units_for_group("controlled", is_mil=False)
    array_size = confirm_array_size("controlled", False, shards_per_task)

    reconstructed = []
    for task_index in range(array_size):
        bundle = resolve_confirm_bundle(
            task_index, "controlled", False, shards_per_task
        )
        assert 1 <= len(bundle) <= shards_per_task
        reconstructed.extend(bundle)

    assert reconstructed == units
    # Every task past the array size resolves to an empty bundle.
    assert resolve_confirm_bundle(array_size, "controlled", False, shards_per_task) == []


def test_five_unit_bundles_cover_confirmation_without_gaps_or_duplicates() -> None:
    for group in ("natural", "controlled"):
        units = confirm_units_for_group(group, is_mil=True)
        bundles = [
            resolve_confirm_bundle(index, group, True, 5)
            for index in range(confirm_array_size(group, True, 5))
        ]
        assert [unit for bundle in bundles for unit in bundle] == units
