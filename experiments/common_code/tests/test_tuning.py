"""Unit tests for shared experiment utilities."""

from common_code.tuning.grid import patch_feature_grid, wsi_bag_grid
from common_code.tuning.registry import patch_feature_method_flags


def test_patch_feature_grid_has_expected_methods() -> None:
    methods = {variant.method for variant in patch_feature_grid()}
    assert "patch_feature_focal" in methods
    assert "patch_feature_progan_aug" in methods


def test_wsi_grid_has_six_method_families() -> None:
    methods = {variant.method for variant in wsi_bag_grid()}
    assert methods == {
        "mil_weighted_ce",
        "mil_focal",
        "mil_balanced_sampler_ce",
        "rankmix_mil",
        "sc_mil",
        "mde_mil",
    }


def test_patch_feature_method_flags_cover_grid_methods() -> None:
    for variant in patch_feature_grid():
        if variant.method in {
            "patch_feature_cfal",
            "patch_feature_divide_conquer",
            "patch_feature_progan_aug",
        }:
            continue
        flags = patch_feature_method_flags(variant.method)
        assert flags
