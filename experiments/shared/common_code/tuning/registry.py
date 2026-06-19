"""Canonical method IDs and CLI flag mappings."""

from __future__ import annotations

PATCH_FEATURE_METHOD_FLAGS: dict[str, list[str]] = {
    "patch_feature_ce": ["--loss=cross_entropy", "--alpha=uniform"],
    "patch_feature_weighted_ce": [
        "--loss=cross_entropy",
        "--alpha=inverse_class_frequency",
    ],
    "patch_feature_focal": ["--loss=focal_loss", "--alpha=inverse_class_frequency"],
    "patch_feature_balanced_sampler_ce": [
        "--loss=cross_entropy",
        "--alpha=uniform",
        "--batch-balancing",
    ],
    "patch_feature_ce_soft_f1_balanced": [
        "--loss=ce_soft_f1",
        "--alpha=uniform",
        "--batch-balancing",
    ],
    "patch_feature_ce_soft_mcc_balanced": [
        "--loss=ce_soft_mcc",
        "--alpha=uniform",
        "--batch-balancing",
    ],
    "patch_feature_cfal": ["--training-method=patch_feature_cfal"],
    "patch_feature_divide_conquer": ["--training-method=patch_feature_divide_conquer"],
    "patch_feature_progan_aug": ["--training-method=patch_feature_progan_aug"],
    "patch_feature_oko": ["--training-method=patch_feature_oko"],
}

WSI_METHOD_FLAGS: dict[str, list[str]] = {
    "mil_ce": [],
    "mil_weighted_ce": ["--weight-power=0.125"],
    "mil_focal": ["--focal-gamma=1.0"],
    "mil_balanced_sampler_ce": ["--sampler-power=1.0"],
    "rankmix_mil": ["--rankmix-alpha=1.0"],
    "sc_mil": ["--sc-mil-temperature=0.1"],
    "mde_mil": ["--mde-mil-consistency-weight=0.3"],
}


def patch_feature_method_flags(method: str) -> list[str]:
    """Return the CLI flag bundle for one patch-feature method."""
    if method not in PATCH_FEATURE_METHOD_FLAGS:
        raise ValueError(f"Unknown patch-feature method: {method}")
    return PATCH_FEATURE_METHOD_FLAGS[method]


def wsi_method_flags(method: str) -> list[str]:
    """Return the CLI flag bundle for one WSI method."""
    if method not in WSI_METHOD_FLAGS:
        raise ValueError(f"Unknown WSI method: {method}")
    return WSI_METHOD_FLAGS[method]
