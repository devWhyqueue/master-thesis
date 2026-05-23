from __future__ import annotations

from typing import Any

PATCH_METHOD_METADATA: dict[str, dict[str, Any]] = {
    "patch_ce": {
        "role": "baseline",
        "taxonomy_category": "baseline",
        "representative_paper": "Unweighted cross-entropy",
    },
    "patch_weighted_ce": {
        "role": "baseline",
        "taxonomy_category": "algorithm-level losses",
        "representative_paper": "Class-weighted cross-entropy",
    },
    "patch_focal": {
        "role": "baseline",
        "taxonomy_category": "algorithm-level losses",
        "representative_paper": "Focal loss",
    },
    "patch_balanced_sampler_ce": {
        "role": "baseline",
        "taxonomy_category": "data-level sampling",
        "representative_paper": "Class-balanced sampling",
    },
    "patch_ce_soft_f1_balanced": {
        "role": "representative",
        "taxonomy_category": "algorithm-level losses",
        "representative_paper": "Scholz et al.; CE + soft F1 + balanced sampling",
    },
    "patch_ce_soft_mcc_balanced": {
        "role": "representative",
        "taxonomy_category": "algorithm-level losses",
        "representative_paper": "Scholz et al.; CE + soft MCC + balanced sampling",
    },
    "patch_progan_aug": {
        "role": "representative",
        "taxonomy_category": "synthetic generation",
        "representative_paper": "Ruiz-Casado et al.; ProGAN augmentation adaptation",
    },
}

PATCH_FEATURE_METHOD_ALIASES = {
    "patch_feature_ce": "patch_ce",
    "patch_feature_weighted_ce": "patch_weighted_ce",
    "patch_feature_focal": "patch_focal",
    "patch_feature_balanced_sampler_ce": "patch_balanced_sampler_ce",
    "patch_feature_ce_soft_f1_balanced": "patch_ce_soft_f1_balanced",
    "patch_feature_ce_soft_mcc_balanced": "patch_ce_soft_mcc_balanced",
    "patch_feature_progan_aug": "patch_progan_aug",
}

WSI_METHOD_METADATA: dict[str, dict[str, Any]] = {
    "mil_ce": {
        "role": "baseline",
        "taxonomy_category": "baseline",
        "representative_paper": "Attention MIL with cross-entropy",
    },
    "mil_weighted_ce": {
        "role": "baseline",
        "taxonomy_category": "algorithm-level losses",
        "representative_paper": "Class-weighted attention MIL",
    },
    "mil_focal": {
        "role": "baseline",
        "taxonomy_category": "algorithm-level losses",
        "representative_paper": "Focal attention MIL",
    },
    "mil_balanced_sampler_ce": {
        "role": "baseline",
        "taxonomy_category": "data-level sampling",
        "representative_paper": "Balanced-sampling attention MIL",
    },
    "rankmix_mil": {
        "role": "representative",
        "taxonomy_category": "data-level augmentation",
        "representative_paper": "Chen and Lu; RankMix",
    },
    "sc_mil": {
        "role": "representative",
        "taxonomy_category": "representation and architecture",
        "representative_paper": "Juyal et al.; SC-MIL",
    },
    "mde_mil": {
        "role": "representative",
        "taxonomy_category": "hybrid approaches",
        "representative_paper": "Ling et al.; MDE-MIL (ensemble, no distillation)",
    },
}


def benchmark_metadata(benchmark: str, method: str) -> dict[str, Any]:
    """Return stable metadata for one benchmark method."""
    tables = {"patch": PATCH_METHOD_METADATA, "wsi_bag": WSI_METHOD_METADATA}
    key = PATCH_FEATURE_METHOD_ALIASES.get(method, method)
    return tables[benchmark][key]
