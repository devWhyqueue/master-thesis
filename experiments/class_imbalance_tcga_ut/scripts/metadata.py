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
    "patch_cfal": {
        "role": "representative",
        "taxonomy_category": "algorithm-level losses",
        "representative_paper": "Mahbub et al.; CFAL",
    },
    "patch_progan_aug": {
        "role": "representative",
        "taxonomy_category": "synthetic generation",
        "representative_paper": "Ruiz-Casado et al.; ProGAN augmentation adaptation",
    },
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
}


def benchmark_metadata(benchmark: str, method: str) -> dict[str, Any]:
    """Return stable metadata for one benchmark method."""
    tables = {"patch": PATCH_METHOD_METADATA, "wsi_bag": WSI_METHOD_METADATA}
    return tables[benchmark][method]
