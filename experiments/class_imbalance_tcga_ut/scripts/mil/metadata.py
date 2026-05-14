from __future__ import annotations

from typing import Any


METHOD_METADATA: dict[str, dict[str, Any]] = {
    "ce": {
        "role": "baseline",
        "taxonomy_category": "baseline",
        "representative_paper": "Unweighted cross-entropy",
        "fidelity": "reference baseline",
        "controlled_omissions": [],
    },
    "weighted_ce": {
        "role": "baseline",
        "taxonomy_category": "algorithm-level losses",
        "representative_paper": "Class-weighted cross-entropy",
        "fidelity": "standard baseline",
        "controlled_omissions": [],
    },
    "focal": {
        "role": "baseline",
        "taxonomy_category": "algorithm-level losses",
        "representative_paper": "Focal loss",
        "fidelity": "standard baseline",
        "controlled_omissions": [],
    },
    "balanced_sampler_ce": {
        "role": "baseline",
        "taxonomy_category": "data-level sampling",
        "representative_paper": "Class-balanced sampling",
        "fidelity": "standard baseline",
        "controlled_omissions": [],
    },
    "knn": {
        "role": "baseline",
        "taxonomy_category": "feature-space probe",
        "representative_paper": "k-nearest neighbors",
        "fidelity": "reference baseline",
        "controlled_omissions": ["uses mean-pooled slide vectors"],
    },
    "ncc": {
        "role": "baseline",
        "taxonomy_category": "feature-space probe",
        "representative_paper": "Nearest-centroid classifier",
        "fidelity": "reference baseline",
        "controlled_omissions": ["uses mean-pooled slide vectors"],
    },
    "rankmix_mil": {
        "role": "representative",
        "taxonomy_category": "data-level sampling and augmentation",
        "representative_paper": "Chen and Lu; RankMix",
        "fidelity": "WSI feature-bag adaptation",
        "controlled_omissions": ["pseudo-label pretraining is omitted"],
    },
    "feature_gan_mil": {
        "role": "representative",
        "taxonomy_category": "synthetic data generation",
        "representative_paper": "GAN/WGAN minority augmentation papers",
        "fidelity": "frozen-feature synthetic analogue",
        "controlled_omissions": [
            "image-level GAN generation is implemented as a separate patch export stage",
            "generated images are not used in the frozen-feature classifier without a Virchow2 encoding pass",
        ],
    },
    "cfal_mil": {
        "role": "representative",
        "taxonomy_category": "algorithm-level losses",
        "representative_paper": "Mahbub et al.; CFAL",
        "fidelity": "prototype-affinity implementation on frozen bags",
        "controlled_omissions": ["no end-to-end image encoder fine-tuning"],
    },
    "mde_mil": {
        "role": "representative",
        "taxonomy_category": "hybrid approaches",
        "representative_paper": "Ling et al.; MDE-MIL",
        "fidelity": "two-expert MIL adaptation",
        "controlled_omissions": ["multimodal pathology-text distillation is omitted"],
    },
    "sc_mil": {
        "role": "representative",
        "taxonomy_category": "representation and architecture",
        "representative_paper": "Juyal et al.; SC-MIL",
        "fidelity": "supervised-contrastive MIL adaptation",
        "controlled_omissions": ["progressive phase schedule is simplified"],
    },
}


BAG_METHODS = {"rankmix_mil", "feature_gan_mil", "cfal_mil", "mde_mil", "sc_mil"}


def method_metadata(method: str) -> dict[str, Any]:
    """Return stable reporting metadata for a method."""
    default = {
        "role": "baseline",
        "taxonomy_category": "unclassified",
        "representative_paper": method.replace("_", " "),
        "fidelity": "experiment method",
        "controlled_omissions": [],
    }
    return METHOD_METADATA.get(method, default)
