from __future__ import annotations

METHOD_LABELS = {
    "patch_ce": "CE",
    "patch_weighted_ce": "Weighted CE",
    "patch_focal": "Focal",
    "patch_balanced_sampler_ce": "Balanced sampler",
    "patch_ce_soft_f1_balanced": "CE + soft F1 (balanced)",
    "patch_ce_soft_mcc_balanced": "CE + soft MCC (balanced)",
    "patch_progan_aug": "ProGAN augmentation",
    "patch_cfal": "CFAL",
    "patch_dnc": "D&C",
    "patch_feature_ce": "CE",
    "patch_feature_weighted_ce": "Weighted CE",
    "patch_feature_focal": "Focal",
    "patch_feature_balanced_sampler_ce": "Balanced sampler",
    "patch_feature_ce_soft_f1_balanced": "CE + soft F1 (balanced)",
    "patch_feature_ce_soft_mcc_balanced": "CE + soft MCC (balanced)",
    "patch_feature_progan_aug": "ProGAN augmentation",
    "patch_feature_cfal": "CFAL",
    "patch_feature_divide_conquer": "D&C",
    "patch_feature_oko": "OKO",
    "mil_ce": "MIL CE",
    "mil_weighted_ce": "Weighted MIL",
    "mil_focal": "Focal MIL",
    "mil_balanced_sampler_ce": "Balanced MIL",
    "rankmix_mil": "RankMix",
    "sc_mil": "SC-MIL",
    "mde_mil": "MDE-MIL (ensemble)",
}


def method_label(method: str) -> str:
    """Return a human-readable label for one benchmark method."""
    return METHOD_LABELS.get(method, method.replace("_", " "))


def latex_method_label(method: str) -> str:
    """Return a method label with characters escaped for LaTeX tables."""
    return method_label(method).replace("&", r"\&")
