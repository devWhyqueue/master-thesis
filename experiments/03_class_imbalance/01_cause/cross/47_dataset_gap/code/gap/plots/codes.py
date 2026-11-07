"""TCGA study abbreviations for the 30 analysed TCGA-UT cancer types (figure labels only)."""

from __future__ import annotations

__all__ = ["TCGA_CODES", "short_name"]

TCGA_CODES: dict[str, str] = {
    "Adrenocortical_carcinoma": "ACC",
    "Bladder_Urothelial_Carcinoma": "BLCA",
    "Brain_Lower_Grade_Glioma": "LGG",
    "Breast_invasive_carcinoma": "BRCA",
    "Cervical_squamous_cell_carcinoma_and_endocervical_adenocarcinoma": "CESC",
    "Colon_adenocarcinoma": "COAD",
    "Esophageal_carcinoma": "ESCA",
    "Glioblastoma_multiforme": "GBM",
    "Head_and_Neck_squamous_cell_carcinoma": "HNSC",
    "Kidney_Chromophobe": "KICH",
    "Kidney_renal_clear_cell_carcinoma": "KIRC",
    "Kidney_renal_papillary_cell_carcinoma": "KIRP",
    "Liver_hepatocellular_carcinoma": "LIHC",
    "Lung_adenocarcinoma": "LUAD",
    "Lung_squamous_cell_carcinoma": "LUSC",
    "Mesothelioma": "MESO",
    "Ovarian_serous_cystadenocarcinoma": "OV",
    "Pancreatic_adenocarcinoma": "PAAD",
    "Pheochromocytoma_and_Paraganglioma": "PCPG",
    "Prostate_adenocarcinoma": "PRAD",
    "Rectum_adenocarcinoma": "READ",
    "Sarcoma": "SARC",
    "Skin_Cutaneous_Melanoma": "SKCM",
    "Stomach_adenocarcinoma": "STAD",
    "Testicular_Germ_Cell_Tumors": "TGCT",
    "Thymoma": "THYM",
    "Thyroid_carcinoma": "THCA",
    "Uterine_Carcinosarcoma": "UCS",
    "Uterine_Corpus_Endometrial_Carcinoma": "UCEC",
    "Uveal_Melanoma": "UVM",
}


def short_name(name: str) -> str:
    """TCGA abbreviation for a TCGA-UT class; BRACS class names are already short."""
    return TCGA_CODES.get(name, name)
