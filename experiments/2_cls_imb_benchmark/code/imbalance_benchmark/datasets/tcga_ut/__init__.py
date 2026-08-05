"""TCGA-UT participant identity, image/feature manifests, and case splits.

The WSI (bag) regime still consumes pre-extracted chunked ``cls_patchmean``
tensors (:mod:`.tensors`). The patch regime reads every image directly from
the Zenodo-authenticated, project-owned SqFS built by :mod:`.pack`
(:mod:`.image`). :mod:`.splits` is shared by both regimes.
"""

from __future__ import annotations

from imbalance_benchmark.datasets.tcga_ut.image import (
    build_image_manifest,
    build_image_rows,
    iter_class_slide_images,
    validate_image_cohort,
    validate_source_provenance,
)
from imbalance_benchmark.datasets.tcga_ut.splits import (
    assert_case_disjoint,
    assign_class_splits,
    split_cases,
    tcga_case_id,
)
from imbalance_benchmark.datasets.tcga_ut.tensors import (
    build_feature_manifest,
    build_manifest,
    collect_slide_labels,
    strip_feature_suffix,
    validate_feature_coverage,
)

__all__ = [
    "assert_case_disjoint",
    "assign_class_splits",
    "build_feature_manifest",
    "build_image_manifest",
    "build_image_rows",
    "build_manifest",
    "collect_slide_labels",
    "iter_class_slide_images",
    "split_cases",
    "strip_feature_suffix",
    "tcga_case_id",
    "validate_feature_coverage",
    "validate_image_cohort",
    "validate_source_provenance",
]
