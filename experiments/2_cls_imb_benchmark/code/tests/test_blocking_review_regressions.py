from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from imbalance_benchmark.analysis.inference.context import BootstrapContext
from imbalance_benchmark.analysis.reporting.calibration_intervals import (
    _distribution_summary,
)
from imbalance_benchmark.analysis.reporting.clustered_endpoints import (
    clustered_endpoints,
)
from imbalance_benchmark.analysis.reporting.secondary_intervals.report import (
    _locked_tiers,
)
from imbalance_benchmark.construction import effective_rho
from imbalance_benchmark.common import compute_sha256
from imbalance_benchmark.datasets.bracs import wsi as bracs_wsi
from imbalance_benchmark.datasets.panda_audit import (
    validate_selection,
    validate_tile_inventory,
)


def test_effective_rho_finds_narrow_highest_feasible_interval() -> None:
    available = [75928, 168239, 36174, 130815, 104503, 32398, 127285]

    result = effective_rho(available, 100.0, 30, 226792)

    assert result == pytest.approx(9.251803719617433, rel=1e-10)


def test_locked_tiers_read_the_current_split_freeze(tmp_path: Path) -> None:
    paths = {"root": tmp_path / "split=1", "data": tmp_path / "split=1" / "data"}
    paths["data"].mkdir(parents=True)
    freeze = {
        "assignment_conditions": {
            "native": {
                "severe": {"allocated_counts": {"A": 10, "B": 100}}
            }
        },
        "tail_assignments": {"native": ["B", "A"]},
    }
    (paths["data"] / "manifest_freeze.json").write_text(json.dumps(freeze))

    assert _locked_tiers(paths, "native", "severe", ["A", "B"]) == {
        "A": "tail",
        "B": "head",
    }


def test_secondary_bootstrap_includes_cluster_macro_endpoints() -> None:
    labels = np.array([0, 1, 1, 0])
    predictions = np.array([[0, 0, 1, 1]])
    probabilities = np.array(
        [[[0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.3, 0.7]]]
    )
    identity = pd.DataFrame(
        {
            "case_id": ["c1", "c1", "c1", "c2"],
            "slide_id": ["s1", "s1", "s2", "s3"],
        }
    )
    context = object.__new__(BootstrapContext)
    context.case_ids = identity["case_id"].to_numpy()
    context.slide_ids = identity["slide_id"].to_numpy()
    context.row_weights = np.ones((4, 3), dtype=np.int64)
    context.n_replicates = 3
    context._seed = 7
    context._seed_indices = {}

    result = context.secondary_distributions(
        labels, predictions, probabilities, ["A", "B"], {}
    )
    observed = clustered_endpoints(
        labels, predictions[0], probabilities[0], identity
    )

    for endpoint in (
        "patch_micro_accuracy",
        "slide_macro_accuracy",
        "patient_macro_accuracy",
        "slide_macro_balanced_accuracy",
        "patient_macro_balanced_accuracy",
        "slide_macro_f1",
        "patient_macro_f1",
        "slide_macro_nll",
        "patient_macro_nll",
        "slide_macro_brier",
        "patient_macro_brier",
    ):
        assert result[endpoint][0] == pytest.approx(observed[endpoint])


def test_wsi_secondary_outputs_use_only_applicable_endpoint_names() -> None:
    labels = np.array([0, 1])
    predictions = np.array([[0, 1]])
    probabilities = np.array([[[0.9, 0.1], [0.1, 0.9]]])
    context = object.__new__(BootstrapContext)
    context.case_ids = np.array(["c1", "c2"])
    context.slide_ids = np.array(["s1", "s2"])
    context.row_weights = np.ones((2, 2), dtype=np.int64)
    context.n_replicates = 2
    context._seed = 7
    context._seed_indices = {}

    tcga = context.secondary_distributions(
        labels,
        predictions,
        probabilities,
        ["LUAD", "LUSC"],
        {},
        is_mil=True,
        ordinal=False,
    )
    panda = context.secondary_distributions(
        labels,
        predictions,
        probabilities,
        ["ISUP0", "ISUP1"],
        {},
        is_mil=True,
        ordinal=True,
    )

    assert "patch_micro_accuracy" not in tcga
    assert "quadratic_weighted_kappa" not in tcga
    assert "ordinal_mean_absolute_error" not in tcga
    assert "patch_micro_accuracy" not in panda
    assert "quadratic_weighted_kappa" in panda
    assert "ordinal_mean_absolute_error" in panda


def test_wsi_run_endpoints_do_not_call_slide_accuracy_patch_accuracy() -> None:
    identity = pd.DataFrame(
        {"case_id": ["c1", "c2"], "slide_id": ["s1", "s2"]}
    )

    endpoints = clustered_endpoints(
        np.array([0, 1]),
        np.array([0, 1]),
        np.array([[0.9, 0.1], [0.1, 0.9]]),
        identity,
        is_mil=True,
    )

    assert "patch_micro_accuracy" not in endpoints


def test_calibration_summary_separates_observed_estimate_from_bootstrap() -> None:
    assert _distribution_summary([0.9, 0.1, 0.2], "ECE") == {
        "ECE": 0.9,
        "ECE 95% CI": "[0.103, 0.198]",
    }


def test_bracs_wsi_manifest_rejects_different_slide_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    official = pd.DataFrame(
        {
            "case_id": ["pA", "pB"],
            "slide_id": ["A", "B"],
            "slide_label": ["N", "N"],
        }
    )
    tiles = pd.DataFrame(
        {"slide_id": ["A", "C"], "image_path": ["A.jpg", "C.jpg"]}
    )
    monkeypatch.setattr(bracs_wsi, "load_wsi_metadata", lambda *_args: official)
    monkeypatch.setattr(bracs_wsi, "load_tile_manifest", lambda *_args: tiles)
    monkeypatch.setattr(
        bracs_wsi.metadata,
        "split_cases",
        lambda *_args: pd.DataFrame({"case_id": ["pA"], "split": ["train"]}),
    )

    with pytest.raises(ValueError, match="slide IDs"):
        bracs_wsi.build_manifest(tmp_path, tmp_path, 0, expected_slides=2)


def test_panda_tile_inventory_rejects_duplicate_selection_rows(
    tmp_path: Path,
) -> None:
    image_a, image_b = tmp_path / "A.jpg", tmp_path / "B.jpg"
    image_a.write_bytes(b"A")
    image_b.write_bytes(b"B")
    selection = pd.DataFrame(
        {
            "slide_id": ["A", "A", "B"],
            "eligible_tile_count": [1, 1, 1],
            "source_level": [0, 0, 0],
            "tile_size": [256, 256, 256],
            "tissue_fraction_min": [0.35, 0.35, 0.35],
            "tissue_intensity_threshold": [210, 210, 210],
        }
    )
    inventory = {
        slide: pd.DataFrame(
            {
                "patch_id": ["p0"],
                "image_path": [str(path)],
                "level": [0],
                "tile_size": [256],
                "x": [0],
                "y": [0],
                "tissue_fraction": [0.5],
                "tissue_intensity_threshold": [210],
            }
        )
        for slide, path in (("A", image_a), ("B", image_b))
    }

    with pytest.raises(ValueError, match="duplicate"):
        validate_tile_inventory(
            selection,
            inventory,
            pd.DataFrame({"slide_id": ["A", "B"]}),
            expected_slides=2,
        )


def test_panda_selection_matches_official_ids_labels_providers_and_masks() -> None:
    official = pd.DataFrame(
        {
            "slide_id": ["A", "B"],
            "slide_label": ["ISUP0", "ISUP1"],
            "provider": ["radboud", "karolinska"],
            "has_mask": [True, False],
        }
    )
    selection = pd.DataFrame(
        {
            "slide_id": ["A", "B"],
            "slide_label": ["ISUP0", "ISUP5"],
            "provider": ["radboud", "karolinska"],
            "has_mask": [True, False],
            "eligible_tile_count": [1, 1],
            "source_level": [0, 0],
            "tile_size": [256, 256],
            "tissue_fraction_min": [0.35, 0.35],
            "tissue_intensity_threshold": [210, 210],
        }
    )

    with pytest.raises(ValueError, match="official slide_label"):
        validate_selection(selection, official, 2)

    selection.loc[1, "slide_label"] = "ISUP1"
    selection.loc[1, "slide_id"] = "C"
    with pytest.raises(ValueError, match="slide IDs"):
        validate_selection(selection, official, 2)


def test_panda_tile_audit_checks_hash_and_provider_specific_mask_label(
    tmp_path: Path,
) -> None:
    image_path, mask_path = tmp_path / "tile.jpg", tmp_path / "mask.tiff"
    image_path.write_bytes(b"tile")
    Image.fromarray(np.full((256, 256), 2, dtype=np.uint8)).save(mask_path)
    selection = pd.DataFrame(
        {
            "slide_id": ["A"],
            "slide_label": ["ISUP0"],
            "provider": ["radboud"],
            "has_mask": [True],
            "eligible_tile_count": [1],
            "source_level": [0],
            "tile_size": [256],
            "tissue_fraction_min": [0.35],
            "tissue_intensity_threshold": [210],
        }
    )
    tiles = pd.DataFrame(
        {
            "patch_id": ["p0"],
            "patch_label": ["cancer"],
            "image_path": [str(image_path)],
            "sha256": [compute_sha256(image_path)],
            "level": [0],
            "tile_size": [256],
            "x": [0],
            "y": [0],
            "tissue_fraction": [0.5],
            "tissue_intensity_threshold": [210],
        }
    )
    official = pd.DataFrame(
        {
            "slide_id": ["A"],
            "provider": ["radboud"],
            "has_mask": [True],
            "mask_path": [str(mask_path)],
        }
    )

    with pytest.raises(ValueError, match="patch label"):
        validate_tile_inventory(selection, {"A": tiles}, official, 1)

    tiles.loc[0, "patch_label"] = "benign"
    tiles.loc[0, "sha256"] = "wrong"
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_tile_inventory(selection, {"A": tiles}, official, 1)
