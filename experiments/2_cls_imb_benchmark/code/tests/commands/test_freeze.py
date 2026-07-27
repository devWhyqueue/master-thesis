from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest
import yaml

from imbalance_benchmark.analysis.reporting.secondary_intervals.report import (
    _locked_tiers,
)
from imbalance_benchmark.common import compute_sha256
from imbalance_benchmark.common import dataset_provenance
from imbalance_benchmark.common import sign_file

def _write_freeze_fixture(tmp_path: Path) -> Path:
    """Config + three signed patient-split manifests, ready for cmd_freeze."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "paths": {"outputs": str(tmp_path / "outputs")},
                "dataset": {
                    "name": "synthetic",
                    "regime": "patch",
                    "target": "synthetic_target",
                    "version": "test-fixture-v1",
                    "eligibility_rules": {"fixture": True},
                },
                "analysis": {"bootstrap_replicates": 2},
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "case_id": f"{cls}_{index}",
            "slide_id": f"{cls}_{index}",
            "patch_id": f"{cls}_{index}_patch",
            "cancer_type": cls,
            "split": "train" if index < 30 else "test",
        }
        for cls in ("A", "B")
        for index in range(40)
    ]
    for split_index in range(3):
        data_dir = tmp_path / "outputs" / f"split={split_index}" / "data"
        data_dir.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(data_dir / "manifest.csv", index=False)
        pilot = data_dir / "pilot_report.json"
        pilot.write_text(
            json.dumps({"definitive_floor": 10, "quotas": {"0": 1}, "excluded": False}),
            encoding="utf-8",
        )
        sign_file(pilot)
    return config_path

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

def test_freeze_rejects_pilot_edited_after_signing(tmp_path: Path) -> None:
    """A pilot edited after signing must not be baked into the definitive freeze.

    Finding: "Pilot evidence is not locked into the definitive freeze." The floor
    is read from the pilot at freeze time, so freezing must verify the pilot's
    signature first; otherwise a post-signing edit silently sets the frozen floor.
    """
    from imbalance_benchmark.commands.freeze import cmd_freeze

    config_path = _write_freeze_fixture(tmp_path)
    tampered = tmp_path / "outputs" / "split=0" / "data" / "pilot_report.json"
    tampered.write_text(  # altered content, stale signature
        json.dumps({"definitive_floor": 1, "quotas": {"0": 1}, "excluded": False}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="signed lock"):
        cmd_freeze(Namespace(config=str(config_path), seed=7, split_index=0))

def test_freeze_uses_one_patch_pool_for_balanced_and_every_assignment(
    tmp_path: Path,
) -> None:
    """A valid reversed assignment must reuse the balanced class pools."""
    from argparse import Namespace

    from imbalance_benchmark.manifest.freezing import _freeze_meta

    config = tmp_path / "config.yaml"
    config.write_text(
        "dataset:\n  name: synthetic\n  regime: patch\n  target: diagnosis\n",
        encoding="utf-8",
    )
    rows = []
    for class_name in ("A", "B"):
        for patient in range(10):
            n_slides = 3 if patient == 0 else 2
            for slide in range(n_slides):
                for patch in range(10):
                    rows.append(
                        {
                            "case_id": f"{class_name}_{patient}",
                            "slide_id": f"{class_name}_{patient}_{slide}",
                            "patch_id": f"{class_name}_{patient}_{slide}_{patch}",
                            "cancer_type": class_name,
                            "split": "train",
                        }
                    )
    meta = _freeze_meta(
        Namespace(seed=4, config=config),
        {"data": tmp_path},
        pd.DataFrame(rows),
        False,
        ["A", "B"],
        200,
        20,
        20,
        False,
        10,
    )

    pool_hashes = {
        info["evidence_pool_hash"]
        for conditions in [meta["conditions"], *meta["assignment_conditions"].values()]
        for info in conditions.values()
    }
    assert len(pool_hashes) == 1

def test_freeze_rejects_missing_dataset_provenance(tmp_path: Path) -> None:
    """Definitive freezes cannot replace required provenance with placeholders."""
    from imbalance_benchmark.commands.freeze_execution import _attach_provenance

    pilot = tmp_path / "pilot_report.json"
    manifest = tmp_path / "manifest.csv"
    pilot.write_text("{}", encoding="utf-8")
    manifest.write_text("case_id,split\nA,train\n", encoding="utf-8")

    with pytest.raises(ValueError, match="dataset.version"):
        _attach_provenance(
            {},
            {"data": tmp_path},
            {"dataset": {"name": "synthetic", "regime": "patch"}},
        )

def test_reject_degenerate_conditions_catches_achieved_rho_collapse() -> None:
    from imbalance_benchmark.manifest.shared_total.degenerate import (
        reject_degenerate_conditions,
    )

    meta = {
        "conditions": {"balanced": {"allocated_counts": {"A": 70, "B": 70}}},
        "assignment_conditions": {
            "native": {
                "moderate": {
                    "achieved_rho": 1.0,
                    "requested_rho": 10.0,
                    "allocated_counts": {"A": 70, "B": 70},
                    "limiting_class": "B",
                    "binding_independent_support_constraint": "independent-support floor",
                },
                "severe": {
                    "achieved_rho": 1.0,
                    "requested_rho": 100.0,
                    "allocated_counts": {"A": 70, "B": 70},
                    "limiting_class": "B",
                    "binding_independent_support_constraint": "independent-support floor",
                },
            }
        },
    }

    with pytest.raises(ValueError, match="Degenerate native/moderate"):
        reject_degenerate_conditions(meta)

def test_reject_degenerate_conditions_allows_a_capacity_bound_adversarial_assignment() -> (
    None
):
    """An adversarial (e.g. reversed) assignment tying moderate to severe at a
    real head-capacity ceiling is a data limit, not a null experiment - it
    must not be confused with collapsing back to the balanced condition."""
    from imbalance_benchmark.manifest.shared_total.degenerate import (
        reject_degenerate_conditions,
    )

    meta = {
        "conditions": {"balanced": {"allocated_counts": {"A": 70, "B": 70}}},
        "assignment_conditions": {
            "reversed": {
                "moderate": {
                    "achieved_rho": 3.5,
                    "requested_rho": 10.0,
                    "allocated_counts": {"A": 20, "B": 120},
                    "limiting_class": "B",
                    "binding_independent_support_constraint": "unique-support availability",
                },
                "severe": {
                    "achieved_rho": 3.5,
                    "requested_rho": 100.0,
                    "allocated_counts": {"A": 20, "B": 120},
                    "limiting_class": "B",
                    "binding_independent_support_constraint": "unique-support availability",
                },
            }
        },
    }

    reject_degenerate_conditions(meta)

def test_dataset_provenance_requires_a_frozen_target() -> None:
    dataset = {
        "name": "panda",
        "regime": "wsi",
        "version": "v1",
        "eligibility_rules": {"slide_qc": "pass"},
    }

    with pytest.raises(ValueError, match="dataset.target"):
        dataset_provenance(dataset)

    provenance = dataset_provenance({**dataset, "target": "isup_grade"})

    assert provenance["target"] == "isup_grade"

def test_freeze_metadata_is_content_locked(tmp_path: Path) -> None:
    """Changing a frozen design field must be detected even without a CSV edit."""
    from imbalance_benchmark.common import write_json
    from imbalance_benchmark.manifest.freeze import (
        lock_manifest_freeze,
        verify_manifest_freeze,
    )

    freeze_path = tmp_path / "manifest_freeze.json"
    write_json(freeze_path, {"shared_T": 100, "conditions": {}})
    freeze = lock_manifest_freeze({"shared_T": 100, "conditions": {}})
    verify_manifest_freeze(freeze)

    freeze["shared_T"] = 200
    with pytest.raises(RuntimeError, match="content"):
        verify_manifest_freeze(freeze)

def test_freeze_verifies_pilot_and_prepared_manifest_artifacts(tmp_path: Path) -> None:
    """Held-out manifest or pilot changes invalidate the frozen record."""
    from imbalance_benchmark.common import sign_file, write_json
    from imbalance_benchmark.manifest.freeze import verify_manifest_freeze

    pilot = tmp_path / "pilot_report.json"
    manifest = tmp_path / "manifest.csv"
    write_json(pilot, {"definitive_floor": 10})
    write_json(manifest, {"held_out": "locked"})
    sign_file(pilot)
    meta = {
        "content_sha256": "",
        "pilot_report": {"path": str(pilot), "sha256": compute_sha256(pilot)},
        "prepared_manifest": {
            "path": str(manifest),
            "sha256": compute_sha256(manifest),
        },
    }
    from imbalance_benchmark.manifest.freeze import lock_manifest_freeze

    frozen = lock_manifest_freeze(meta)
    verify_manifest_freeze(frozen)
    write_json(manifest, {"held_out": "changed"})
    with pytest.raises(RuntimeError, match="Prepared manifest altered"):
        verify_manifest_freeze(frozen)
