"""The corrected extraction scope is enforced before any fit starts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from imbalance_benchmark.common import sign_file
from transfer import preflight


def test_feature_audit_requires_corrected_scope(tmp_path: Path, monkeypatch) -> None:
    config = {"dataset": {"name": "bracs"}}
    monkeypatch.setattr(preflight, "_LOCK_DIR", tmp_path)
    monkeypatch.setattr(preflight, "output_root", lambda cfg: tmp_path)
    (tmp_path / "phase05_scope_amendment.json").write_text(
        json.dumps({"bracs": {"union_patches": 3, "union_unique_slides": 1}})
    )
    audit_path = tmp_path / "data" / "feature_audit.json"
    audit_path.parent.mkdir()
    audit = {
        "unresolved_missing": 0,
        "unresolved_corrupt": 0,
        "unresolved_mismatched": 0,
        "uni2h": {"requested_patches": 2, "matched_patches": 2, "requested_slides": 1},
        "virchow2": {"matched_patches": 2},
    }
    audit_path.write_text(json.dumps(audit))
    sign_file(audit_path)
    with pytest.raises(RuntimeError, match="corrected scope"):
        preflight._verify_feature_audit(config)
    audit["uni2h"].update(requested_patches=3, matched_patches=3)
    audit["virchow2"]["matched_patches"] = 3
    audit_path.write_text(json.dumps(audit))
    sign_file(audit_path)
    assert preflight._verify_feature_audit(config) == audit
