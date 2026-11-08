"""Sparse, resumable, queue-capped submission and the technical pilot check (no SLURM, no SSH)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from lora_decomposition import check, protocol, submit
from lora_decomposition.protocol import ARMS, load_protocol_config

_CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    loaded = load_protocol_config(_CONFIGS / "bracs.yaml")
    monkeypatch.setattr(protocol, "EXPERIMENT_DIR", tmp_path)
    loaded["paths"]["outputs"] = str(tmp_path / "outputs" / "bracs" / "patch")
    return loaded


def _array_indices(script: str) -> list[int]:
    spec = re.search(r"#SBATCH --array=([0-9,]+)", script).group(1)
    return [int(i) for i in spec.split(",")]


def test_phase_indices_and_validation() -> None:
    assert submit.phase_indices("pilot") == (0, 10, 20)
    main = submit.phase_indices("main")
    assert len(main) == 27 and not set(main) & {0, 10, 20} and set(main) | {0, 10, 20} == set(range(30))
    with pytest.raises(ValueError):
        submit.phase_indices("everything")
    with pytest.raises(ValueError):
        submit.check_indices([0, 30])


def test_completed_and_in_flight_work_is_not_resubmitted(config: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(submit, "is_marked", lambda c, s, d, arm: (arm, s * 10 + d) == ("B", 0))
    submit.record_submission(config, "777", "F", [10], "pilot")
    scripts: list[str] = []
    ids = iter(["900", "901", "902", "903"])

    def sbatch(script: str) -> str:
        scripts.append(script)
        return next(ids)

    submitted = submit.submit_jobs(config, "pilot", False, queue=lambda: ["777_10"], sbatch=sbatch)
    planned = {arm: _array_indices(s) for arm, s in zip(("B", "F", "C", "R"), scripts)}
    assert planned == {"B": [10, 20], "F": [0, 20], "C": [0, 10, 20], "R": [0, 10, 20]}
    assert submitted == ["900", "901", "902", "903"]
    assert all("fit --arm" in s and "imb-fit" in s and "--time" not in s for s in scripts)
    stored = json.loads(submit.submissions_path(config).read_text(encoding="utf-8"))
    assert {"job_id": "903", "arm": "R", "indices": [0, 10, 20], "phase": "pilot"}.items() <= stored[-1].items()


def test_queue_cap_reserves_every_array_worker(config: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(submit, "is_marked", lambda *a: False)
    queue = [f"5_{i}" for i in range(98)]
    scripts: list[str] = []

    def sbatch(script: str) -> str:
        scripts.append(script)
        queue.extend(f"6_{i}" for i in _array_indices(script))
        return "6"

    submit.submit_jobs(config, "pilot", False, queue=lambda: list(queue), sbatch=sbatch)
    assert [len(_array_indices(s)) for s in scripts] == [2]
    assert len(queue) == 100


def test_empty_pending_and_dry_run_touch_no_scheduler(config: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args):
        raise AssertionError("scheduler called")

    monkeypatch.setattr(submit, "is_marked", lambda *a: True)
    assert submit.submit_jobs(config, "pilot", False, queue=lambda: [], sbatch=forbidden) == []
    monkeypatch.setattr(submit, "is_marked", lambda *a: False)
    dry = submit.submit_jobs(config, "pilot", True, queue=forbidden, sbatch=forbidden)
    assert dry == [f"dry-run-{arm}" for arm in ARMS]


def test_main_phase_needs_passed_pilot_check_and_review(config: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(submit, "is_marked", lambda *a: False)
    with pytest.raises(ValueError, match="pilot"):
        submit.submit_jobs(config, "main", True, queue=list, sbatch=str)
    report = tmp_path / "report"
    report.mkdir()
    (report / "check_pilot.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    (report / "pilot_check.md").write_text("reviewed", encoding="utf-8")
    assert len(submit.submit_jobs(config, "main", True, queue=list, sbatch=str)) == 4


def _audit(seq: str, rows: list[int]) -> dict:
    return {
        "initialization_sha256": "init", "sample_sequence_sha256": seq, "unique_rows_per_class": rows,
        "sample_count": 600 * 256, "loss_finite": True, "first_loss_mean": 2.0, "last_loss_mean": 1.0,
    }


def _block_provenance(broken: bool = False) -> dict[str, dict]:
    balanced, skewed = [5, 5], [9, 1]
    base = {"seed": 48000, "recipe": {"steps": 600, "batch_size": 256}}
    prov = {
        "B": {**base, "data_counts": balanced, "target_counts": balanced, "class_weights": [1, 1],
              "training_identity_sha256": "r1", "training_audit": _audit("s", [5, 5])},
        "F": {**base, "data_counts": balanced, "target_counts": skewed, "class_weights": [1.8, 0.2],
              "training_identity_sha256": "r1", "training_audit": _audit("s", [5, 5])},
        "C": {**base, "data_counts": skewed, "target_counts": balanced, "class_weights": [5 / 9, 5],
              "training_identity_sha256": "r100", "training_audit": _audit("s", [9, 1])},
        "R": {**base, "data_counts": skewed, "target_counts": skewed, "class_weights": [1, 1],
              "training_identity_sha256": "r100", "training_audit": _audit("s", [9, 1])},
    }
    if broken:
        prov["F"]["training_audit"]["initialization_sha256"] = "other"
    return prov


def test_block_audit_is_technical_only() -> None:
    assert check.block_problems(_block_provenance()) == []
    problems = check.block_problems(_block_provenance(broken=True))
    assert problems == ["initialization differs across arms"]
