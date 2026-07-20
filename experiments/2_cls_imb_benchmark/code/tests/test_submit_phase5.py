"""Tests for Phase 5's dependency-linked Hydra submission workflow."""

from __future__ import annotations

from imbalance_benchmark.hydra.workflow import (
    build_workflow,
    render_sbatch,
    submit_workflow,
)

SQUASHFS_SOURCE = "/home/space/datasets-sqfs/panda-native-tiles-20x-256.sqfs"
SQUASHFS_MOUNT = "/home/space/datasets/panda/native_tiles_20x_256"
PANDA_RAW = "/home/space/datasets/panda/raw"


def _config() -> dict[str, object]:
    return {
        "dataset": {"root": PANDA_RAW},
        "slurm": {
            "project_root": "/home/example/master-thesis",
            "code_dir": "/home/example/master-thesis/experiments/2_cls_imb_benchmark/code",
            "container": "/home/example/environment.sif",
            "test_partition": "gpu-test",
            "squashfs": [
                {
                    "source": SQUASHFS_SOURCE,
                    "mount": SQUASHFS_MOUNT,
                    "stages": ["prepare"],
                }
            ],
        }
    }


def test_workflow_has_afterok_condition_arrays() -> None:
    """The main DAG serializes setup and fans training work out by condition."""
    jobs = build_workflow(_config())
    assert [job.name for job in jobs] == [
        "prepare",
        "pilot",
        "freeze",
        "tune",
        "confirm",
        "analyze",
    ]
    assert jobs[3].array_conditions == ("natural", "balanced", "moderate", "severe")
    assert jobs[4].dependency == "tune"
    script = render_sbatch(jobs[4], _config(), "config.yaml")
    assert "#SBATCH --array=0-11" in script
    assert (
        "python /home/example/master-thesis/experiments/2_cls_imb_benchmark/code/__main__.py --config config.yaml --split-index"
        in script
    )
    assert "APPTAINERENV_PYTHONPATH" in script
    assert (
        "/outputs:/home/example/master-thesis/experiments/2_cls_imb_benchmark/outputs:rw"
        in script
    )


def test_submit_links_actual_job_ids() -> None:
    """Submission turns stage names into the preceding scheduler job IDs."""
    submitted_scripts: list[str] = []

    def fake_submit(script: str, dry_run: bool) -> str:
        submitted_scripts.append(script)
        return str(len(submitted_scripts))

    submitted = submit_workflow(_config(), submit=fake_submit)
    assert submitted == {
        "prepare": "1",
        "pilot": "2",
        "freeze": "3",
        "tune": "4",
        "confirm": "5",
        "analyze": "6",
    }
    assert "#SBATCH --dependency=afterok:4" in submitted_scripts[4]
    assert "#SBATCH --dependency=afterok:5" in submitted_scripts[5]


def test_squashfs_is_staged_only_for_configured_workflow_stages() -> None:
    """Large image copies must not be repeated across downstream array jobs."""
    scripts = {
        job.name: render_sbatch(job, _config(), "config.yaml")
        for job in build_workflow(_config())
    }

    assert f"cp {SQUASHFS_SOURCE}" in scripts["prepare"]
    assert (
        f'BINDS+=("$STAGE_DIR/0.sqfs:{SQUASHFS_MOUNT}:image-src=/")'
        in scripts["prepare"]
    )
    assert f'-B "{PANDA_RAW}:{PANDA_RAW}:ro"' in scripts["prepare"]
    for stage in ("pilot", "freeze", "tune", "confirm", "analyze"):
        assert SQUASHFS_SOURCE not in scripts[stage]
        assert PANDA_RAW not in scripts[stage]


def test_smoke_workflow_uses_test_partition() -> None:
    """The synthetic validation has a one-job test-partition submission path."""
    jobs = build_workflow(_config(), smoke=True)
    assert len(jobs) == 1
    assert jobs[0].partition == "gpu-test"
    assert jobs[0].command == "smoke"
