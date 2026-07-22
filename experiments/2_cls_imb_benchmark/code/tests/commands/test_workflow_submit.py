from __future__ import annotations



from imbalance_benchmark.hydra.workflow import (
    build_workflow,
    render_sbatch,
    submit_workflow,
)

PANDA_RAW = "/home/space/datasets/panda/raw"
SQUASHFS_SOURCE = "/home/space/datasets-sqfs/panda-native-tiles-20x-256.sqfs"
SQUASHFS_MOUNT = "/home/space/datasets/panda/native_tiles_20x_256"
GENERATED_TILES = "/tmp/bracs_roi_tiles"
GENERATED_SQUASHFS = "/home/example/outputs/bracs/roi_tiles.sqfs"

def _config() -> dict[str, object]:
    return {
        "dataset": {"root": PANDA_RAW},
        "slurm": {
            "project_root": "/home/example/master-thesis",
            "code_dir": "/home/example/master-thesis/experiments/2_cls_imb_benchmark/code",
            "container": "/home/example/environment.sif",
            "test_partition": "gpu-test",
            "tune_natural_observations_per_candidate": 6,
            "tune_shards_per_task": 4,
            "resources": {
                "tune_natural": {"memory": "32G"},
                "tune_controlled": {"memory": "32G"},
                "confirm_natural": {"partition": "gpu-5h"},
                "confirm_controlled": {"partition": "gpu-2h"},
            },
            "squashfs": [
                {
                    "source": SQUASHFS_SOURCE,
                    "mount": SQUASHFS_MOUNT,
                    "stages": ["prepare"],
                }
            ],
        }
    }

def test_workflow_has_resumable_sharded_tuning_dag() -> None:
    """Tuning fans out by candidate and reduces before confirmation."""
    jobs = build_workflow(_config())
    assert [job.name for job in jobs] == [
        "prepare",
        "pilot",
        "freeze",
        "tune-base-natural",
        "tune-base-controlled",
        "tune-base-reduce",
        "tune-dependent-posthoc-natural",
        "tune-dependent-crt-natural",
        "tune-dependent-controlled",
        "tune-final-reduce",
        "confirm-natural",
        "confirm-controlled",
        "analyze",
    ]
    assert jobs[-3].dependencies == ("tune-final-reduce",)
    assert jobs[-2].dependencies == ("tune-final-reduce",)
    assert jobs[-1].dependencies == ("confirm-natural", "confirm-controlled")
    assert jobs[3].array_size == 198
    assert "--observations-per-candidate 6" in jobs[3].command
    assert "--shards-per-task 4" in jobs[3].command
    assert "--bundle-by-observation" in jobs[3].command
    assert jobs[3].memory == "32G"
    assert jobs[4].array_size == 99
    assert jobs[6].array_size == 0
    assert "--shard-index 0" in jobs[6].command
    assert jobs[7].array_size == 6
    assert "--observations-per-candidate 6" in jobs[7].command
    assert "--shard-offset 1" in jobs[7].command
    assert "--bundle-by-observation" in jobs[7].command
    assert jobs[8].array_size == 4
    natural_script = render_sbatch(jobs[3], _config(), "config.yaml")
    assert "#SBATCH --mem=32G" in natural_script

def test_confirm_shards_naturally_and_controlled_across_two_partitions() -> None:
    """Confirmation no longer shares one two-day array across every condition."""
    jobs = build_workflow(_config())
    confirm_natural = next(j for j in jobs if j.name == "confirm-natural")
    confirm_controlled = next(j for j in jobs if j.name == "confirm-controlled")

    # patch roster minus post-hoc = 10 methods; 3 splits x 5 seeds each.
    assert confirm_natural.array_size == 3 * 10 * 5
    assert confirm_controlled.array_size == 3 * 3 * 10 * 5
    assert confirm_natural.partition == "gpu-5h"
    assert confirm_controlled.partition == "gpu-2h"
    assert confirm_natural.partition != "gpu-2d"
    assert confirm_controlled.partition != "gpu-2d"

    natural_script = render_sbatch(confirm_natural, _config(), "config.yaml")
    controlled_script = render_sbatch(confirm_controlled, _config(), "config.yaml")
    assert f"#SBATCH --array=0-{3 * 10 * 5 - 1}" in natural_script
    assert f"#SBATCH --array=0-{3 * 3 * 10 * 5 - 1}" in controlled_script
    assert "confirm-shard --group natural" in natural_script
    assert "confirm-shard --group controlled" in controlled_script
    assert '--shard-index "$SLURM_ARRAY_TASK_ID"' in natural_script
    assert '--shard-index "$SLURM_ARRAY_TASK_ID"' in controlled_script

def test_submit_links_actual_job_ids() -> None:
    """Submission turns stage names into the preceding scheduler job IDs."""
    submitted_scripts: list[str] = []

    def fake_submit(script: str, dry_run: bool) -> str:
        submitted_scripts.append(script)
        return str(len(submitted_scripts))

    submitted = submit_workflow(_config(), submit=fake_submit)
    assert submitted["prepare"] == "1"
    assert submitted["analyze"] == str(len(submitted_scripts))
    assert "#SBATCH --dependency=afterok:4:5" in submitted_scripts[5]
    assert "#SBATCH --dependency=afterok:7:8:9" in submitted_scripts[9]
    assert "#SBATCH --dependency=afterok:10" in submitted_scripts[10]
    assert "#SBATCH --dependency=afterok:10" in submitted_scripts[11]
    assert "#SBATCH --dependency=afterok:11:12" in submitted_scripts[12]


def test_resume_tuning_skips_completed_setup() -> None:
    jobs = build_workflow(_config(), resume_tuning=True)

    assert jobs[0].name == "tune-base-natural"
    assert all(job.name not in {"prepare", "pilot", "freeze"} for job in jobs)

def test_squashfs_is_staged_only_for_configured_workflow_stages() -> None:
    """Large image copies must not be repeated across downstream array jobs."""
    scripts = {
        job.name: render_sbatch(job, _config(), "config.yaml")
        for job in build_workflow(_config())
    }

    assert f"cp {SQUASHFS_SOURCE}" in scripts["prepare"]
    assert (
        f'BINDS+=("-B" "$STAGE_DIR/0.sqfs:{SQUASHFS_MOUNT}:image-src=/")'
        in scripts["prepare"]
    )
    assert f'-B "{PANDA_RAW}:{PANDA_RAW}:ro"' in scripts["prepare"]
    for stage in (
        "pilot",
        "freeze",
        "tune-base-natural",
        "tune-base-controlled",
        "confirm-natural",
        "confirm-controlled",
        "analyze",
    ):
        assert SQUASHFS_SOURCE not in scripts[stage]
        assert PANDA_RAW not in scripts[stage]
        # Shared datasets under /home/space must stay reachable at every stage:
        # patch-regime features are read by absolute path during tune/confirm.
        assert '-B "/home/space:/home/space:ro"' in scripts[stage]

def test_staged_binds_get_their_own_dash_b_flag() -> None:
    """A ``BINDS`` array entry with no ``-B`` flag is passed as a bare apptainer
    positional argument instead of a bind spec, so apptainer tries to open it
    as the container image and fails."""
    scripts = {
        job.name: render_sbatch(job, _config(), "config.yaml")
        for job in build_workflow(_config())
    }
    assert 'BINDS+=("-B" "$STAGE_DIR/0.sqfs' in scripts["prepare"]

def test_prepare_packs_generated_tiles_and_reuses_the_squashfs() -> None:
    """BRACS tiles stay node-local and persist as one reusable image."""
    config = _config()
    config["dataset"].update(
        tile_root=GENERATED_TILES,
        tile_squashfs=GENERATED_SQUASHFS,
    )

    scripts = {
        job.name: render_sbatch(job, config, "config.yaml")
        for job in build_workflow(config)
    }

    prepare = scripts["prepare"]
    assert (
        f'BINDS+=("-B" {GENERATED_SQUASHFS}:{GENERATED_TILES}:image-src=/)'
        in prepare.replace("'", "")
    )
    assert f"squash-dataset {GENERATED_TILES} {GENERATED_SQUASHFS}.partial" in prepare
    assert f"mv {GENERATED_SQUASHFS}.partial {GENERATED_SQUASHFS}" in prepare
    for stage in (
        "pilot",
        "freeze",
        "tune-base-natural",
        "tune-base-controlled",
        "confirm-natural",
        "confirm-controlled",
        "analyze",
    ):
        assert GENERATED_SQUASHFS not in scripts[stage]

def test_time_limit_directive_is_omitted_unless_explicitly_configured() -> None:
    """Partitions already cap wall time (e.g. cpu-2h -> 2h); an explicit
    --time should only appear when a stage config asks for less than that."""
    scripts = {
        job.name: render_sbatch(job, _config(), "config.yaml")
        for job in build_workflow(_config())
    }
    for stage, script in scripts.items():
        assert "#SBATCH --time=" not in script, stage

    config = _config()
    config["slurm"]["resources"] = {"freeze": {"time": "00:30:00"}}
    freeze_job = next(j for j in build_workflow(config) if j.name == "freeze")
    script = render_sbatch(freeze_job, config, "config.yaml")
    assert "#SBATCH --time=00:30:00" in script

def test_smoke_workflow_uses_test_partition() -> None:
    """The synthetic validation has a one-job test-partition submission path."""
    jobs = build_workflow(_config(), smoke=True)
    assert len(jobs) == 1
    assert jobs[0].partition == "gpu-test"
    assert jobs[0].command == "smoke"
