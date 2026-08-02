from __future__ import annotations

from pathlib import Path

from imbalance_benchmark.hydra.workflow import (
    build_workflow,
    render_sbatch,
    submit_workflow,
)
from imbalance_benchmark.common import load_config

PANDA_RAW = "/home/space/datasets/panda/raw"
SQUASHFS_SOURCE = "/home/space/datasets-sqfs/panda-native-tiles-20x-256.sqfs"
SQUASHFS_MOUNT = "/home/space/datasets/panda/native_tiles_20x_256"
GENERATED_TILES = "/tmp/bracs_roi_tiles"
GENERATED_SQUASHFS = "/home/example/outputs/bracs/roi_tiles.sqfs"
CAMELYON_CONFIG = (
    Path(__file__).resolve().parents[3] / "configs" / "camelyon16_patch.yaml"
)

def _config() -> dict[str, object]:
    return {
        "dataset": {"root": PANDA_RAW},
        "slurm": {
            "project_root": "/home/example/master-thesis",
            "code_dir": "/home/example/master-thesis/experiments/2_cls_imb_benchmark/code",
            "container": "/home/example/environment.sif",
            "test_partition": "gpu-test",
            "tune_natural_observations_per_candidate": 6,
            "tune_shards_per_task": 2,
            "confirm_natural_shards_per_task": 5,
            "confirm_controlled_shards_per_task": 5,
            "max_array_concurrency": 8,
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
    """Base tuning fans out by candidate, reduces, then hands off to tune-decide.

    Dependent-phase (crt/post-hoc) round-0 jobs are no longer statically
    chained here: CE may still need further adaptive rounds after this
    reduce, so tune-decide submits them itself only once CE resolves.
    """
    jobs = build_workflow(_config())
    assert [job.name for job in jobs] == [
        "prepare",
        "pilot",
        "freeze",
        "tune-base-natural",
        "tune-base-controlled",
        "tune-base-reduce",
        "tune-decide-base-natural",
        "tune-decide-base-balanced",
        "tune-decide-base-moderate",
        "tune-decide-base-severe",
    ]
    assert jobs[5].dependencies == ("tune-base-natural", "tune-base-controlled")
    assert jobs[6].dependencies == ("tune-base-reduce",)
    assert jobs[6].command == "tune-decide --phase base --condition natural --round 0"
    assert jobs[9].command == "tune-decide --phase base --condition severe --round 0"
    assert jobs[3].array_size == 396
    assert "--observations-per-candidate 6" in jobs[3].command
    assert "--shards-per-task 2" in jobs[3].command
    assert "--bundle-by-observation" in jobs[3].command
    assert jobs[3].memory == "32G"
    assert jobs[4].array_size == 198
    natural_script = render_sbatch(jobs[3], _config(), "config.yaml")
    assert "#SBATCH --mem=32G" in natural_script


def test_decide_jobs_run_on_host_not_in_apptainer() -> None:
    """tune-decide shells out to sbatch/squeue itself - the container has no
    SLURM client, so it must run on the host (via uv) instead, while still
    keeping its #SBATCH directives (partition, dependency) intact."""
    jobs = build_workflow(_config())
    decide = next(j for j in jobs if j.name == "tune-decide-base-natural")
    script = render_sbatch(decide, _config(), "config.yaml")
    assert "apptainer" not in script
    assert "uv run python" in script
    assert "#SBATCH --partition=cpu-2h" in script
    assert "#SBATCH --dependency=afterok:tune-base-reduce" in script


def test_dependent_round_zero_jobs_match_the_frozen_shapes() -> None:
    """Dependent-phase round-0 jobs (submitted by tune-decide) keep today's sizes."""
    from imbalance_benchmark.hydra.dependent_jobs import dependent_round_zero_jobs

    posthoc, crt, controlled = dependent_round_zero_jobs(_config(), is_mil=False)
    assert posthoc.array_size == 0
    assert "--shard-index 0" in posthoc.command
    assert crt.array_size == 12
    assert "--observations-per-candidate 6" in crt.command
    assert "--shard-offset 1" in crt.command
    assert "--bundle-by-observation" in crt.command
    assert controlled.array_size == 8


def test_confirm_only_builds_just_confirm_and_analyze() -> None:
    """confirm_only submits the later stage separately once tuning is locked."""
    jobs = build_workflow(_config(), confirm_only=True)
    assert [job.name for job in jobs] == [
        "confirm-natural",
        "confirm-controlled",
        "analyze",
        "analyze-combine",
    ]
    assert jobs[0].dependencies == ()
    assert jobs[1].dependencies == ()
    assert jobs[2].dependencies == ("confirm-natural", "confirm-controlled")
    assert jobs[2].array_splits == (0, 1, 2)
    assert jobs[3].dependencies == ("analyze",)

def test_confirm_shards_naturally_and_controlled_across_two_partitions() -> None:
    """Confirmation no longer shares one two-day array across every condition."""
    jobs = build_workflow(_config(), confirm_only=True)
    confirm_natural = next(j for j in jobs if j.name == "confirm-natural")
    confirm_controlled = next(j for j in jobs if j.name == "confirm-controlled")

    # patch roster minus post-hoc = 10 methods; 3 splits x 5 seeds each.
    assert confirm_natural.array_size == 3 * 10
    assert confirm_controlled.array_size == 3 * 3 * 10
    assert confirm_natural.partition == "gpu-5h"
    assert confirm_controlled.partition == "gpu-2h"
    assert confirm_natural.partition != "gpu-2d"
    assert confirm_controlled.partition != "gpu-2d"

    natural_script = render_sbatch(confirm_natural, _config(), "config.yaml")
    controlled_script = render_sbatch(confirm_controlled, _config(), "config.yaml")
    assert f"#SBATCH --array=0-{3 * 10 - 1}%8" in natural_script
    assert f"#SBATCH --array=0-{3 * 3 * 10 - 1}%8" in controlled_script
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
    assert submitted["tune-decide-base-severe"] == str(len(submitted_scripts))
    assert "#SBATCH --dependency=afterok:4:5" in submitted_scripts[5]
    assert "#SBATCH --dependency=afterok:6" in submitted_scripts[6]


def test_confirm_only_links_actual_job_ids() -> None:
    """A separate confirm_only submission chains confirm into analyze on its own."""
    submitted_scripts: list[str] = []

    def fake_submit(script: str, dry_run: bool) -> str:
        submitted_scripts.append(script)
        return str(len(submitted_scripts))

    submitted = submit_workflow(_config(), confirm_only=True, submit=fake_submit)
    assert submitted["confirm-natural"] == "1"
    assert submitted["confirm-controlled"] == "2"
    assert "#SBATCH --dependency=" not in submitted_scripts[0]
    assert "#SBATCH --dependency=afterok:1:2" in submitted_scripts[2]
    assert submitted["analyze"] == "3"
    assert "#SBATCH --dependency=afterok:3" in submitted_scripts[3]
    assert submitted["analyze-combine"] == "4"


def test_resume_tuning_skips_completed_setup(monkeypatch) -> None:
    monkeypatch.setattr(
        "imbalance_benchmark.hydra.workflow.resume_plan",
        lambda *_: type("Plan", (), {"natural_indices": (2, 5), "controlled_indices": (0,)})(),
    )
    jobs = build_workflow(_config(), resume_tuning=True)

    assert jobs[0].name == "tune-base-natural"
    assert jobs[0].array_indices == (2, 5)
    assert all(job.name not in {"prepare", "pilot", "freeze"} for job in jobs)


def test_resume_omits_only_a_fingerprint_valid_base_natural_array(monkeypatch) -> None:
    config = _config()
    monkeypatch.setattr(
        "imbalance_benchmark.hydra.workflow.resume_plan",
        lambda *_: type("Plan", (), {"natural_indices": (), "controlled_indices": (1, 3)})(),
    )

    jobs = build_workflow(config, resume_tuning=True)
    names = [job.name for job in jobs]
    controlled = next(job for job in jobs if job.name == "tune-base-controlled")

    assert "tune-base-natural" not in names
    assert controlled.array_indices == (1, 3)
    assert "#SBATCH --array=1,3%8" in render_sbatch(controlled, config, "config.yaml")
    assert next(job for job in jobs if job.name == "tune-base-reduce").dependencies == (
        "tune-base-controlled",
    )

def test_squashfs_is_staged_only_for_configured_workflow_stages() -> None:
    """Large image copies must not be repeated across downstream array jobs."""
    scripts = {
        job.name: render_sbatch(job, _config(), "config.yaml")
        for job in [
            *build_workflow(_config()),
            *build_workflow(_config(), confirm_only=True),
        ]
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
        for job in [*build_workflow(config), *build_workflow(config, confirm_only=True)]
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


def test_camelyon_natural_jobs_use_three_shards_on_40gb_gpu_5h(monkeypatch) -> None:
    config = load_config(CAMELYON_CONFIG)
    base_natural, base_controlled = [
        job
        for job in build_workflow(config)
        if job.name in {"tune-base-natural", "tune-base-controlled"}
    ]
    confirm_natural, confirm_controlled = [
        job
        for job in build_workflow(config, confirm_only=True)
        if job.name in {"confirm-natural", "confirm-controlled"}
    ]

    assert "--shards-per-task 3" in base_natural.command
    assert "--shards-per-task 4" in base_controlled.command
    assert base_natural.partition == confirm_natural.partition == "gpu-5h"
    assert "40gb" in base_natural.constraint
    assert "40gb" in confirm_natural.constraint
    assert "40gb" in confirm_controlled.constraint
    for resource in (
        "tune_natural",
        "tune_post_hoc_natural",
        "confirm_natural",
    ):
        assert config["slurm"]["resources"][resource]["partition"] == "gpu-5h"
        assert "40gb" in config["slurm"]["resources"][resource]["constraint"]
    assert "%35" in render_sbatch(base_natural, config)
    assert "%35" in render_sbatch(confirm_natural, config)

    from imbalance_benchmark.hydra.dependent_jobs import dependent_round_zero_jobs

    _, dependent_natural, dependent_controlled = dependent_round_zero_jobs(
        config, is_mil=False
    )
    assert "--shards-per-task 3" in dependent_natural.command
    assert "--shards-per-task 4" in dependent_controlled.command

    monkeypatch.setattr(
        "imbalance_benchmark.hydra.workflow.resume_plan",
        lambda *_: type(
            "Plan",
            (),
            {
                "natural_indices": tuple(range(base_natural.array_size)),
                "controlled_indices": tuple(range(base_controlled.array_size)),
            },
        )(),
    )
    resumed = build_workflow(config, resume_tuning=True)
    natural = next(job for job in resumed if job.name == "tune-base-natural")
    assert len(natural.array_indices) == len(set(natural.array_indices))
    assert natural.array_indices == tuple(range(base_natural.array_size))


def test_camelyon_resume_natural_indices_cover_three_shard_bundles_once(
    monkeypatch,
) -> None:
    from imbalance_benchmark.hydra import resume
    from imbalance_benchmark.modeling.workflows.tuning.tuning_schedule import (
        bundled_observation_array_size,
        candidate_array_size,
        phase_methods,
    )

    config = load_config(CAMELYON_CONFIG)
    methods = phase_methods(False, "base")
    seen: list[int] = []
    monkeypatch.setattr(
        resume,
        "_natural_bundle_complete",
        lambda index, *_args: seen.append(index) or False,
    )

    pending = resume._pending_natural(
        config, {}, methods, False, {}, [], 3
    )

    expected = bundled_observation_array_size(
        candidate_array_size(methods), 6, 3
    )
    assert pending == tuple(range(expected))
    assert seen == list(range(expected))
