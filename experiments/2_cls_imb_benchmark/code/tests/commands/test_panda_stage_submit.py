from __future__ import annotations

from imbalance_benchmark.hydra.rendering import render_sbatch
from imbalance_benchmark.hydra.workflow import build_workflow
from imbalance_benchmark.datasets.panda_materialize import select_physical_shard
import pandas as pd


def _config() -> dict[str, object]:
    return {
        "dataset": {"name": "panda", "regime": "patch"},
        "materialize_panda": {"shard_count": 48},
        "slurm": {
            "project_root": "/home/example/master-thesis",
            "code_dir": "/home/example/code",
            "output_dir": "/home/example/outputs",
            "container": "/home/example/environment.sif",
            "resources": {
                "materialize": {"partition": "cpu-2d", "gpus": 0, "cpus": 8},
                "extract": {"partition": "gpu-5h", "gpus": 1, "cpus": 8},
                "extract_reduce": {"partition": "cpu-2h", "gpus": 0, "cpus": 2},
            },
            "sharded_squashfs": [
                {
                    "source_template": "/home/space/datasets-sqfs/panda/patch/shard={index}.sqfs",
                    "mount_template": "/home/space/datasets/panda/patch/shard={index}",
                    "stages": ["prepare-extract-shard"],
                }
            ],
        },
    }


def test_panda_stage_only_extract_never_enqueues_prepare_or_tuning() -> None:
    jobs = build_workflow(_config(), stage="extract")

    assert [job.name for job in jobs] == [
        "prepare-extract-shard",
        "prepare-extract-reduce",
    ]
    assert jobs[0].array_size == 48
    assert jobs[1].dependencies == ("prepare-extract-shard",)
    assert all("tune" not in job.name for job in jobs)


def test_panda_stage_only_pilot_can_run_one_split() -> None:
    jobs = build_workflow(_config(), stage="pilot", split_index=0)

    assert [job.name for job in jobs] == ["pilot"]
    assert jobs[0].array_splits == (0,)


def test_panda_extract_stages_only_its_array_shard() -> None:
    extract = build_workflow(_config(), stage="extract")[0]
    script = render_sbatch(extract, _config(), "config.yaml")

    assert 'cp "/home/space/datasets-sqfs/panda/patch/shard=${SLURM_ARRAY_TASK_ID}.sqfs"' in script
    assert '"$STAGE_DIR/0.sqfs:/home/space/datasets/panda/patch/shard=${SLURM_ARRAY_TASK_ID}:image-src=/"' in script


def test_panda_extract_selects_the_physical_shard_it_stages() -> None:
    frame = pd.DataFrame(
        {"slide_id": ["a", "b"], "shard_index": [0, 1], "image_path": ["a.jpg", "b.jpg"]}
    )

    result = select_physical_shard(frame, {"name": "panda"}, 1, 48)

    assert result["slide_id"].tolist() == ["b"]
