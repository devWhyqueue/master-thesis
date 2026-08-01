from __future__ import annotations

from pathlib import Path

from imbalance_benchmark.common import load_config
from imbalance_benchmark.hydra.workflow import build_workflow, render_sbatch


def test_camelyon_patch_prepare_is_one_staged_multi_gpu_job() -> None:
    config = load_config(
        Path(__file__).resolve().parents[3] / "configs" / "camelyon16_patch.yaml"
    )
    prepare = next(job for job in build_workflow(config) if job.name == "prepare")
    script = render_sbatch(prepare, config)

    assert (prepare.partition, prepare.gpus, prepare.cpus, prepare.memory) == (
        "gpu-2h",
        4,
        32,
        "64G",
    )
    assert "#SBATCH --array=" not in script
    assert script.count("cp /home/space/datasets-sqfs/camelyon16-patches-20x.sqfs") == 1
