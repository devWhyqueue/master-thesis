from __future__ import annotations

import argparse
import subprocess
import sys

from scripts.common import EXPERIMENT_ROOT
from scripts.tuning.grid import task_for_array_index


def parse_args() -> argparse.Namespace:
    """Parse validation-tuning run arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(EXPERIMENT_ROOT / "configs" / "default.yaml")
    )
    parser.add_argument(
        "--benchmark", required=True, choices=["patch_feature", "wsi_bag"]
    )
    parser.add_argument("--array-task-id", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run one validation-tuning task selected by array index."""
    args = parse_args()
    variant, seed = task_for_array_index(args.benchmark, args.array_task_id)
    cmd = _trainer_command(
        args, variant.method, variant.variant, variant.params_json, seed
    )
    subprocess.run(cmd, cwd=EXPERIMENT_ROOT, check=True)


def _trainer_command(
    args: argparse.Namespace,
    method: str,
    tuning_id: str,
    tuning_params: str,
    seed: int,
) -> list[str]:
    module = (
        "scripts.train_patch_features"
        if args.benchmark == "patch_feature"
        else "scripts.training.train"
    )
    cmd = [
        sys.executable,
        "-m",
        module,
        "--config",
        args.config,
        "--method",
        method,
        "--seed",
        str(seed),
        "--tuning-id",
        tuning_id,
        "--tuning-params",
        tuning_params,
    ]
    if args.smoke:
        cmd.append("--smoke")
    return cmd


if __name__ == "__main__":
    main()
