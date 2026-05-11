from __future__ import annotations

import logging
import argparse
import subprocess
import sys

from scripts.common import EXPERIMENT_ROOT, load_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for end-to-end pipeline execution."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(EXPERIMENT_ROOT / "configs" / "default.yaml")
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def run(args: list[str]) -> None:
    """Run one subprocess command in the experiment root."""
    logger.info(" ".join(args))
    subprocess.run(args, cwd=EXPERIMENT_ROOT, check=True)


def _smoke_overrides(
    methods: list[str], seeds: list[int], smoke: bool
) -> tuple[list[str], list[int]]:
    """Adjust seed/method selection for smoke mode."""
    if not smoke:
        return methods, seeds
    selected_seeds = seeds[:1]
    selected_methods = [method for method in ["ce", "knn"] if method in methods]
    return selected_methods, selected_seeds


def _run_split_jobs(python: str, config_path: str, seeds: list[int]) -> None:
    """Run split generation for all seeds."""
    for seed in seeds:
        run(
            [
                python,
                "-m",
                "scripts.prep.splits",
                "--config",
                config_path,
                "--seed",
                str(seed),
            ]
        )


def _run_train_jobs(
    python: str, config_path: str, methods: list[str], seeds: list[int], smoke: bool
) -> None:
    """Run model training jobs for all method/seed pairs."""
    for method in methods:
        for seed in seeds:
            command = [
                python,
                "-m",
                "scripts.training.train",
                "--config",
                config_path,
                "--method",
                method,
                "--seed",
                str(seed),
            ]
            if smoke:
                command.append("--smoke")
            run(command)


def main() -> None:
    """Run the full experiment workflow in sequence."""
    args = parse_args()
    config = load_config(args.config)
    python = sys.executable
    seeds = config["training"]["seeds"]
    methods = config["methods"]
    methods, seeds = _smoke_overrides(methods, seeds, args.smoke)
    first_seed = str(seeds[0])
    run([python, "-m", "scripts.prep.check_env"])
    run([python, "-m", "scripts.prep.manifest", "--config", args.config])
    _run_split_jobs(python, args.config, seeds)
    run(
        [
            python,
            "-m",
            "scripts.prep.explore",
            "--config",
            args.config,
            "--seed",
            first_seed,
        ]
    )
    _run_train_jobs(python, args.config, methods, seeds, args.smoke)
    run([python, "-m", "scripts.report.aggregate", "--config", args.config])
    run([python, "-m", "scripts.report.figures", "--config", args.config])


if __name__ == "__main__":
    main()
