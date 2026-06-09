from __future__ import annotations

import argparse
import subprocess
import sys

from scripts.common import EXPERIMENT_ROOT, load_config


def parse_args() -> argparse.Namespace:
    """Parse end-to-end pipeline arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(EXPERIMENT_ROOT / "configs" / "default.yaml")
    )
    parser.add_argument(
        "--benchmark", choices=["patch", "wsi_bag", "all"], default="all"
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _run(args: list[str]) -> None:
    subprocess.run(args, cwd=EXPERIMENT_ROOT, check=True)


def _prepare(python: str, config_path: str, seeds: list[int]) -> None:
    _run([python, "-m", "scripts.data.prep.check_env"])
    _run([python, "-m", "scripts.data.prep.manifest.feature", "--config", config_path])
    for seed in seeds:
        _run(
            [
                python,
                "-m",
                "scripts.data.prep.manifest.splits",
                "--config",
                config_path,
                "--seed",
                str(seed),
            ]
        )
        _run(
            [
                python,
                "-m",
                "scripts.data.prep.manifest.patch",
                "--config",
                config_path,
                "--seed",
                str(seed),
            ]
        )
    _run(
        [
            python,
            "-m",
            "scripts.data.prep.explore",
            "--config",
            config_path,
            "--seed",
            str(seeds[0]),
        ]
    )


def _train_patch(python: str, config_path: str, config: dict, smoke: bool) -> None:
    seeds = _seeds(config["patch_training"]["seeds"], smoke)
    for method in config["patch_methods"]:
        for seed in seeds:
            cmd = [
                python,
                "-m",
                "scripts.modeling.patch.train",
                "--config",
                config_path,
                "--method",
                method,
                "--seed",
                str(seed),
            ]
            if smoke:
                cmd.append("--smoke")
            _run(cmd)
    _run(
        [
            python,
            "-m",
            "scripts.analysis.report.build_db",
            "--config",
            config_path,
        ]
    )
    _run(
        [
            python,
            "-m",
            "scripts.analysis.report.aggregate",
            "--config",
            config_path,
            "--benchmark",
            "patch",
        ]
    )
    _run(
        [
            python,
            "-m",
            "scripts.analysis.report.figures",
            "--config",
            config_path,
            "--benchmark",
            "patch",
        ]
    )


def _train_wsi(python: str, config_path: str, config: dict, smoke: bool) -> None:
    seeds = _seeds(config["wsi_training"]["seeds"], smoke)
    for method in config["wsi_bag_methods"]:
        for seed in seeds:
            cmd = [
                python,
                "-m",
                "scripts.modeling.training.train",
                "--config",
                config_path,
                "--method",
                method,
                "--seed",
                str(seed),
            ]
            if smoke:
                cmd.append("--smoke")
            _run(cmd)
    _run(
        [
            python,
            "-m",
            "scripts.analysis.report.build_db",
            "--config",
            config_path,
        ]
    )
    _run(
        [
            python,
            "-m",
            "scripts.analysis.report.aggregate",
            "--config",
            config_path,
            "--benchmark",
            "wsi_bag",
        ]
    )
    _run(
        [
            python,
            "-m",
            "scripts.analysis.report.figures",
            "--config",
            config_path,
            "--benchmark",
            "wsi_bag",
        ]
    )


def main() -> None:
    """Run shared preparation and the requested benchmark pipelines."""
    args = parse_args()
    config = load_config(args.config)
    python = sys.executable
    seeds = _seeds(
        sorted(
            set(config["patch_training"]["seeds"])
            | set(config["wsi_training"]["seeds"])
        ),
        args.smoke,
    )
    _prepare(python, args.config, seeds)
    if args.benchmark in {"patch", "all"}:
        _train_patch(python, args.config, config, args.smoke)
    if args.benchmark in {"wsi_bag", "all"}:
        _train_wsi(python, args.config, config, args.smoke)


def _seeds(seeds: list[int], smoke: bool) -> list[int]:
    """Return configured seeds, restricted to one seed for smoke runs."""
    selected = sorted(seeds)
    return selected[:1] if smoke else selected


if __name__ == "__main__":
    main()
