"""Grid flattening and CLI argument-parsing smoke tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from mitigation.grid import stage1_jobs, stage2_jobs

_MAIN_PATH = Path(__file__).resolve().parents[1] / "code" / "__main__.py"
_spec = importlib.util.spec_from_file_location("exp40_main", _MAIN_PATH)
assert _spec is not None and _spec.loader is not None
_main_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_main_module)
_parser = _main_module._parser


def _config() -> dict:
    return {
        "mitigation": {
            "grid": {
                "arms": ["r1", "r100"],
                "stage1": {"ce": [None], "bs": [0.5, 1.0]},
                "stage2": {"posthoc_la": [1.0], "crt": [None]},
            }
        }
    }


def test_stage1_jobs_flattens_arm_method_param_grid() -> None:
    """Every arm is crossed with every configured method/param pair."""
    jobs = stage1_jobs(_config())
    assert len(jobs) == 2 * 3  # 2 arms x (1 ce + 2 bs params)
    assert ("r1", "ce", None) in jobs
    assert ("r100", "bs", 0.5) in jobs


def test_stage2_jobs_flattens_arm_method_param_grid() -> None:
    """Stage-two grid flattens the same way as stage one, over its own method list."""
    jobs = stage2_jobs(_config())
    assert len(jobs) == 2 * 2  # 2 arms x (posthoc_la + crt)
    assert ("r1", "crt", None) in jobs


def test_cli_parser_builds_fit_stage2_and_submit_subcommands() -> None:
    """The argparse tree accepts one representative invocation of each subcommand."""
    parser = _parser()
    fit_args = parser.parse_args(
        ["--config", "c.yaml", "fit", "--shard-index", "0", "--method", "ce", "--arm", "r1"]
    )
    assert fit_args.command == "fit" and fit_args.param is None
    stage2_args = parser.parse_args(["--config", "c.yaml", "stage2", "--shard-index", "0"])
    assert stage2_args.command == "stage2"
    submit_args = parser.parse_args(["--config", "c.yaml", "submit", "--stage", "fit", "--dry-run"])
    assert submit_args.command == "submit" and submit_args.dry_run
