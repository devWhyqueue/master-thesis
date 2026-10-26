"""CLI wiring: stage and draw arguments, repeated analyze configs."""

from __future__ import annotations

import pytest

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "exp45_cli", Path(__file__).resolve().parents[1] / "code" / "__main__.py"
)
_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cli)
_parser, main = _cli._parser, _cli.main


def test_fit_requires_stage_and_split() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["--config", "a", "fit", "--split-index", "0"])
    args = _parser().parse_args(["--config", "a", "fit", "--stage", "cpu", "--split-index", "2", "--draw-index", "9"])
    assert (args.stage, args.split_index, args.draw_index) == ("cpu", 2, 9)


def test_cuda_needs_a_draw() -> None:
    with pytest.raises(SystemExit, match="draw-index"):
        main(["--config", "a", "fit", "--stage", "cuda", "--split-index", "0"])


def test_analyze_takes_repeated_configs() -> None:
    args = _parser().parse_args(["--config", "a", "--config", "b", "analyze"])
    assert args.config == ["a", "b"]
