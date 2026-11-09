"""CLI dispatch: stages map to the right fitters, failed gates exit non-zero."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from imbalance_benchmark.common import write_json

_MAIN = Path(__file__).resolve().parents[1] / "code" / "__main__.py"
_CONFIG = str(Path(__file__).resolve().parents[1] / "configs" / "bracs.yaml")


@pytest.fixture
def cli(monkeypatch, tmp_path) -> dict:
    module = runpy.run_path(str(_MAIN), run_name="intended_cli")
    calls: list[tuple] = []
    main = module["main"]
    namespace = main.__globals__
    monkeypatch.setitem(namespace, "fit_lr_shard", lambda c, s, d, stage: calls.append(("lr", s, d, stage)))
    monkeypatch.setitem(namespace, "fit_sgd_draw", lambda c, s, d: calls.append(("sgd", s, d)))

    def gate(passed: bool):
        def run(config):
            path = tmp_path / "gate.json"
            write_json(path, {"passed": passed})
            return path

        return run

    return {"main": main, "calls": calls, "namespace": namespace, "gate": gate}


def test_fit_stages_dispatch(cli: dict) -> None:
    cli["main"](["--config", _CONFIG, "fit", "--stage", "lr", "--split-index", "1", "--draw-index", "4"])
    cli["main"](["--config", _CONFIG, "fit", "--stage", "ce", "--split-index", "0", "--draw-index", "2"])
    cli["main"](["--config", _CONFIG, "fit", "--stage", "sgd", "--split-index", "2"])
    assert cli["calls"][:2] == [("lr", 1, 4, "rest"), ("lr", 0, 2, "ce")]
    assert cli["calls"][2:] == [("sgd", 2, d) for d in range(10)]


def test_shard_index_dispatch(cli: dict) -> None:
    cli["main"](["--config", _CONFIG, "fit", "--stage", "sgd", "--shard-index", "13"])
    cli["main"](["--config", _CONFIG, "fit", "--stage", "lr", "--shard-index", "29"])
    assert cli["calls"] == [("sgd", 1, 3), ("lr", 2, 9, "rest")]
    with pytest.raises(SystemExit):
        cli["main"](["--config", _CONFIG, "fit", "--stage", "lr"])


@pytest.mark.parametrize(("command", "name"), [("preflight", "run_preflight"), ("check-ce", "check_ce")])
def test_failed_gate_exits_nonzero(cli: dict, monkeypatch, command: str, name: str) -> None:
    monkeypatch.setitem(cli["namespace"], name, cli["gate"](False))
    with pytest.raises(SystemExit) as error:
        cli["main"](["--config", _CONFIG, command])
    assert error.value.code == 1
    monkeypatch.setitem(cli["namespace"], name, cli["gate"](True))
    cli["main"](["--config", _CONFIG, command])
