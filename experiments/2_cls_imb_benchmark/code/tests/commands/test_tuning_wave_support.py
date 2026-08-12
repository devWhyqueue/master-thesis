from __future__ import annotations

from pathlib import Path

from imbalance_benchmark.commands.tuning.wave_support import (
    record_attempted,
    unattempted,
)


def test_unattempted_returns_everything_when_nothing_recorded(tmp_path: Path) -> None:
    assert unattempted(tmp_path, "base-controlled", [0, 1, 2]) == [0, 1, 2]


def test_unattempted_excludes_previously_recorded_indices(tmp_path: Path) -> None:
    record_attempted(tmp_path, "base-controlled", [0, 2])

    assert unattempted(tmp_path, "base-controlled", [0, 1, 2, 3]) == [1, 3]


def test_record_attempted_accumulates_across_calls(tmp_path: Path) -> None:
    record_attempted(tmp_path, "base-controlled", [0])
    record_attempted(tmp_path, "base-controlled", [1])

    assert unattempted(tmp_path, "base-controlled", [0, 1, 2]) == [2]


def test_scopes_do_not_leak_into_each_other(tmp_path: Path) -> None:
    record_attempted(tmp_path, "base-natural", [0, 1])

    assert unattempted(tmp_path, "base-controlled", [0, 1]) == [0, 1]
