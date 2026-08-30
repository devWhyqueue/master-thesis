from __future__ import annotations

from pathlib import Path
import threading

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


def test_record_attempted_loses_no_index_under_concurrent_writers(
    tmp_path: Path,
) -> None:
    """Regression: a lost read-modify-write under concurrent multi-condition
    load used to undercount real progress, silently stalling a round well
    short of full coverage."""
    threads = [
        threading.Thread(target=record_attempted, args=(tmp_path, "base-natural", [i]))
        for i in range(50)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert unattempted(tmp_path, "base-natural", list(range(50))) == []
