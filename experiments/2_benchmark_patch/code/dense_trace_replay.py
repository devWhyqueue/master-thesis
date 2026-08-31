"""Offline replay for the dense-trace gate (plan items 3/4 in
``after-a-first-run-linear-wave.md``): decide whether tuning-budget truncation
or a log-spaced checkpoint schedule would change which candidate gets
selected, using the traces ``dense_trace_gate.py`` wrote.

Pure analysis, no GPU/cluster access needed -- run against a local copy of
the trace directory (``dense_trace_gate.py``'s ``--out``, default
``<dataset root>/diagnostics/dense_trace``).

Usage:
    python dense_trace_replay.py --trace-dir /path/to/dense_trace
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TRUNCATION_FRACTIONS = (1.0, 0.75, 0.5, 0.40)
TARGET_CHECKPOINTS = 170


def load_traces(trace_dir: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Group every candidate's trace record by (condition, method).

    ``dense_trace_gate.py`` names each file ``{condition}__{method}__{index}.json``;
    the condition is read from the filename since the record itself omits it.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(trace_dir.glob("*.json")):
        condition = path.stem.split("__", 1)[0]
        record = json.loads(path.read_text())
        groups[(condition, record["method"])].append(record)
    return dict(groups)


def _best_by(
    trace: list[dict[str, float]], cutoff_step: int
) -> tuple[float, float, float] | None:
    """BA/F1/NLL of the tie-break-best traced entry at or before ``cutoff_step``."""
    eligible = [entry for entry in trace if entry["step"] <= cutoff_step]
    if not eligible:
        return None
    best = eligible[0]
    for entry in eligible[1:]:
        if _wins(entry, best):
            best = entry
    return best["balanced_accuracy"], best["macro_f1"], best["nll"]


def _key(entry: dict[str, float]) -> tuple[float, float, float]:
    return entry["balanced_accuracy"], entry["macro_f1"], entry["nll"]


def _better(
    key: tuple[float, float, float], best_key: tuple[float, float, float] | None
) -> bool:
    """Mirror ``evaluation.checkpoint_step``'s BA -> F1 -> NLL tie-break."""
    if best_key is None:
        return True
    ba, f1, nll = key
    b_ba, b_f1, b_nll = best_key
    return ba > b_ba or (
        abs(ba - b_ba) < 1e-6 and (f1 > b_f1 or (abs(f1 - b_f1) < 1e-6 and nll < b_nll))
    )


def _wins(candidate: dict[str, float], incumbent: dict[str, float]) -> bool:
    return _better(_key(candidate), _key(incumbent))


def _candidate_budget(record: dict[str, Any]) -> int:
    return max(entry["step"] for entry in record["trace"])


def truncated_selection(
    records: list[dict[str, Any]], fraction: float
) -> tuple[int, float]:
    """Return (candidate_index, its BA) selected among ``records`` at this truncation."""
    best_index, best_key = None, None
    for record in records:
        cutoff = max(1, round(_candidate_budget(record) * fraction))
        key = _best_by(record["trace"], cutoff)
        if key is not None and _better(key, best_key):
            best_key, best_index = key, record["candidate_index"]
    assert best_index is not None and best_key is not None
    return best_index, best_key[0]


def truncation_gate(
    groups: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """For every (condition, method), report each truncation's agreement with 100%."""
    rows = []
    for (condition, method), records in groups.items():
        if len(records) < 2:
            continue  # nothing to select among
        full_index, full_ba = truncated_selection(records, 1.0)
        for fraction in TRUNCATION_FRACTIONS:
            index, ba = truncated_selection(records, fraction)
            rows.append(
                {
                    "condition": condition,
                    "method": method,
                    "fraction": fraction,
                    "changed_selection": index != full_index,
                    "ba_loss": full_ba - ba,
                }
            )
    return rows


def log_spaced_steps(max_steps: int, count: int) -> list[int]:
    """Roughly log-spaced checkpoint steps in [1, max_steps], dense early, sparse late."""
    count = min(count, max_steps)
    positions = [
        1 - math.log1p(count - 1 - i) / math.log1p(count - 1) for i in range(count)
    ]
    steps = sorted({max(1, round(p * max_steps)) for p in positions})
    return steps or [max_steps]


def uniform_steps(max_steps: int, count: int) -> list[int]:
    """Evenly spaced checkpoint steps in [1, max_steps], the current production cadence."""
    interval = max(1, round(max_steps / count))
    steps = list(range(interval, max_steps + 1, interval))
    return steps if steps and steps[-1] == max_steps else [*steps, max_steps]


def _nearest_traced_step(trace: list[dict[str, float]], target_step: int) -> int:
    return int(
        min(
            (entry["step"] for entry in trace),
            key=lambda step: abs(step - target_step),
        )
    )


def schedule_selection(
    records: list[dict[str, Any]], schedule: dict[int, list[int]]
) -> tuple[int, float]:
    """Selection induced by only observing each candidate's schedule steps."""
    best_index, best_key = None, None
    for record in records:
        allowed = set(schedule[record["candidate_index"]])
        visible = [entry for entry in record["trace"] if entry["step"] in allowed]
        if not visible:
            continue
        best = visible[0]
        for entry in visible[1:]:
            if _wins(entry, best):
                best = entry
        key = _key(best)
        if _better(key, best_key):
            best_key, best_index = key, record["candidate_index"]
    assert best_index is not None and best_key is not None
    return best_index, best_key[0]


def _build_schedule(
    records: list[dict[str, Any]], builder: Any
) -> dict[int, list[int]]:
    return {
        record["candidate_index"]: [
            _nearest_traced_step(record["trace"], step)
            for step in builder(_candidate_budget(record), TARGET_CHECKPOINTS)
        ]
        for record in records
    }


def schedule_gate(
    groups: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Item 3: does a log-spaced TARGET_CHECKPOINTS grid pick worse than uniform?"""
    rows = []
    for (condition, method), records in groups.items():
        if len(records) < 2:
            continue
        full_index, full_ba = truncated_selection(records, 1.0)
        for name, builder in (
            ("uniform", uniform_steps),
            ("log_spaced", log_spaced_steps),
        ):
            index, ba = schedule_selection(records, _build_schedule(records, builder))
            rows.append(
                {
                    "condition": condition,
                    "method": method,
                    "schedule": name,
                    "changed_selection": index != full_index,
                    "ba_loss": full_ba - ba,
                }
            )
    return rows


def _summarize(rows: list[dict[str, Any]], group_key: str) -> None:
    by_group: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[row[group_key]].append(row)
    for group, group_rows in sorted(by_group.items(), key=str):
        changed = sum(row["changed_selection"] for row in group_rows)
        mean_loss = sum(row["ba_loss"] for row in group_rows) / len(group_rows)
        logger.info(
            "%s=%s: changed %d/%d (%.1f%%), mean BA loss %.4f",
            group_key,
            group,
            changed,
            len(group_rows),
            100 * changed / len(group_rows),
            mean_loss,
        )


def main() -> None:
    """Load traces, replay both gates, and log a per-fraction/schedule summary."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    groups = load_traces(trace_dir)
    logger.info("loaded %d (condition, method) groups", len(groups))

    truncation_rows = truncation_gate(groups)
    logger.info("--- item 4: tuning-budget truncation ---")
    _summarize(truncation_rows, "fraction")

    schedule_rows = schedule_gate(groups)
    logger.info("--- item 3: checkpoint schedule ---")
    _summarize(schedule_rows, "schedule")

    if args.out:
        payload = {"truncation": truncation_rows, "schedule": schedule_rows}
        Path(args.out).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
