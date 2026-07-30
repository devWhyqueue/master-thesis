from __future__ import annotations

from imbalance_benchmark.commands.analyze import (
    cmd_analyze,
    cmd_analyze_combine,
    cmd_combine_rq3,
)
from imbalance_benchmark.commands.confirm import cmd_confirm
from imbalance_benchmark.commands.confirm.shard import cmd_confirm_shard
from imbalance_benchmark.commands.freeze import cmd_freeze
from imbalance_benchmark.commands.pilot import cmd_pilot
from imbalance_benchmark.commands.prepare import (
    cmd_prepare,
    cmd_tile_wsi,
    cmd_tile_wsi_reduce,
)
from imbalance_benchmark.commands.smoke import cmd_smoke
from imbalance_benchmark.commands.tuning import cmd_tune, cmd_tune_reduce
from imbalance_benchmark.commands.tuning.shard import cmd_tune_shard
from imbalance_benchmark.hydra import cmd_submit

__all__ = [
    "cmd_prepare",
    "cmd_pilot",
    "cmd_freeze",
    "cmd_tune",
    "cmd_tune_shard",
    "cmd_tune_reduce",
    "cmd_confirm",
    "cmd_confirm_shard",
    "cmd_analyze",
    "cmd_analyze_combine",
    "cmd_combine_rq3",
    "cmd_submit",
    "cmd_smoke",
    "cmd_tile_wsi",
    "cmd_tile_wsi_reduce",
]
