from __future__ import annotations

from imbalance_benchmark.commands.analyze import (
    cmd_analyze,
    cmd_analyze_combine,
    cmd_combine_rq3,
    cmd_match,
)
from imbalance_benchmark.commands.confirm import cmd_confirm
from imbalance_benchmark.commands.confirm.shard import cmd_confirm_shard
from imbalance_benchmark.commands.freeze import (
    cmd_amend_grids,
    cmd_freeze,
    cmd_signals,
)
from imbalance_benchmark.commands.pilot import cmd_pilot
from imbalance_benchmark.commands.tuning.panda_prepare import (
    cmd_materialize_panda,
    cmd_materialize_panda_audit,
    cmd_materialize_panda_combine,
    cmd_materialize_panda_pack,
    cmd_materialize_panda_publish,
    cmd_prepare_extract_reduce,
)
from imbalance_benchmark.commands.prepare import (
    cmd_materialize_tcga_ut,
    cmd_prepare,
    cmd_prepare_extract_shard,
    cmd_tile_wsi,
    cmd_tile_wsi_reduce,
)
from imbalance_benchmark.commands.smoke import cmd_smoke
from imbalance_benchmark.commands.tuning import cmd_tune, cmd_tune_reduce
from imbalance_benchmark.commands.tuning.wave import cmd_tune_wave
from imbalance_benchmark.commands.tuning.decide import cmd_tune_decide
from imbalance_benchmark.commands.tuning.shard import cmd_tune_shard
from imbalance_benchmark.hydra import cmd_submit

__all__ = [
    "cmd_prepare",
    "cmd_prepare_extract_shard",
    "cmd_prepare_extract_reduce",
    "cmd_pilot",
    "cmd_freeze",
    "cmd_amend_grids",
    "cmd_signals",
    "cmd_materialize_tcga_ut",
    "cmd_materialize_panda",
    "cmd_materialize_panda_audit",
    "cmd_materialize_panda_combine",
    "cmd_materialize_panda_pack",
    "cmd_materialize_panda_publish",
    "cmd_tune",
    "cmd_tune_shard",
    "cmd_tune_reduce",
    "cmd_tune_wave",
    "cmd_tune_decide",
    "cmd_confirm",
    "cmd_confirm_shard",
    "cmd_analyze",
    "cmd_analyze_combine",
    "cmd_combine_rq3",
    "cmd_match",
    "cmd_submit",
    "cmd_smoke",
    "cmd_tile_wsi",
    "cmd_tile_wsi_reduce",
]
