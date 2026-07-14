from __future__ import annotations

import argparse
from imbalance_benchmark.commands.freeze_execution import freeze_split
from imbalance_benchmark.manifest.freezing import _build_conditions

__all__ = ["cmd_freeze", "_build_conditions"]


def cmd_freeze(args: argparse.Namespace) -> None:
    """Freeze the definitive condition manifests and content-hashed analysis manifest."""
    if args.split_index is None:
        for index in range(3):
            cmd_freeze(argparse.Namespace(**{**vars(args), "split_index": index}))
        return
    freeze_split(args)
