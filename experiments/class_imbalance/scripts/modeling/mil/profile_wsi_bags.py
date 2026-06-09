from __future__ import annotations

import argparse
import json
import logging
import time
from typing import cast

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.common import ensure_dirs, load_config
from scripts.analysis.results import connect, init_schema, replace_table
from scripts.modeling.mil.bag.dataset import (
    AttentionMil,
    BagFeatureDataset,
    bag_collate,
    infer_input_dim,
)
from scripts.modeling.training.support import _resolve_device

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse WSI-bag profiling arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Profile available bag lengths before choosing any optional cap."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    frame = pd.read_csv(paths["data"] / f"manifest_splits_seed={args.seed}.csv")
    payload = _profile_payload(frame, config, args.seed)
    connection = connect(paths["db"])
    init_schema(connection)
    replace_table(
        connection,
        "wsi_bag_profile",
        pd.DataFrame(
            [{"seed": args.seed, "payload_json": json.dumps(payload, sort_keys=True)}]
        ),
    )
    connection.close()


def _profile_payload(frame: pd.DataFrame, config: dict, seed: int) -> dict[str, object]:
    train = pd.DataFrame(frame[frame["split"] == "train"])
    class_to_idx = {
        name: idx for idx, name in enumerate(sorted(frame["cancer_type"].unique()))
    }
    length_frame = _scan_bag_lengths(train)
    profile_rows = _largest_batch_rows(
        length_frame, int(config["wsi_training"]["bag_batch_size"])
    )
    dataset = BagFeatureDataset(profile_rows, class_to_idx, None)
    lengths = torch.tensor(length_frame["instances"].tolist())
    runtime = _profile_runtime(dataset, len(class_to_idx), config["wsi_training"])
    return {
        "seed": seed,
        "n_bags": len(length_frame),
        "min_instances": int(lengths.min()),
        "mean_instances": float(lengths.float().mean()),
        "median_instances": float(lengths.float().median()),
        "max_instances": int(lengths.max()),
        **runtime,
    }


def _scan_bag_lengths(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, row in enumerate(frame.to_dict("records"), start=1):
        tensor = torch.load(str(row["feature_path"]), map_location="cpu")
        features = _tensor_payload(tensor)
        rows.append({**row, "instances": _feature_count(features)})
        if index % 100 == 0:
            logger.info("profiled_bag_lengths=%s", index)
    return pd.DataFrame(rows)


def _tensor_payload(tensor: object) -> torch.Tensor:
    if isinstance(tensor, dict):
        return next(value for value in tensor.values() if torch.is_tensor(value))
    return cast(torch.Tensor, tensor)


def _feature_count(tensor: torch.Tensor) -> int:
    if tensor.ndim == 1:
        return 1
    if tensor.ndim > 2:
        return int(tensor.reshape(-1, tensor.shape[-1]).shape[0])
    return int(tensor.shape[0])


def _largest_batch_rows(frame: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    rows = frame.sort_values("instances", ascending=False).head(batch_size)
    logger.info(
        "profile_runtime_batch_size=%s profile_runtime_instances=%s",
        len(rows),
        int(rows["instances"].sum()),
    )
    return pd.DataFrame(rows)


def _profile_runtime(
    dataset: BagFeatureDataset, n_classes: int, training: dict
) -> dict[str, object]:
    device = _resolve_device(str(training["device"]))
    batch_size = int(training["bag_batch_size"])
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=bag_collate
    )
    bags, targets = next(iter(loader))
    model = AttentionMil(
        infer_input_dim(dataset),
        int(training["hidden_dim"]),
        n_classes,
        float(training["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"])
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    logits, _, _ = model.forward_bags([bag.to(device) for bag in bags])
    loss = nn.functional.cross_entropy(logits, targets.to(device))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "device": str(device),
        "profiled_batch_size": len(bags),
        "profiled_batch_instances": int(sum(len(bag) for bag in bags)),
        "forward_backward_seconds": elapsed,
        "cuda_peak_memory_bytes": _peak_memory(device),
    }


def _peak_memory(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


if __name__ == "__main__":
    main()
