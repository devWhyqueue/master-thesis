"""CLI and compatibility helpers for patch-level ProGAN manifests."""

from __future__ import annotations

import argparse
import logging

from code.common import load_config
from code.data.progan.manifest import (
    decode_progan_array_task,
    generate_class_progan,
    generate_patch_gan_manifest,
    merge_patch_gan_manifest,
    progan_array_upper_bound,
)

__all__ = ["generate_patch_gan_manifest", "progan_array_upper_bound"]


def parse_args() -> argparse.Namespace:
    """Parse ProGAN manifest CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--class-name")
    parser.add_argument("--array-task-id", type=int)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Generate or merge patch-level ProGAN augmentations."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()
    config = load_config(args.config)
    if args.array_task_id is not None:
        decoded = decode_progan_array_task(config, args.array_task_id, args.smoke)
        if decoded is None:
            logging.info("Skipping unused array task %s", args.array_task_id)
            return
        seed, class_name = decoded
        generate_class_progan(config, seed, class_name, args.smoke)
        return
    if args.seed is None:
        raise ValueError("Provide --seed or --array-task-id.")
    if args.class_name:
        generate_class_progan(config, args.seed, args.class_name, args.smoke)
        return
    if args.merge_only:
        path = merge_patch_gan_manifest(config, args.seed, args.smoke)
    else:
        path = generate_patch_gan_manifest(config, args.seed, args.smoke)
    logging.info("Wrote %s", path)


if __name__ == "__main__":
    main()
