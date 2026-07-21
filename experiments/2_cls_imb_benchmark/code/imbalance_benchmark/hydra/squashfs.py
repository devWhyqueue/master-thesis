from __future__ import annotations

import posixpath
import shlex
from typing import Any


def _stage_images(config: dict[str, Any], stage: str) -> list[tuple[str, str]]:
    images = config.get("slurm", {}).get("squashfs", [])
    return [
        (str(image["source"]), str(image["mount"]))
        for image in images
        if stage in image.get("stages", ("prepare",))
    ]


def _staging_lines(images: list[tuple[str, str]]) -> list[str]:
    if not images:
        return []
    lines = [
        'STAGE_DIR="/tmp/imbalance-benchmark-${SLURM_JOB_ID}"',
        'mkdir -p "$STAGE_DIR"',
    ]
    for index, (source, mount) in enumerate(images):
        local = f"$STAGE_DIR/{index}.sqfs"
        lines.extend(
            [
                f'cp {shlex.quote(source)} "{local}"',
                f'BINDS+=("-B" "{local}:{mount}:image-src=/")',
            ]
        )
    return lines


def _generated_tile_squashfs(
    config: dict[str, Any], stage: str
) -> tuple[str, str] | None:
    dataset = config.get("dataset", {})
    image = dataset.get("tile_squashfs")
    if stage != "prepare" or not image:
        return None
    source = str(dataset["tile_root"])
    if not source.startswith("/tmp/"):
        raise ValueError("Generated SquashFS content must use job-local /tmp.")
    return source, str(image)


def _mount_generated_tile_lines(generated: tuple[str, str] | None) -> list[str]:
    if not generated:
        return []
    source, image = generated
    return [
        f"if [ -f {shlex.quote(image)} ]; then",
        f'  BINDS+=("-B" {shlex.quote(f"{image}:{source}:image-src=/")})',
        "fi",
    ]


def _pack_generated_tile_lines(generated: tuple[str, str] | None) -> list[str]:
    if not generated:
        return []
    raw_image = generated[1]
    source, image = map(shlex.quote, generated)
    partial = shlex.quote(f"{raw_image}.partial")
    return [
        f"if [ ! -f {image} ]; then",
        f"  mkdir -p {shlex.quote(posixpath.dirname(raw_image))}",
        f"  rm -f {partial}",
        f"  squash-dataset {source} {partial}",
        f"  mv {partial} {image}",
        "fi",
    ]
