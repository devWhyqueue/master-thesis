from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from code.common import ensure_dirs
from code.data.progan import storage as st
from code.data.progan.core import ProGanSettings
from code.data.progan.train import train_class_progan, write_generated_images
from code.data.staging.io import resolve_raw_image_path
from code.modeling.training.support import _resolve_device


_REFERENCE_FINAL_DEPTH_EPOCHS = 25
"""Default final-depth epoch count used by the image-level patch_progan_aug path."""


def progan_settings(config: dict, smoke: bool = False) -> ProGanSettings:
    """Build ProGAN settings from experiment config."""
    r = dict(config["patch_synthetic_progan"])
    g = tuple(int(v) for v in r.get("final_depth_epoch_grid", (10, 25, 50)))
    if smoke:
        r.update(image_size=8, epochs_per_depth=1)
        r.update(max_classes=1, max_real_patches_per_class=16)
        g = (1, 2, 3)
    args = [
        int(r["image_size"]),
        int(r["latent_dim"]),
        int(r["epochs_per_depth"]),
        float(r["learning_rate"]),
        float(r["beta1"]),
        int(r["max_real_patches_per_class"]),
        str(r["balance_target"]),
        int(r["max_classes"]) if r.get("max_classes") is not None else None,
        float(r["fade_in_fraction"]),
        int(r["base_channels"]),
        g,
    ]
    return ProGanSettings(*args)


def patch_seeds(config: dict, smoke: bool) -> list[int]:
    """Return patch benchmark seeds for the current run mode."""
    seeds = [int(seed) for seed in config["patch_training"]["seeds"]]
    return [seeds[0]] if smoke else seeds


def train_frame_for_seed(config: dict, seed: int) -> pd.DataFrame:
    """Load the training split of a patch manifest."""
    frame = pd.read_csv(ensure_dirs(config)["data"] / f"patch_manifest_seed={seed}.csv")
    return cast(pd.DataFrame, frame[frame["split"] == "train"])


def balance_target(train_frame: pd.DataFrame, settings: ProGanSettings) -> int:
    """Resolve the post-augmentation patch count target."""
    if settings.balance_target == "max_train_class_count":
        return int(train_frame["cancer_type"].value_counts().max())
    raise ValueError(f"Unknown ProGAN balance target: {settings.balance_target}")


def tail_classes(train_frame: pd.DataFrame, settings: ProGanSettings) -> list[str]:
    """Return minority classes that require synthetic augmentation."""
    counts = cast(pd.Series, train_frame["cancer_type"].value_counts())
    target = balance_target(train_frame, settings)
    sel = [str(n) for n, count in counts.items() if int(count) < target]
    return sel if settings.max_classes is None else sel[: settings.max_classes]


def expected_generated_counts(df: pd.DataFrame, s: ProGanSettings) -> dict[str, int]:
    """Return required synthetic patch counts per tail class."""
    t = balance_target(df, s)
    vc = cast(pd.Series, df["cancer_type"].value_counts())
    return {c: max(0, t - int(vc[c])) for c in tail_classes(df, s)}


def output_root_for_seed(config: dict, seed: int) -> Path:
    """Return the seed-level synthetic output directory (parent of per-variant dirs)."""
    return st.synthetic_output_root(ensure_dirs(config)["root"], seed)


def progan_array_upper_bound(config: dict, smoke: bool = False) -> int:
    """Return the inclusive upper bound for the parallel ProGAN SLURM array."""
    s = progan_settings(config, smoke)
    seeds = patch_seeds(config, smoke)
    max_tail = max(
        len(tail_classes(train_frame_for_seed(config, sd), s)) for sd in seeds
    )
    return max_tail * len(seeds) - 1


def decode_progan_array_task(
    config: dict, task_id: int, smoke: bool = False
) -> tuple[int, str] | None:
    """Map a SLURM array task id to a seed and tail class name."""
    seeds = patch_seeds(config, smoke)
    seed = seeds[task_id % len(seeds)]
    s = progan_settings(config, smoke)
    classes = tail_classes(train_frame_for_seed(config, seed), s)
    idx = task_id // len(seeds)
    return (seed, classes[idx]) if idx < len(classes) else None


def _progan_subsample_seed(benchmark_seed: int, class_name: str) -> int:
    """Return stable RNG seed for subsampling real patches."""
    return int.from_bytes(
        hashlib.sha256(f"{benchmark_seed}:{class_name}".encode()).digest()[:4], "big"
    )


def _class_image_paths(
    df: pd.DataFrame, class_name: str, s: ProGanSettings, raw: Path, seed: int
) -> list[Path]:
    values = df.loc[df["cancer_type"] == class_name, "image_path"]
    paths = [resolve_raw_image_path(Path(p), raw) for p in values.tolist()]
    limit = s.max_real_patches_per_class
    if len(paths) > limit:
        rng = np.random.default_rng(_progan_subsample_seed(seed, class_name))
        return [
            paths[int(i)] for i in rng.choice(len(paths), size=limit, replace=False)
        ]
    return paths


def _train_and_write_class(
    df: pd.DataFrame, s: ProGanSettings, root: Path, name: str, seed: int, raw: Path
) -> dict[int, dict[str, object]]:
    """Train one class GAN, write one synthetic directory per grid variant."""
    image_paths = _class_image_paths(df, name, s, raw, seed)
    n_real = int((df["cancer_type"] == name).sum())
    expected = expected_generated_counts(df, s)[name]
    device = _resolve_device("auto")
    snapshots, training = train_class_progan(image_paths, s, device, seed)
    per_variant: dict[int, dict[str, object]] = {}
    for var, generator in snapshots.items():
        v_dir = root / f"epochs={var}"
        gen_imgs = write_generated_images(
            generator, v_dir / name, name, s, device, expected
        )
        diag = {
            "balance_target": balance_target(df, s),
            "class_name": name,
            "final_depth_epochs": var,
            "fid": st.fid_payload(image_paths, gen_imgs, device),
            "generated_patches": expected,
            "real_train_patches": n_real,
            "training": training,
        }
        st.save_class_diagnostics(v_dir, diag)
        per_variant[var] = diag
    return per_variant


def _all_variants_complete(
    root: Path, s: ProGanSettings, expected: dict[str, int]
) -> bool:
    """Return True if every variant directory has the expected patch count for every class."""
    return all(
        st.class_is_complete(root / f"epochs={v}" / c, n)
        for v in s.final_depth_epoch_grid
        for c, n in expected.items()
    )


def generate_class_progan(
    config: dict, seed: int, class_name: str, smoke: bool = False
) -> dict[int, dict[str, object]]:
    """Train one class-specific ProGAN and write synthetic patches for each grid variant."""
    settings = progan_settings(config, smoke)
    train_frame = train_frame_for_seed(config, seed)
    seed_root = output_root_for_seed(config, seed)
    expected = expected_generated_counts(train_frame, settings).get(class_name, 0)
    grid = settings.final_depth_epoch_grid

    def dp(v):
        return st.diagnostics_path(seed_root / f"epochs={v}", class_name)

    all_done = all(
        st.class_is_complete(seed_root / f"epochs={v}" / class_name, expected)
        and dp(v).exists()
        for v in grid
    )
    if all_done:
        return {v: st.load_class_diagnostics(dp(v)) for v in grid}
    for v in grid:
        if (class_dir := seed_root / f"epochs={v}" / class_name).exists():
            shutil.rmtree(class_dir)
    raw = Path(config["paths"]["raw_root"])
    return _train_and_write_class(
        train_frame, settings, seed_root, class_name, seed, raw
    )


def merge_patch_gan_manifest(config: dict, seed: int, smoke: bool = False) -> Path:
    """Merge per-class synthetic outputs into per-variant and combined manifests."""
    settings = progan_settings(config, smoke)
    expected = expected_generated_counts(train_frame_for_seed(config, seed), settings)
    seed_root = output_root_for_seed(config, seed)
    is_comp = st.class_is_complete
    for variant in settings.final_depth_epoch_grid:
        v_dir = seed_root / f"epochs={variant}"
        missing = {c: n for c, n in expected.items() if not is_comp(v_dir / c, n)}
        if missing:
            msg = ", ".join(f"{c}={n}" for c, n in missing.items())
            raise RuntimeError(f"ProGAN classes incomplete: {msg}")
        rows = st.collect_rows(v_dir)
        if not smoke and expected and not rows:
            raise RuntimeError(f"ProGAN seed {seed} produced no patches")
        diag = st.load_diagnostics(v_dir)
        st.write_variant_manifest(v_dir, rows, diag, seed, variant, settings)
    return st.write_combined_manifest(seed_root, settings)


def merge_patch_gan_manifest_reference(
    config: dict, seed: int, smoke: bool = False
) -> Path:
    """Return the reference-variant manifest for the image-level patch_progan_aug path."""
    grid = progan_settings(config, smoke).final_depth_epoch_grid
    ref = (
        _REFERENCE_FINAL_DEPTH_EPOCHS
        if _REFERENCE_FINAL_DEPTH_EPOCHS in grid
        else max(grid)
    )
    return (
        output_root_for_seed(config, seed)
        / f"epochs={ref}"
        / "synthetic_patch_manifest.csv"
    )


def generate_patch_gan_manifest(config: dict, seed: int, smoke: bool = False) -> Path:
    """Train all class-specific GANs sequentially and write the combined manifest."""
    settings = progan_settings(config, smoke)
    train_frame = train_frame_for_seed(config, seed)
    expected = expected_generated_counts(train_frame, settings)
    root = output_root_for_seed(config, seed)
    if not _all_variants_complete(root, settings, expected):
        for class_name in tail_classes(train_frame, settings):
            generate_class_progan(config, seed, class_name, smoke)
    return merge_patch_gan_manifest(config, seed, smoke)
