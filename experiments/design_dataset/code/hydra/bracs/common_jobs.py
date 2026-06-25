"""Shared command builders for BRACS Hydra jobs."""

from __future__ import annotations

import argparse

from job_defs import prefix


def bracs_root(config: dict[str, str]) -> str:
    """Return the writable BRACS working root for derived artifacts."""
    return config.get("bracs_root", "/home/space/datasets/patho_ds/bracs")


def bracs_data_root(config: dict[str, str]) -> str:
    """Return the read-only BRACS raw data root (pre-existing shared dataset)."""
    return config.get("bracs_data_root", bracs_root(config))


def report_output_dir(config: dict[str, str]) -> str:
    """Return the design-dataset report output directory."""
    return config.get("report_output_dir", "experiments/design_dataset/report/outputs")


def power_law_report_dir(config: dict[str, str]) -> str:
    """Return the isolated BRACS power-law report output directory."""
    return f"{report_output_dir(config)}/bracs_power_law"


def native_tune_cmd(
    args: argparse.Namespace,
    config: dict[str, str],
    benchmark: str,
    *,
    gpu: bool = False,
) -> list[str]:
    """Build a native BRACS tuning command."""
    root = bracs_root(config)
    return prefix(config, args, gpu=gpu) + [
        "-m",
        "analysis.evaluation.native_tuning_run",
        f"--benchmark={benchmark}",
        f"--manifest-root={root}/manifests",
        f"--results-dir={config.get('bracs_results_dir', '')}",
        f"--feature-path={root}/features/virchow2",
        f"--prepare-report={root}/bracs_prepare_report.json",
        "--required-mode=native",
    ]


def constructed_tune_cmd(
    args: argparse.Namespace,
    config: dict[str, str],
    benchmark: str,
    *,
    gpu: bool = False,
) -> list[str]:
    """Build a BRACS power-law tuning command."""
    root = bracs_root(config)
    return prefix(config, args, gpu=gpu) + [
        "-m",
        "analysis.evaluation.tuning_run",
        f"--benchmark={benchmark}",
        f"--config={args.config}",
        f"--constructed-dataset-dir={root}/constructed_power_law",
        f"--results-dir={config.get('bracs_results_dir', '')}",
        f"--feature-path={root}/features/virchow2",
        f"--prepare-report={root}/bracs_prepare_report.json",
        "--required-mode=power_law",
    ]


def bracs_progan_cmd(
    args: argparse.Namespace, config: dict[str, str], seed: int
) -> list[str]:
    """Build the native BRACS ProGAN cache command."""
    root = bracs_root(config)
    stem = f"{root}/manifests/native_seed={seed}"
    return mode_gate(config, args, "native", gpu=True) + [
        "--",
        "-m",
        "data.progan.cache",
        f"--manifest-path={stem}/manifest_splits.csv",
        f"--manifest-save-path={stem}/manifest_splits_progan.csv",
        f"--file-save-path={stem}/patch_feature_cache_progan.pt",
        f"--synthetic-root={stem}/synthetic_patch_images",
        f"--raw-root={root}/tiles",
        "--raw-resolution=.",
        f"--seed={seed}",
        *progan_options(config),
    ]


def power_law_progan_cmd(
    args: argparse.Namespace,
    config: dict[str, str],
    parameter: float,
    seed: int,
) -> list[str]:
    """Build the BRACS power-law ProGAN cache command."""
    root = bracs_root(config)
    stem = (
        f"{root}/constructed_power_law/"
        f"constructed_order=native_prevalence_parameter={parameter}_seed={seed}"
    )
    return mode_gate(config, args, "power_law", gpu=True) + [
        "--",
        "-m",
        "data.progan.cache",
        f"--manifest-path={stem}/manifest_splits.csv",
        f"--manifest-save-path={stem}/manifest_splits_progan.csv",
        f"--file-save-path={stem}/patch_feature_cache_progan.pt",
        f"--synthetic-root={stem}/synthetic_patch_images",
        f"--raw-root={root}/tiles",
        "--raw-resolution=.",
        f"--seed={seed}",
        *progan_options(config),
    ]


def mode_gate(
    config: dict[str, str],
    args: argparse.Namespace,
    required_mode: str,
    *,
    gpu: bool = False,
) -> list[str]:
    """Wrap a command so it only runs for the selected BRACS benchmark mode."""
    root = bracs_root(config)
    return prefix(config, args, gpu=gpu) + [
        "-m",
        "data.bracs.mode_gate",
        f"--prepare-report={root}/bracs_prepare_report.json",
        f"--required-mode={required_mode}",
    ]


def progan_options(config: dict[str, str]) -> list[str]:
    """Return shared ProGAN cache options."""
    return [
        f"--device={config.get('progan_device', 'cuda')}",
        f"--image-size={config.get('progan_image_size', '256')}",
        f"--latent-dim={config.get('progan_latent_dim', '256')}",
        f"--epochs-per-depth={config.get('progan_epochs_per_depth', '50')}",
        f"--learning-rate={config.get('progan_learning_rate', '0.001')}",
        f"--beta1={config.get('progan_beta1', '0.0')}",
        "--max-real-patches-per-class="
        f"{config.get('progan_max_real_patches_per_class', '2048')}",
        f"--fade-in-fraction={config.get('progan_fade_in_fraction', '0.5')}",
        f"--base-channels={config.get('progan_base_channels', '256')}",
        "--feature-model-name="
        f"{config.get('progan_feature_model_name', 'hf-hub:paige-ai/Virchow2')}",
        f"--feature-batch-size={config.get('progan_feature_batch_size', '64')}",
        f"--feature-dtype={config.get('progan_feature_dtype', 'float16')}",
    ]
