import argparse

from job_defs import Job, execute_progan_pipeline, parameters, prefix


def patch_cache_progan(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Return all ProGAN jobs as a flat list (shard jobs followed by finalize).

    Used for local / dry-run sequential execution via the standard runner.
    For SLURM parallel dispatch use execute_progan instead.
    """
    n_shards = int(config.get("progan_n_shards", "10"))
    seeds = [0, 1, 2] if args.sweep else [args.seed]
    jobs = []
    for parameter in parameters(args):
        for seed in seeds:
            for i in range(n_shards):
                jobs.append(_shard_job(args, config, parameter, seed, i, n_shards))
            jobs.append(_finalize_job(args, config, parameter, seed))
    return jobs


def execute_progan(
    args: argparse.Namespace,
    config: dict[str, str],
    local: bool,
    dry_run: bool,
) -> None:
    """Submit N shard jobs per (parameter, seed), then a finalize job with afterok dependency."""
    n_shards = int(config.get("progan_n_shards", "10"))
    seeds = [0, 1, 2] if args.sweep else [args.seed]
    for parameter in parameters(args):
        for seed in seeds:
            shards = [
                _shard_job(args, config, parameter, seed, i, n_shards)
                for i in range(n_shards)
            ]
            finalize = _finalize_job(args, config, parameter, seed)
            execute_progan_pipeline(shards, finalize, config, local, dry_run)


def _shard_job(
    args: argparse.Namespace,
    config: dict[str, str],
    parameter: float,
    seed: int,
    shard_index: int,
    n_shards: int,
) -> Job:
    cmd = _progan_cache_cmd(args, config, parameter, seed) + [
        f"--class-shard={shard_index}/{n_shards}",
        "--train-only",
    ]
    return Job(
        cmd,
        "patch_cache_progan_shard",
        "logs/patch/patch_cache_progan_shard%j.out",
        partition=config.get("progan_partition", "gpu-5h"),
        gpus_per_node=1,
        constraint=config.get("progan_gpu_constraint") or None,
    )


def _finalize_job(
    args: argparse.Namespace,
    config: dict[str, str],
    parameter: float,
    seed: int,
) -> Job:
    return Job(
        _progan_cache_cmd(args, config, parameter, seed),
        "patch_cache_progan",
        "logs/patch/patch_cache_progan%j.out",
        partition=config.get("progan_partition", "gpu-5h"),
        gpus_per_node=1,
        mem=config.get("progan_finalize_mem", "128G"),
        constraint=config.get("progan_gpu_constraint") or None,
    )


def _progan_cache_cmd(
    args: argparse.Namespace,
    config: dict[str, str],
    parameter: float,
    seed: int,
) -> list[str]:
    stem = _constructed_stem(args, config, parameter, seed)
    cmd = prefix(config, args, gpu=True) + [
        "-m",
        "data.progan.cache",
        f"--manifest-path={stem}/manifest_splits.csv",
        f"--manifest-save-path={stem}/manifest_splits_progan.csv",
        f"--file-save-path={stem}/patch_feature_cache_progan.pt",
        f"--synthetic-root={stem}/synthetic_patch_images",
        f"--raw-root={config.get('raw_root', '')}",
        f"--raw-resolution={config.get('raw_resolution', '0')}",
        f"--seed={seed}",
        f"--device={config.get('progan_device', 'cuda')}",
        f"--image-size={config.get('progan_image_size', '256')}",
        f"--latent-dim={config.get('progan_latent_dim', '256')}",
        f"--epochs-per-depth={config.get('progan_epochs_per_depth', '50')}",
        f"--learning-rate={config.get('progan_learning_rate', '0.001')}",
        f"--beta1={config.get('progan_beta1', '0.0')}",
        f"--max-real-patches-per-class={config.get('progan_max_real_patches_per_class', '2048')}",
        f"--fade-in-fraction={config.get('progan_fade_in_fraction', '0.5')}",
        f"--base-channels={config.get('progan_base_channels', '256')}",
        f"--feature-model-name={config.get('progan_feature_model_name', 'hf-hub:paige-ai/Virchow2')}",
        f"--feature-batch-size={config.get('progan_feature_batch_size', '64')}",
        f"--feature-dtype={config.get('progan_feature_dtype', 'float16')}",
    ]
    max_classes = config.get("progan_max_classes")
    if max_classes not in (None, ""):
        cmd.append(f"--max-classes={max_classes}")
    final_depth = config.get("progan_final_depth_epochs", [10, 25, 50])
    if isinstance(final_depth, list):
        cmd.append("--final-depth-epochs")
        cmd.extend(str(value) for value in final_depth)
    return cmd


def _constructed_stem(
    args: argparse.Namespace,
    config: dict[str, str],
    parameter: float,
    seed: int,
) -> str:
    name = (
        f"constructed_order={args.class_order_name}_parameter={parameter}_seed={seed}"
    )
    return f"{config.get('constructed_dataset_dir', '')}/{name}"
