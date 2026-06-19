import argparse
import json

from job_defs import Job, prefix


def verify_features(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Build a job that validates the cls_patchmean feature store."""
    feature_dir = config.get("feature_path", "")
    cmd = prefix(config, args) + [
        "-m",
        "data.full_scale_cli",
        "--verify-features",
        f"--feature-dir={feature_dir}",
        "--slide-manifest-path=unused",
        "--file-save-path=unused",
        "--parameter=0",
    ]
    return [Job(cmd, "verify-features", "logs/verify/features%j.out")]
