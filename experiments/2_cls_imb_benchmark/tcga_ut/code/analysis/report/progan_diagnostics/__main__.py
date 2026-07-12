from __future__ import annotations

import argparse
from pathlib import Path

from code.common import ensure_dirs, load_config
from code.analysis.results import connect, init_schema, replace_table
from code.analysis.report.progan_diagnostics.metrics import (
    assert_summary_complete,
    build_metrics_frame,
    expected_augmented_classes,
    write_summary_latex,
)
from code.analysis.report.progan_diagnostics.plots import (
    plot_examples,
    select_example_classes,
)


def _patch_feature_cache_dir(config: dict, seed: int) -> Path:
    return ensure_dirs(config)["data"] / "patch_feature_cache" / f"seed={seed}"


def parse_args() -> argparse.Namespace:
    """Parse ProGAN diagnostic generation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--example-classes",
        default="Cholangiocarcinoma,Uveal_Melanoma,Lymphoid_Neoplasm_Diffuse_Large_B-cell_Lymphoma",
    )
    parser.add_argument("--examples-per-class", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    """Build ProGAN quality diagnostics for the paper."""
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    cache_dir = _patch_feature_cache_dir(config, args.seed)
    frame = build_metrics_frame(paths, args.seed, cache_dir)
    assert_summary_complete(
        frame, expected_augmented_classes(paths, config, args.seed), args.seed
    )
    stored = frame.copy()
    connection = connect(paths["db"])
    init_schema(connection)
    replace_table(connection, "progan_diagnostics", stored)
    connection.close()
    stem = f"progan_diagnostics_seed{args.seed}"
    write_summary_latex(frame, paths["tables"] / f"{stem}.tex")
    plot_examples(
        paths,
        config,
        args.seed,
        cache_dir,
        frame,
        select_example_classes(frame, args.example_classes),
        args.examples_per_class,
        paths["figures"] / f"progan_examples_seed{args.seed}.png",
    )


if __name__ == "__main__":
    main()
