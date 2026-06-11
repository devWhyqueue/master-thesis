import argparse

from job_defs import Job, prefix


def report(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Build report aggregation jobs."""
    cmd = prefix(config, args) + [
        "-m",
        "tcga_ut_imbalanced.plotting.report",
        f"--constructed-dataset-dir={config.get('constructed_dataset_dir', '')}",
        f"--results-dir={config.get('results_dir', '')}",
        f"--output-dir={config.get('report_output_dir', '')}",
    ]
    return [Job(cmd, "report", "logs/report/report%j.out")]
