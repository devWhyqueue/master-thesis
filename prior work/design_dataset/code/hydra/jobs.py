from sampling_jobs import sample_balanced, sample_full_scale, sample_imbalanced
from training_jobs import train
from wsi_jobs import train_wsi
from visualization_jobs import visualize
from report_jobs import report
from job_defs import Job, execute, load_config

COMMAND_HANDLERS = {
    "sample-balanced": sample_balanced,
    "sample-imbalanced": sample_imbalanced,
    "sample-full-scale": sample_full_scale,
    "train": train,
    "train-wsi": train_wsi,
    "visualize": visualize,
    "report": report,
}

__all__ = ["COMMAND_HANDLERS", "Job", "execute", "load_config"]
