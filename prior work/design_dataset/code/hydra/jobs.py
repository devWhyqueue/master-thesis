from sampling_jobs import sample_balanced, sample_imbalanced
from training_jobs import train
from visualization_jobs import visualize
from job_defs import Job, execute, load_config

COMMAND_HANDLERS = {
    "sample-balanced": sample_balanced,
    "sample-imbalanced": sample_imbalanced,
    "train": train,
    "visualize": visualize,
}

__all__ = ["COMMAND_HANDLERS", "Job", "execute", "load_config"]
