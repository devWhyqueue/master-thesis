from sampling_jobs import sample_balanced, sample_full_scale, sample_imbalanced
from training_jobs import train
from tuning_jobs import tune, tune_aggregate, tune_wsi
from wsi_cache_jobs import wsi_cache
from wsi_jobs import train_wsi
from visualization_jobs import visualize
from report_jobs import report
from verify_jobs import verify_features
from job_defs import Job, execute, load_config

COMMAND_HANDLERS = {
    "sample-balanced": sample_balanced,
    "sample-imbalanced": sample_imbalanced,
    "sample-full-scale": sample_full_scale,
    "train": train,
    "train-wsi": train_wsi,
    "wsi-cache": wsi_cache,
    "tune": tune,
    "tune-wsi": tune_wsi,
    "tune-aggregate": tune_aggregate,
    "visualize": visualize,
    "report": report,
    "verify-features": verify_features,
}

__all__ = ["COMMAND_HANDLERS", "Job", "execute", "load_config"]
