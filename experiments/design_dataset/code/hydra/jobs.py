from patch_cache_jobs import patch_cache
from progan_cache_jobs import patch_cache_progan
from sampling_jobs import max_feasible_pool_size, sample_full_scale
from training_jobs import train
from tuning_jobs import tune, tune_aggregate, tune_wsi
from wsi_cache_jobs import wsi_cache
from wsi_jobs import train_wsi
from visualization_jobs import visualize
from report_jobs import report
from verify_jobs import verify_features
from job_defs import Job, execute, load_config

COMMAND_HANDLERS = {
    "max-feasible-pool-size": max_feasible_pool_size,
    "sample-full-scale": sample_full_scale,
    "train": train,
    "train-wsi": train_wsi,
    "wsi-cache": wsi_cache,
    "patch-cache": patch_cache,
    "patch-cache-progan": patch_cache_progan,
    "tune": tune,
    "tune-wsi": tune_wsi,
    "tune-aggregate": tune_aggregate,
    "visualize": visualize,
    "report": report,
    "verify-features": verify_features,
}

__all__ = ["COMMAND_HANDLERS", "Job", "execute", "load_config"]
