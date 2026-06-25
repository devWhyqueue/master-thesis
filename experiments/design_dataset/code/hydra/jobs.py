from bracs.native_jobs import (
    bracs_features,
    bracs_patch_cache,
    bracs_prepare,
    bracs_progan_cache,
    bracs_report,
    bracs_stage,
    bracs_tune,
    bracs_tune_aggregate,
    bracs_tune_wsi,
    bracs_wsi_cache,
)
from bracs.power_law_jobs import (
    bracs_power_law,
    bracs_progan_power_law,
    bracs_report_power_law,
    bracs_tune_aggregate_power_law,
    bracs_tune_power_law,
    bracs_tune_wsi_power_law,
)
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
from job_defs import Job, execute, execute_progan_pipeline, load_config

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
    "bracs-stage": bracs_stage,
    "bracs-prepare": bracs_prepare,
    "bracs-features": bracs_features,
    "bracs-patch-cache": bracs_patch_cache,
    "bracs-wsi-cache": bracs_wsi_cache,
    "bracs-progan-cache": bracs_progan_cache,
    "bracs-progan-power-law": bracs_progan_power_law,
    "bracs-power-law": bracs_power_law,
    "bracs-tune": bracs_tune,
    "bracs-tune-wsi": bracs_tune_wsi,
    "bracs-tune-power-law": bracs_tune_power_law,
    "bracs-tune-wsi-power-law": bracs_tune_wsi_power_law,
    "bracs-tune-aggregate": bracs_tune_aggregate,
    "bracs-tune-aggregate-power-law": bracs_tune_aggregate_power_law,
    "bracs-report": bracs_report,
    "bracs-report-power-law": bracs_report_power_law,
}

__all__ = [
    "COMMAND_HANDLERS",
    "Job",
    "execute",
    "execute_progan_pipeline",
    "load_config",
]
