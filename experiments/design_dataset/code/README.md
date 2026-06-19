# TCGA-UT Controlled Imbalance Benchmark

This experiment stack builds strict full-scale TCGA-UT training splits for the controlled follow-up to the native class-imbalance benchmark.

## Supported workflow

The active workflow is:

1. prepare case-disjoint slide manifests for seeds `0,1,2`,
2. derive a shared feasible pool size `N`,
3. sample strict constructed splits for `lambda in {0.8, 1.1, 1.3}` under `native_prevalence`,
4. build patch and WSI caches,
5. run patch and WSI validation tuning,
6. aggregate report tables and figures.

Legacy balanced-to-imbalanced sampling is retired and is not part of the supported workflow anymore.

## Strict constructed sampling

Use `data.full_scale_cli` through the Hydra wrapper.

### Derive one-seed feasibility

To compute the maximum feasible pool size for one seed and one `lambda`:

```bash
python hydra/run.py --local --no-container max-feasible-pool-size \
  --parameter 1.3 \
  --seed 0
```

The active config must provide `full_scale_pool_size` for real sampling runs. For the benchmark, compute the per-seed maximum at the steepest reported regime and use the minimum across seeds as the shared `N`.

### Run strict sampling

```bash
python hydra/run.py sample-full-scale --parameter 0.8 --seed 0
```

The sampler fails fast when the requested `(lambda, N)` target distribution is infeasible. There is no redistribution or replacement fallback.

## Training and tuning

Patch-feature tuning and WSI-bag tuning operate on the constructed manifests and reuse the shared tuning grid from the native benchmark.

```bash
python hydra/run.py tune
python hydra/run.py tune-wsi
```

Patch-feature tuning includes the full first-report method panel, including OKO and ProGAN-augmented runs when the constructed manifests include the corresponding synthetic rows.

## Hydra config

The active config must define:

- `slide_manifest_csv`
- `split_assignment_csv`
- `constructed_dataset_dir`
- `feature_path`
- `results_dir`
- `report_output_dir`
- `full_scale_pool_size`

Copy `hydra/config.json.template` to `hydra/config.json` and fill in the environment-specific paths before running jobs.
