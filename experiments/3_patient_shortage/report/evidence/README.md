# Result provenance

Retrieved from Hydra on 2026-09-10. Source root:
`/home/yannik.qu/master-thesis/experiments/3_patient_shortage/outputs/{dataset}/patch/`.

Each dataset directory preserves `tables/decodability.json`, `data/preflight.json`,
`data/probe_selection.json`, and the original SHA-256 sidecars for the two data files.
Both preflight statuses are `pass`; all four sidecars match their copied files.
The embedded selection in each result is the validation-selected probe configuration.

Result JSON SHA-256 values match those read directly on Hydra:

- BRACS: `1dd27831c60ff33b2cfafdcc60aa421e136fcffc6b7f8aa6f23bdbf7fc3655b4`
- TCGA-UT: `30dafeabebe71d0a8fe52d547ec3b39a1b23f8af660290312cb6f657f2a44523`

Remote result modification times: BRACS 2026-09-10 10:43:35 +0200;
TCGA-UT 2026-09-10 10:53:47 +0200. Remote checkout HEAD:
`7732e66cb2350e69a6eb92da546f14c530d69f2e`.
The inspected remote `analyze/contrasts.py`, `analyze/__init__.py`, and benchmark
`analysis/inference/context.py` have identical SHA-256 values to the local files.
This records inspected source parity, not proof of a clean remote checkout or the
exact source revision at execution time.

`results.tex` transcribes primary estimates from `contrasts`, with two-decimal
rounding. Split probe estimates come from `split_endpoints`; secondary pooled
accuracies and class-recall benefit counts average the three splits equally.
Differences are calculated before rounding. The cross-condition TCGA-UT comparison
is a point difference only; no interval is inferred from marginal intervals.

Reporting limitation: `_gather_split_endpoints` uses `preds_stack[0]` for the MLP.
Thus split/secondary MLP values are not five-run means and are excluded from those
comparisons. The separate primary contrast calculation uses all five MLP runs.
No benchmark code, model results, or cluster jobs were changed for this integration.
