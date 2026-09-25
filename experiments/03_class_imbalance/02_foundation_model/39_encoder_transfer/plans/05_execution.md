# Phase 05 — bounded Hydra execution

Input: accepted code and locked artifacts. Output: complete main run records, job/resource ledger, and a completeness audit. No jobs are authorized as already executed by this document.

Phase-02 `input_audit.json` mislabeled test-only patch counts as validation-plus-test counts. The signed `configs/phase05_scope_amendment.json` preserves that original audit and records the corrected requested unions, checked independently against the frozen schedule and source manifests by Hydra job 4923112. Feature preflight now checks the corrected counts before fitting; the patient schedule, model, readout grid, and inference plan are unchanged.

## Order and job boundaries

1. Re-read `CLUSTER.md` and the project hydra-cluster skill. Check live partitions, available storage, existing jobs, and repo state. Use login-shell SSH for SLURM. Do not discard remote changes to make synchronization work.
2. Build/verify the extraction SIF and obtain weights on an appropriate compute allocation. The login node handles lightweight checks and submission only. Credentials must not appear in scripts, logs, or committed configs.
3. Run phase 02's `gpu-test` pilot, then `cpu-test` for imports and a tiny synthetic fit/analyze round trip. Run the real engineering pilot on draw 10000, split 0, both datasets and both encoders. Use validation outcomes for numerical and grid decisions; keep its test outcomes outside main inference.
4. Freeze the protocol, model/cache schema, lambda grid, and resource estimates. Submit remaining extraction shards; merge/audit only after successful extraction dependencies.
5. Fit main draws 10–19 across three splits for each encoder/dataset. Reuse audited Virchow2 features, but rerun its probes under the same frozen protocol as UNI2-h.
6. Analyze only after all required fit records pass the completeness check. Produce report assets only from an accepted analysis artifact. Dependencies use `afterok`; failed upstream jobs leave analysis blocked.

## Fit budget

The base main design contains 2 datasets × 2 encoders × 3 splits × 10 draws × 7 arms = **840 selected arm models**. With 11 lambdas, it requires **9,240 candidate optimizations**, not 840 optimizer calls. Fixed-B-lambda sensitivities reuse these candidates. The engineering pilot adds 28 selected arms / 308 candidate optimizations at the initial grid size. Any grid expansion changes this ledger explicitly.

There are 120 main fit shards if each shard handles one dataset/encoder/split/draw and its seven arms. Submit in batches or arrays that keep the account's total queued plus running task count at or below 100, including unrelated existing jobs. Do not set an array concurrency throttle.

Start extraction with one GPU per task. Pilot on an available compatible GPU; use `gpu-2h` or `gpu-5h` only after throughput supports the wall time. Existing exp-27 configurations request 16 CPUs and 64 GB on `cpu-5h`; these are reference requests, not verified requirements for UNI2-h. Measure a full seven-arm shard and select the shortest sufficient partition with headroom. Avoid nested BLAS/joblib oversubscription.

Record GPU model, precision, batch size, images/sec, peak GPU/host memory, staging duration, extraction duration, candidate-fit duration, cache bytes, and actual SLURM resource use. Project remaining cost from these measurements. Do not state a fixed runtime before measuring it.

## Failure and resume rules

- Keep job scripts and job dependencies on shared storage. Job-local `/tmp` holds only disposable staging data. Use job-ID-specific logs.
- Worker completion means successful exit plus valid atomic output and matching locks. Mere tensor/run-record existence is not enough.
- Access/model/identity failures stop extraction. Missing classes or changed sampling stop fitting. Never drop a failed patient, arm, class, or encoder to complete the grid.
- A transient failed shard may retry unchanged. OOM may lower extraction batch size after a numerical consistency check; it must not alter requested IDs or precision silently.
- If the frozen grid has no converged candidate, mark the cell failed, diagnose, and record a protocol amendment before a complete consistent rerun of affected comparisons. No success-only aggregation.
- Inspect `sacct` and logs for failed dependencies. Cancel only this experiment's obsolete dependent jobs. Resubmit only incomplete compatible shards, then the dependent audit/analysis stages.
- A failed infrastructure gate requires repair. There is no gate requiring UNI2-h to outperform Virchow2 and no optional stopping on significance.

Exit gate: all 840 selected arm records and their required candidate evidence are present, paired, and valid; the exact final count reflects any documented protocol amendment. Archive job IDs, final locks, logs, and the completeness report alongside results.
