# Phase 03 — extract identity-matched features

Input: locked identities, model, and environment from phase 02. Output: audited UNI2-h tensors and manifests for both datasets, plus verified pointers to Virchow2 features.

## Minimal code changes

The existing benchmark explicitly permits only frozen Virchow2. Relevant files under `experiments/01_benchmark/02_benchmark_patch/code/imbalance_benchmark/`:

- `datasets/features/__init__.py`: model loading, transforms, pooling, batch and slide extraction.
- `datasets/feature_provenance.py`: pinned encoder checks and hard-coded `FEATURE_DIM=2560` validation.
- `datasets/features/attach.py` and `provenance_lock.py`: feature references and manifest locks.
- `datasets/features/cache_manifest.py`: atomic tensors, ordered patch hashes, resumable per-slide records.
- `commands/prepare.py`: dataset preparation orchestration; do not rerun the complete benchmark protocol merely to change encoder.

Trace their callers before editing. Keep the old benchmark's Virchow2-only validation and defaults intact. Add experiment-local UNI2-h loading/provenance and use existing generic tensor/cache utilities. If shared extraction needs a small explicit encoder argument, make UNI2-h opt-in from experiment 39 and test old rejection behavior. Do not build a general plugin/encoder registry for two models. Never relabel UNI tensors with Virchow2 provenance or globally change `FEATURE_DIM`.

## Extraction contract

1. Partition unique images deterministically by slide or bounded slide group. Each shard owns disjoint tensors; one merge stage publishes the completed cache manifest.
2. Stage the existing dataset SquashFS to job-local storage where feasible. Bind datasets read-only; write only experiment-owned outputs. Keep scripts and checkpoints needed by jobs on shared storage, not login-node `/tmp`.
3. Load UNI2-h from its exact locked checkpoint and apply its locked evaluation transform to RGB source images. Run frozen inference. Store one 1,536-vector per requested patch; do not concatenate patch tokens using the Virchow2 recipe.
4. Write tensors atomically. Record ordered image IDs, tensor hash, dimensions, dtype, row count, encoder lock hash, transform hash, and input manifest hash. Keep each encoder in a separate cache namespace.
5. Resume only when the complete provenance and tensor checks pass. A file's existence is insufficient. Reject stale dimensions, changed row order, missing rows, and mismatched locks; do not silently skip corrupt images or narrow the cohort.
6. Produce model-specific manifest copies by joining immutable patch IDs. Change only feature reference fields. Preserve labels, patient/slide IDs, split labels, and the original sampling order. Keep source manifests untouched.

Avoid thousands of per-patch files; retain the existing per-slide tensor pattern. Use a single GPU worker first, then a bounded array if the resource pilot justifies it. Multi-GPU process machinery is unnecessary for this study.

## Acceptance checks

- Every requested patch has exactly one feature per encoder. Duplicate references across splits may share that feature, but never produce duplicate identity rows within a partition.
- Model-specific evaluation manifests match exactly after removing feature-reference columns. Scheduled training identities and weights match exactly.
- Every tensor has the expected row count, dimension, dtype, and finite values. Sample image and coordinate checks include both datasets, multiple slides, and all labels.
- A shuffled feature manifest fails the identity check. A modified tensor or lock fails resume validation. Restarting a completed shard makes no content changes.
- Freeze audited cache hashes and reject later mutation before fitting.

Exit gate: publish `feature_audit.json` with zero unresolved missing/corrupt/mismatched requested rows. Bulk extraction completion alone is not acceptance.
