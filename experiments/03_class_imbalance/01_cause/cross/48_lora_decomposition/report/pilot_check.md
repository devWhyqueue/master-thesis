# Pilot review (exp-48)

Pilot: draw 0 of splits 0–2, four arms, both datasets (24 fits, Hydra arrays 5197162–5197169).
Machine check: `check_pilot.json` (job 5197669) passed on both datasets with no failures.

## Technical review

- Every arm complete and compatible; within each block the initialization digest, seed, and
  recipe agree; B/F and C/R share the sampled row stream and per-class exposure; F's loss target
  equals R's realized allocation; B and R weights are one; sample count = updates × 256.
- Training loss finite throughout. Mean loss over the first ten updates: BRACS 1.42–1.92,
  TCGA-UT 2.96–3.59; over the last ten: BRACS 0.02–0.07, TCGA-UT 0.004–0.025. The fixed recipe
  fits the training patches almost exactly, as in the mitigation studies (same budget and data);
  no divergence or stalled optimization.
- Validation patient-macro balanced accuracy (diagnostic only, not a criterion) lies in the range
  of the mitigation studies' cross-entropy runs: BRACS 0.31–0.42, TCGA-UT 0.69–0.73.
- Wall time: BRACS about 30 min per fit (gpu-2h), TCGA-UT 1.4–2.2 h per fit (gpu-5h).

## Concerns recorded

- Arms of a block ran on different GPU models (A100 80GB, H100, RTX PRO 6000 Blackwell) under
  the source partitions' constraints. Initialization is identical across arms, but bf16/TF32
  numerics differ by device, which adds a small amount of unpaired training noise to the
  contrasts. The device is recorded per fit; the constraint is kept unchanged for the main phase
  so all blocks follow the same rule.

No recipe change; the pilot blocks are retained and the main phase (216 fits) proceeds.
