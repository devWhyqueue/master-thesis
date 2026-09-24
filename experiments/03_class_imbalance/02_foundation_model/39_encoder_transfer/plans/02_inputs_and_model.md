# Phase 02 — audit inputs and obtain UNI2-h

Input: phase 01 design and `CLUSTER.md`. Outputs: `input_audit.json`, `encoder_lock.json`, an immutable extraction environment, and a measured pilot resource record. Perform checks on Hydra during implementation; cluster state was not verified while writing these plans.

## Input audit

1. Inspect `~/master-thesis` state and exact benchmark output locations. Read manifests and freeze metadata under `experiments/01_benchmark/02_benchmark_patch/outputs/{tcga_ut,bracs}/patch`. Hash originals; never overwrite their splits or caches.
2. Resolve image identity from dataset, patient, slide/ROI, patch ID, and tile coordinates where applicable. Preserve label order. Check patient disjointness within each split and record overlap between different splits.
3. Resolve TCGA-UT source JPGs from its provenance-backed project SquashFS and BRACS's existing `prepared/roi_tiles.sqfs`. Reuse the existing full 256×256 ROI tiles. Do not retile BRACS or replace TCGA-UT images with public slide embeddings.
4. Verify the common depth-160 eligibility pool before drawing any arm. Freeze all main and pilot selections without reading features. Count unique requested train, validation, and test patches, not their duplicated split occurrences.
5. Default extraction scope: union of scheduled training rows plus every frozen validation/test row. This covers both datasets without extracting unused training patches. Document scope explicitly; expanding the study later requires a new manifest. Preserve the full original training metadata for eligibility and allocation, even when some unselected rows have no UNI feature.
6. Audit existing Virchow2 tensors against their locked revision, weights hash, pooling, ordered row IDs, and image provenance. Reuse only verified features. If unavailable, re-extract the required image union with the original locked pipeline. Do not substitute current hub weights.

## Model and access

The [official UNI2-h card](https://huggingface.co/MahmoodLab/UNI2-h), inspected 2026-09-24, specifies a gated checkpoint, a custom ViT-H/14, 1,536-dimensional image embeddings, and a timm loading recipe. Its published input example uses 224 pixels. Follow its full architecture configuration and recommended evaluation transform; do not copy Virchow2 token pooling.

Use the user's existing cluster Hugging Face credentials through the standard cache/environment. Test authenticated file access without printing tokens, environment dumps, or credential files. Token presence alone does not establish gated access. If denied, record the access error and required account action; complete other preparation while access is unresolved. Keep checkpoint files private and follow the repository's academic-use terms.

Resolve the repository to an immutable commit, download that revision once, and record the actual checkpoint SHA-256. No invented revision/hash placeholders may pass preflight. Record model kwargs, pooling, resolved transform including resize/crop/interpolation/normalization, output dimension, inference precision, storage dtype, and package versions. Pin weights locally for offline extraction jobs.

Model comparison includes each encoder's recommended transform and pooling. Use identical source pixels; do not force an undocumented common image transform. Differences cannot be attributed uniquely to architecture, pretraining corpus, dimension, or pooling. Audit public pretraining descriptions for possible dataset overlap and report any unresolved overlap; do not assert no contamination from silence.

## Environment and feasibility

Inspect `experiments/environment.def` and `requirements-experiment.txt` first. Reuse the existing SIF if its torch/timm stack can instantiate UNI2-h exactly. Otherwise build a dedicated SIF on a compute node, leaving the established environment intact. Record image hash and exact dependencies.

On `gpu-test`, run a small fixed image batch from each dataset using `eval()` and inference mode. Begin with batch size 8 on one compatible GPU, then measure safe throughput and memory. Compare repeated single-image and batched outputs. Verify shape N×1536, finite values, nonzero variance, and identical patch order. Start with float32 inference; qualify mixed precision against this reference before adopting it. Use float16 storage to match existing caches only after measuring conversion error.

For the same images, audit Virchow2's N×2560 output against its existing pipeline and cache. Store maximum absolute and relative feature differences plus logit/BA differences for a tiny fixed probe. Freeze numerical tolerances before bulk extraction; exact tensor equality across devices is not assumed.

Estimate extraction time as unique images / measured images per second, plus observed staging and write overhead. Estimate raw UNI storage as N×1536×bytes per element, then include cache metadata, temporary writes, and staging space. If all 1,608,060 TCGA-UT source images were used, float16 vectors alone would occupy about 4.94 GB decimal; this is an illustration, not the audited extraction size. Measure BRACS independently.

Exit gate: source identities resolve, requested rows exist, authenticated checkpoint access works, and a pinned model/environment passes the GPU pilot. Access failure blocks extraction, not completion of the scientific plan.
