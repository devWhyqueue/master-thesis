# Phase 05 execution log

Status: complete, 2026-09-25. Times and peak RSS below are from `sacct`. Main numerical results are kept in analysis artifacts rather than interpreted here.

| Dataset | Stage | Job ID | State | Elapsed | Peak RSS | Note |
| --- | --- | ---: | --- | ---: | ---: | --- |
| BRACS | UNI2-h extraction | 4921278_0 | completed | 27:21 | 16,564,600K | 376 slide tensors |
| BRACS | merge | 4921279 | completed | 2:10 | 2,515,320K | Main cache merged |
| BRACS | first feature audit | 4921280 | failed | 13:45 | 2,717,640K | Numeric case IDs in source pointers did not match string keys |
| BRACS | repaired feature audit | 4923095 | completed | 8:18 | 4,994,028K | 168,585 requested patches; zero unresolved |
| BRACS | corrected scope preflight | 4923124 | completed | 0:24 | 1,447,280K | Signed amendment checked |
| BRACS | initial reserved draw pilots | 4923128, 4923129 | failed | 1:22, 0:59 | 6,848,268K, 4,747,924K | Draw 10000 needed training references outside the main cache |
| BRACS | separate pilot extraction | 4923141 | completed | 2:56 | 15,342,580K | Does not mutate main cache |
| BRACS | Virchow2 pilot | 4923142 | completed | 2:17 | 6,900,724K | Seven arms, 11 candidates each |
| BRACS | UNI2-h pilot | 4923143 | completed | 1:38 | 4,376,084K | Seven arms, 11 candidates each |
| TCGA-UT | UNI2-h extraction | 4921752_0–7 | completed | 0:38:56–1:57:46 | 7,468,184K–7,892,324K observed | Eight unthrottled shards |
| TCGA-UT | merge | 4921753 | completed | 14:53 | 9,155,132K | Main cache merged |
| TCGA-UT | first feature audit | 4921754 | out of memory | 56:58 | 16,775,492K | Audit retained slide tensors; repaired cache eviction and memory request |
| TCGA-UT | repaired feature audit | 4923097 | completed | 1:26:00 | 4,713,860K | 1,201,809 requested patches; zero unresolved |
| TCGA-UT | corrected scope preflight | 4923098 | completed | 0:35 | 1,446,420K | Signed amendment checked |
| TCGA-UT | separate pilot extraction | 4923159 | completed | 23:14 | 7,077,948K | Does not mutate main cache |
| TCGA-UT | Virchow2 pilot | 4923160 | completed | 16:40 | 25,511,712K | Seven arms, 11 candidates each |
| TCGA-UT | UNI2-h pilot | 4923161 | completed | 10:06 | 16,034,596K | Seven arms, 11 candidates each |
| BRACS | full main shard probe | 4923724 | completed | 4:57 | 5,949,872K | Split 0, draw 10, Virchow2; seven locked records |
| BRACS | remaining main fit array | 4923736_1–59 | completed | individual times in `sacct` | individual peaks in `sacct` | No concurrency throttle or failed shard |
| BRACS | fit completeness audit | 4923867 | completed | 0:33 | 1,447,800K | 420 expected, 420 valid, zero missing |
| BRACS | first analysis | 4923868 | failed | 2:20 | 6,452,460K | Sensitivity switched feature dimensions without clearing the process-local feature bank |
| BRACS | analysis with per-encoder bank reset | 4923884 | completed | 3:21 | 6,454,028K | Accepted analysis and diagnostics written; fit source lock preserved |
| BRACS | final-wrapper analysis repeat | 4924030 | completed | 2:48 | 6,455,216K | Analysis, diagnostics, and distribution files match job 4923884 byte for byte |
| TCGA-UT | full main shard probe | 4923727 | completed | 18:16 | 24,866,060K | Split 0, draw 10, Virchow2; seven locked records |
| TCGA-UT | remaining main fit array | 4923798_1–59 | completed | individual times in `sacct` | individual peaks in `sacct` | No concurrency throttle or failed shard |
| TCGA-UT | fit completeness audit | 4923914 | completed | 1:04 | 1,449,120K | 420 expected, 420 valid, zero missing |
| TCGA-UT | analysis | 4923915 | completed | 30:47 | 34,614,052K | `afterok` dependency on 4923914; accepted analysis and diagnostics written |

Independent scope check 4923112 completed in 44 seconds: phase-02 `input_audit.json` counted test rows but omitted validation rows from its claimed validation-plus-test union. The signed `configs/phase05_scope_amendment.json` records corrected BRACS (168,585) and TCGA-UT (1,201,809) requested patch unions. The locked sampling schedule, readout grid, and inference plan did not change.

BRACS pilot check: all 14 selected arm fits completed on split 0/draw 10000, and all 154 lambda candidates reported convergence. Selected lambdas stayed within the frozen 11-value grid. Pilot test outcomes are excluded from main inference.

BRACS main fit request selected from the pilot: `cpu-2h`, 4 CPUs, 16 GB per shard. The pilot used 16 CPUs/64 GB. Its initial failed jobs completed the B arm before missing references stopped a later arm; the resumed jobs completed the other six. Combined elapsed times were 3:39 for Virchow2 and 2:37 for UNI2-h, with peak RSS below 7 GB on fit jobs. The smaller CPU request trades some throughput for greater schedulability while retaining a two-hour wall limit. No array concurrency throttle is configured.

TCGA-UT pilot check: all 14 selected arm fits and all 154 lambda candidates converged. UNI2-h R100 selected the lower grid bound, `1e-8`, with validation macro recall 0.7025388 versus 0.7025316 at `1e-7`; this near plateau does not require a grid expansion. TCGA-UT main fit request: `cpu-2h`, 8 CPUs, 48 GB per shard, based on the pilot's 16:40 maximum and 25.6 GB peak at 16 CPUs/64 GB. No array concurrency throttle is configured.

Extraction/resource measurements: UNI2-h used an eight-image batch, float32 output and no inference autocast, with one GPU per task. BRACS ran on SLURM GPU label `blackwell` (head057); TCGA-UT's eight shards ran on node labels `h100`, `blackwell`, `3090`, and `80gb`. `sacct` GPU-memory maxima were 3,610 MB for BRACS and 3,228–3,610 MB for TCGA-UT. BRACS extracted 168,585 requested patches in 27:21 (103 patches per GPU wall second end to end); TCGA-UT extracted 1,201,809 over 34,423 summed GPU wall seconds (35 patches per GPU wall second end to end). These rates include setup and staging; the jobs did not separately instrument staging or embedding time. Main UNI2-h caches occupy 1,036,618,747 bytes for BRACS and 7,404,349,153 bytes for TCGA-UT; separate pilot caches occupy 21,211,610 and 176,966,902 bytes. `sacct` host peak RSS and complete per-task resource counters will be retained in the archive. Candidate optimization time was not isolated from data loading; full seven-arm probe wall times were 4:57 BRACS and 18:16 TCGA-UT.

The BRACS audit validates all 420 selected-arm records. Its first analysis reached the B-selected-lambda sensitivity and failed because the global feature bank retained Virchow2-width rows when UNI2-h rows were loaded. `analysis_reset.py` clears that bank before each encoder's sensitivity pass. It lives outside `code/`, whose bytes are locked into the completed fit records; analysis job 4923884 completed with the original fit source lock. The wrapper was then cleaned up, and job 4924030 produced identical SHA-256 hashes for `analysis.json`, `diagnostics.json`, and `distributions.npz`. Both datasets therefore use the final archived wrapper. The BRACS artifact has 10 draws per split, 41 estimates, 24 probability-quality estimates, both encoder sensitivities, finite reported estimates, and zero test-class support rejections.

TCGA-UT's sparse main array completed without failed tasks. SLURM rejected an `afterok` dependency on the array job ID, so the audit was submitted only after `sacct` showed probe 4923727 and all 59 array tasks `COMPLETED|0:0`. Analysis is dependent on the audit. The `slurm`-only audit/analysis requests (32 GB and 96 GB) do not change the locked fit config hash.

Final acceptance: both dataset audits report 420 expected/420 valid selected-arm records and no missing records, for 840/840 overall. Both analyses completed with 10 draws per split, 41 estimates, 24 probability-quality estimates, both encoder sensitivities, finite analysis/diagnostic numbers, and zero test-class support rejections. Candidate evidence is covered by the fit audit. No optional stopping or outcome-based grid change occurred. The result archives under each dataset's output root contain the source snapshot, signed locks, `sacct` ledger, job logs, and SHA-256 checksums for the accepted analysis artifacts. BRACS archive: 226 `sacct` rows and 150 logs; TCGA-UT archive: 235 rows and 156 logs. The ignored output trees are copied locally after checksum verification; no results are committed by this phase.
