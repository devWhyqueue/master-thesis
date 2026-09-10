# Report evidence

The `bracs/` and `tcga_ut/` folders contain unchanged analysis summaries (`influence.json`) and allocation audits (`preflight.json`) downloaded from Hydra:

`/home/yannik.qu/master-thesis/experiments/4_patient_influence/outputs/<dataset>/patch/`

The analysis summaries came from `tables/`; the allocation audits came from `data/`.

- `source_sha256.txt`: remote hashes checked against the downloaded summaries and the local weighting, contrast-analysis, and logistic-regression code.
- `fit_status.txt`: convergence, weighting, regularization, and iteration metadata for all twelve patient-average fits, plus successful audit/analysis job statuses. BRACS analysis job: 4881249; TCGA-UT analysis job: 4881254.
- `patient_contributions.csv`: all 222 dataset/class/split/support contribution records. `D_c` and `max_share` are fractions, not percentages.
- `class_recall.csv`: all 74 dataset/class/support summaries. Each objective's recall is averaged equally over the three split-level patient-macro class recalls and expressed as a percentage; gains are percentage-point differences.

The LaTeX results tables summarize these records. Primary intervals are the saved paired Bayesian-bootstrap intervals; class-recall supplements contain descriptive point estimates only. No training or analysis jobs were launched while completing the report.
