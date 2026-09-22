# 01_benchmark

Controlled patch-classification benchmark establishing whether allocation damages accuracy and which mitigation recovers it. See `../AGENTS.md` for the shared code/configs/tests/report layout.

| # | Study | Question | Result |
|---|---|---|---|
| 02 | benchmark_patch | Controlled benchmark (MLP readout): does allocation damage accuracy, which mitigation recovers it? | Nominal patch imbalance recovered by prevalence/support weighting. Reduced patient coverage leaves replicated discrimination deficit no method recovers. Calibration needs separate treatment. |
