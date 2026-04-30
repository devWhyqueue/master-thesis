# Scoring Notes

`bib screen` uses conservative triage semantics:

- `keep`: clear positive signals
- `review`: incomplete, ambiguous, or mixed evidence
- `exclude`: strong negative signals, especially retractions

Signals considered by default:

- venue strength
- citation traction normalized by paper age
- publication type
- retraction or update signals
- metadata completeness and match confidence
- duplicate candidates by DOI or normalized title

Default positive venue signals are non-exhaustive and currently include:

- General ML and computer vision: NeurIPS, CVPR, ICML, ICLR, ICCV, ECCV, AISTATS
- Medical imaging: MICCAI, ISBI

Reasons should be interpreted as triage suggestions, not objective quality claims.
