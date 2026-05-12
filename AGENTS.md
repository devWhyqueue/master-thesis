# Commands

- `Update papers`: Look for uncommited Git changes, run compress command of skill pdf for these and bib refresh command of skill bib.

# Cluster

- Read `CLUSTER.md` before using the TU Berlin Hydra cluster. It contains the SSH shortcut, SLURM workflow, storage rules, and dataset safety notes.

# Code

- If editing Python code, run clean-code skill after a change. To avoid producing violations in the first place, look into the skill's clean_code_rules.yml.

# LaTeX

- TeX Live is installed on Windows and can be used from WSL via `/mnt/c/texlive/2026/bin/windows/pdflatex.exe` and `/mnt/c/texlive/2026/bin/windows/bibtex.exe`.
- Compile from the directory containing the `.tex` file. Use a temporary build directory and copy out only the final PDF, for example:
  - `mkdir -p .latex-build`
  - `cp *.bib .latex-build/`
  - `/mnt/c/texlive/2026/bin/windows/pdflatex.exe -interaction=nonstopmode -halt-on-error -output-directory=.latex-build main.tex`
  - `(cd .latex-build && /mnt/c/texlive/2026/bin/windows/bibtex.exe main)`
  - `/mnt/c/texlive/2026/bin/windows/pdflatex.exe -interaction=nonstopmode -halt-on-error -output-directory=.latex-build main.tex`
  - `/mnt/c/texlive/2026/bin/windows/pdflatex.exe -interaction=nonstopmode -halt-on-error -output-directory=.latex-build main.tex`
  - `cp .latex-build/main.pdf main.pdf && rm -rf .latex-build`
- Do not leave LaTeX auxiliary files in the worktree.

# NotebookLM

Notebook IDs for `papers/` imports:

| Papers folder | Notebook ID | Ready sources |
| --- | --- | ---: |
| `papers/CPath` | `fed9a50a-3954-419c-a5da-6b1631ad0cab` | 6 |
| `papers/applications` | `4283f4c7-55e1-4756-bdb0-462cb1be5c33` | 1 |
| `papers/calibration` | `aeaaa9d1-8c87-41d9-b873-47966bc82ee5` | 1 |
| `papers/class imbalance` | `24007b80-2757-4af1-bb58-6b5bdb296748` | 11 |
| `papers/datasets` | `3abadafc-3bcc-43c7-a074-e16837ff4e43` | 6 |
| `papers/few-example` | `0815c8cd-d39e-4fa5-bf11-fed6ec03512b` | 9 |
| `papers/foundation models` | `0316f1d3-4623-4069-ba7e-17a7c9af4688` | 15 |
| `papers/methods` | `d36a4eb6-b28a-40d5-97c4-915ec3515466` | 7 |
| `papers/tools` | `cb82e977-5d4d-4982-890b-a07103ff9e0f` | 1 |
| `papers` (all papers) | `f342aa91-2900-429e-b3a9-26c2e25a4c23` | 57 |
