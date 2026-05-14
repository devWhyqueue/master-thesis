# master-thesis

Working repository for a master's thesis on computational pathology. The repo
combines literature material, thesis notes, and experiment code for
class-imbalance mitigation in computational pathology.

## What is in this repo

- `papers/`: PDFs and the main bibliography file `papers/sources.bib`
- `docs/`: thesis-related reference material and notes
- `experiments/class_imbalance_tcga_ut/`: native TCGA-UT class-imbalance
  experiment, paper draft, Hydra scripts, and generated paper tables/figures
- `.agents/skills/`: repo-local agent skills for thesis workflows
- `code/TCGA-UT/`: external or supporting code used during the thesis work

The active experiment package is
[`experiments/class_imbalance_tcga_ut/`](./experiments/class_imbalance_tcga_ut/).
It evaluates baselines and representative imbalance methods on frozen Virchow2
WSI feature bags, including an image-GAN-to-Virchow2 synthetic feature bridge.

## Setup

This project uses `uv` and requires Python 3.10+.

```bash
uv sync
```

That installs the package and exposes the `bib` command through `uv run`.

## Repo-Local Skills

- `bib`: maintain `papers/sources.bib`, enrich metadata, sync PDFs, screen
  papers, and detect duplicates.
- `scientific-writing`: revise paper/thesis prose, LaTeX structure, captions,
  tables, equations, and research figures.
- `notebooklm`: automate NotebookLM notebooks, sources, chats, generated
  artifacts, and downloads.

Common bibliography commands:

```bash
uv run bib enrich papers/sources.bib --dry-run
uv run bib enrich papers/sources.bib --in-place
uv run bib screen papers/sources.bib --in-place
uv run bib dedupe papers/sources.bib
uv run bib pdf-sync papers/sources.bib --pdf-dir papers --dry-run
uv run bib pdf-sync papers/sources.bib --pdf-dir papers --in-place
```

## Experiment

See
[`experiments/class_imbalance_tcga_ut/README.md`](./experiments/class_imbalance_tcga_ut/README.md)
for the concise Hydra runbook. The latest paper draft is:

- `experiments/class_imbalance_tcga_ut/paper/main.pdf`

## Development

```bash
uv run pytest .agents/skills/bib/tests
uv run pyright
uv run ruff check .
```

## Notes

- `papers/sources.bib` already stores local `file` paths and screening metadata such as `x_screening_bucket`.
- `bib pdf-sync` is conservative: low-confidence PDF matches are reported for manual review.
- Cluster usage notes are in `CLUSTER.md`.
