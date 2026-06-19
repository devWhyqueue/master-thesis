# master-thesis

Master's thesis: class-imbalance mitigation in computational pathology, using frozen Virchow2 WSI feature bags on TCGA-UT.

## Key paths

| Path | Purpose |
|---|---|
| `experiments/class_imbalance/` | Active experiment — baselines + imbalance methods + GAN synthetic feature bridge |
| `experiments/class_imbalance/report/main.pdf` | Class-imbalance paper draft |
| `experiments/design_dataset/` | Dataset design experiment (code, report, visualizations) |
| `experiments/design_dataset/report/main.pdf` | Dataset design paper draft |
| `experiments/shared/` | Common code and tests shared across experiments |
| `papers/sources.bib` | Main bibliography (local `file` paths + screening metadata) |
| `papers/` | PDFs organised by topic |
| `docs/` | Thesis reference PDFs, glossary, FAQ |
| `meetings/` | Meeting notes by date |
| `CLUSTER.md` | Hydra cluster runbook (SSH, SLURM, storage) |
| `CLAUDE.md` / `AGENTS.md` | Agent instructions (Claude / OpenAI-compatible) |
| `.agents/skills/` | Repo-local agent skills |

## Setup

```bash
uv sync   # Python 3.10+, installs package + exposes `bib` CLI
```

## Bibliography commands

```bash
uv run bib enrich papers/sources.bib --dry-run
uv run bib enrich papers/sources.bib --in-place
uv run bib screen papers/sources.bib --in-place
uv run bib dedupe papers/sources.bib
uv run bib pdf-sync papers/sources.bib --pdf-dir papers --dry-run
uv run bib pdf-sync papers/sources.bib --pdf-dir papers --in-place
```

`pdf-sync` is conservative: low-confidence matches are flagged for manual review.

## Development

```bash
uv run pytest .agents/skills/bib/tests
uv run pyright
uv run ruff check .
```

## Repo-local skills

- `bib` — bibliography: enrich, screen, dedupe, pdf-sync
- `scientific-writing` — LaTeX prose, captions, tables, equations
- `notebooklm` — automate NotebookLM notebooks and source uploads
- `hydra-cluster` — cluster job submission and dataset workflows

## Experiments

See each experiment's own `README.md` for the Hydra runbook:
- [`experiments/class_imbalance/README.md`](./experiments/class_imbalance/README.md)
- [`experiments/design_dataset/`](./experiments/design_dataset/)
