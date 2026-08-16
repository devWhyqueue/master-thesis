# master-thesis

Master's thesis: controlled class-imbalance mitigation in computational pathology, using frozen Virchow2 features.

## Key paths

| Path | Purpose |
|---|---|
| `experiments/2_benchmark_patch/` | Unified controlled class-imbalance benchmark |
| `experiments/2_benchmark_patch/code/` | Importable benchmark package and test suite |
| `experiments/2_benchmark_patch/report/main.pdf` | Controlled-benchmark report |
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
uv run pytest experiments/2_benchmark_patch/code/tests
uv run pyright
uv run ruff check .
```

## Repo-local skills

- `bib` — bibliography: enrich, screen, dedupe, pdf-sync
- `scientific-writing` — LaTeX prose, captions, tables, equations
- `notebooklm` — automate NotebookLM notebooks and source uploads
- `hydra-cluster` — cluster job submission and dataset workflows

## Experiments

The unified benchmark CLI is available after `uv sync`:

```bash
uv run python experiments/2_benchmark_patch/code/__main__.py --help
uv run python experiments/2_benchmark_patch/code/__main__.py submit --dry-run
```
