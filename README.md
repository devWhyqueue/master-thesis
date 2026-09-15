# master-thesis

Master's thesis: controlled class-imbalance mitigation in computational pathology, using frozen Virchow2 features.

## Key paths

| Path | Purpose |
|---|---|
| `experiments/` | Numbered experiments and reports (summary: `experiments/CLAUDE.md`) |
| `papers/` | PDFs by topic, `sources.bib` bibliography |
| `docs/` | Thesis reference PDFs, glossary, FAQ |
| `meetings/` | Meeting notes by date |
| `CLUSTER.md` | Hydra cluster runbook (SSH, SLURM, storage) |
| `CLAUDE.md` / `AGENTS.md` | Agent instructions |
| `.agents/skills/` | Repo-local agent skills (`bib`, `scientific-writing`, `notebooklm`, `hydra-cluster`) |

## Setup

```bash
uv sync   # Python 3.12+, installs package + exposes `bib` CLI
```
