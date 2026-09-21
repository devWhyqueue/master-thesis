# master-thesis

Controlled patch-classification studies on frozen Virchow2 features (BRACS, TCGA-UT).

## Research direction

```mermaid
flowchart TD
    S1["<b>1 · Survey the problem</b><br/>Class imbalance, data shortage,<br/>class difficulty variation in CPath<br/><i>exp. 00–01</i>"]
    S2["<b>2 · Simulate deprivations</b><br/>Controlled patch / patient / difficulty<br/>shortages on BRACS and TCGA-UT<br/><i>exp. 02</i>"]
    S3["<b>3 · Measure mitigation success</b><br/>Discrimination and calibration of<br/>common learning methods<br/><i>exp. 02–04</i>"]
    S4["<b>4 · Isolate the hard case</b><br/>Patch shortage is repairable;<br/>patient shortage is not<br/><i>exp. 05–15</i>"]
    S5["<b>5 · Explain the damage</b><br/>Few patients estimate the class<br/>distribution moments badly:<br/>centre error ~3/4, directions ~1/4<br/><i>exp. 16–18</i>"]
    S6["<b>6 · Confirm the hard limit</b><br/>Cohort-only moment repairs close<br/>at most ~15% of the 5→20 gap,<br/>none on BRACS; missing patient<br/>directions need more patients<br/><i>exp. 19–24</i>"]

    N1["<b>Next · Split the deprivations</b><br/>One small experiment per deprivation<br/>(patch / patient / difficulty shortage),<br/>one factor varied at a time; measure<br/>damage to baseline CE only"]

    S1 --> S2 --> S3 --> S4 -- "residual gap" --> S5 --> S6
    S2 -. "refactor" .-> N1

    classDef done fill:#e8f1fb,stroke:#3b6ea5,stroke-width:1px,color:#13293d;
    classDef next fill:#fff6e0,stroke:#c08a1e,stroke-width:1px,stroke-dasharray:4 3,color:#3d2a06;
    class S1,S2,S3,S4,S5,S6 done;
    class N1 next;
```

Per-experiment questions and results: [`experiments/AGENTS.md`](experiments/AGENTS.md).

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
