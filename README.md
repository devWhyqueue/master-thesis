# master-thesis

Controlled patch-classification studies on frozen Virchow2 features (BRACS, TCGA-UT).

## Research direction

```mermaid
flowchart TD
    S1["<b>00_survey</b><br/>Class imbalance, data shortage,<br/>class difficulty variation in CPath<br/><i>exp. 00–01</i>"]
    S2["<b>01_benchmark</b><br/>Controlled patch / patient / difficulty<br/>shortages on BRACS and TCGA-UT<br/><i>exp. 02</i>"]
    S3["<b>02_patient_shortage/00_damage</b><br/>Breadth beats depth at fixed patch<br/>budget; effective support explains<br/>BRACS, TCGA-UT keeps a residual<br/><i>exp. 05</i>"]
    S4["<b>02_patient_shortage/01_cause</b><br/>Patch shortage is repairable; patient<br/>shortage is not — few patients estimate<br/>class-distribution moments badly<br/>(centre error ~3/4, directions ~1/4)<br/><i>exp. 03–04, 06–18</i>"]
    S5["<b>02_patient_shortage/02_mitigation</b><br/>Cohort-only moment repairs close<br/>at most ~15% of the 5→20 gap,<br/>none on BRACS; missing patient<br/>directions need more patients<br/><i>exp. 19–24</i>"]

    S7["<b>03_class_imbalance/00_damage</b><br/>Fixed patients and budget: BA loses<br/>1 pp at ρ≈20, 2.9 pp at ρ=100 (TCGA-UT),<br/>1 pp at ρ=2, 7.7 pp at ρ=100 (BRACS);<br/>temperature scaling repairs calibration<br/><i>exp. 25–26</i>"]
    N1["<b>03_class_imbalance/01_cause</b><br/>Split the ratio arm into class-prior<br/>and patch-support components,<br/>one factor at a time<br/><i>exp. 27–28</i>"]

    S1 --> S2 --> S3 --> S4 --> S5
    S2 -.-> S7 --> N1

    classDef done fill:#e8f1fb,stroke:#3b6ea5,stroke-width:1px,color:#13293d;
    classDef next fill:#fff6e0,stroke:#c08a1e,stroke-width:1px,stroke-dasharray:4 3,color:#3d2a06;
    class S1,S2,S3,S4,S5,S7 done;
    class N1 next;
```

Per-experiment questions and results: [`experiments/AGENTS.md`](experiments/AGENTS.md).

## Key paths

| Path | Purpose |
|---|---|
| `experiments/` | Experiments grouped by research question and reports (summary: `experiments/AGENTS.md`) |
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
