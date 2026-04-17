# master-thesis

Working repository for a master's thesis on computational pathology. The repo combines a local paper library under `papers/` with a small Python toolchain for keeping `papers/sources.bib` consistent, enriched, and reviewable.

## What is in this repo

- `papers/`: PDFs and the main bibliography file `papers/sources.bib`
- `docs/`: thesis-related reference material and notes
- `.agents/skill/bib/`: the project-local `bib` package, examples, tests, and references
- `code/TCGA-UT/`: external or supporting code used during the thesis work

The main maintained software surface in this repo is the `bib` CLI exposed by this package:

- enrich BibTeX entries from Crossref/OpenAlex
- score and tag entries for screening
- detect likely duplicates
- sync local PDFs with BibTeX entries and create missing records when confidence is high enough

## Setup

This project uses `uv` and requires Python 3.10+.

```bash
uv sync
```

That installs the package and exposes the `bib` command through `uv run`.

## Quick Start

Preview metadata enrichment:

```bash
uv run bib enrich papers/sources.bib --dry-run
```

Write enrichment results back to the bibliography:

```bash
uv run bib enrich papers/sources.bib --in-place
```

Screen entries and write `x_screening_*` fields:

```bash
uv run bib screen papers/sources.bib --in-place
```

Report likely duplicate entries:

```bash
uv run bib dedupe papers/sources.bib
```

Match PDFs under `papers/` to existing entries and create missing ones when confidence is sufficient:

```bash
uv run bib pdf-sync papers/sources.bib --pdf-dir papers --dry-run
uv run bib pdf-sync papers/sources.bib --pdf-dir papers --in-place
```

## Configuration

All commands accept an optional YAML config file:

```bash
uv run bib enrich papers/sources.bib --config path/to/config.yaml --in-place
```

Use [.agents/skill/bib/examples/config.yaml](./.agents/skill/bib/examples/config.yaml) as the starting point. It covers:

- provider toggles and timeouts
- cache location and TTL
- enrichment overwrite rules
- PDF sync thresholds
- screening weights and bucket rules

By default, online enrichment uses Crossref and OpenAlex. You can disable network-backed resolution per command with:

```bash
uv run bib enrich papers/sources.bib --disable-online-enrichment --dry-run
```

## Development

The Python package is defined in [pyproject.toml](./pyproject.toml) and lives under [.agents/skill/bib/src](./.agents/skill/bib/src). Tests live in [.agents/skill/bib/tests](./.agents/skill/bib/tests).

Useful commands:

```bash
uv run pytest .agents/skill/bib/tests
uv run pyright
uv run ruff check .
```

## Notes

- `papers/sources.bib` already stores local `file` paths and screening metadata such as `x_screening_bucket`.
- `bib pdf-sync` is intentionally conservative. Low-confidence cases are reported for manual review rather than forced into the bibliography.
- There is a narrower package-level README at [.agents/skill/bib/README.md](./.agents/skill/bib/README.md) if you only need details for the `bib` tool itself.
