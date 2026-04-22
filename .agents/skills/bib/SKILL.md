---
name: bib
description: Work with BibTeX and .bib files for enrichment, metadata repair, screening, duplicate detection, and PDF-backed bibliography maintenance. Use when Codex needs to resolve missing DOI or venue metadata, normalize bibliography records, sync local PDFs into a BibTeX file, triage papers into keep/review/exclude buckets, or inspect duplicate candidates without corrupting the BibTeX file.
---

# Bib

## Overview

Use the local `bib` CLI for general BibTeX workflows. The default thesis-repo maintenance flow is `refresh`, which runs `pdf-sync`, `enrich`, `dedupe`, and `screen` in one pass. The main subcommands are `refresh`, `enrich`, `screen`, `dedupe`, and `pdf-sync`.

Keep enrichment conservative: only write canonical fields when a provider match is confident enough. Weak or ambiguous matches should remain unchanged.

## Commands

```bash
uv run bib refresh
uv run bib refresh --dry-run
uv run bib refresh --input-bib papers/sources.bib --pdf-dir papers
uv run bib enrich papers/sources.bib --dry-run
uv run bib enrich papers/sources.bib --in-place
uv run bib screen papers/sources.bib --out papers/sources.screened.bib
uv run bib dedupe papers/sources.bib
uv run bib pdf-sync papers/sources.bib --pdf-dir papers --dry-run
```

## Behavior

- Preserve unknown BibTeX fields.
- Use validated temp-file writes for all mutations.
- `bib refresh` defaults to mutating `papers/sources.bib` in place and scanning PDFs under `papers/`.
- Resolve metadata from DOI when present, otherwise search by title, year, URL, and venue.
- Sync local PDFs into BibTeX entries and write a relative `file` field.
- `bib refresh` reports duplicates, but does not merge or remove them automatically.
- Treat `pdf-sync` as best-effort automation: it should resolve straightforward PDFs directly, but leave hard cases for the calling agent instead of forcing brittle guesses.
- When `pdf-sync` cannot confidently match or create an entry, it reports `needs_review` together with the best extracted local metadata and stopping reason so the calling agent can inspect the PDF, use web lookup, or apply OCR as needed.
- Keep `x_screening_*` fields owned by `bib screen` only.

## Resources

- Implementation: `src/bib/`
- Config example: `examples/config.yaml`
- Field conventions: `references/field-conventions.md`
- Screening notes: `references/scoring.md`
