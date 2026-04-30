# bib

Project-local BibTeX skill and CLI for metadata enrichment, PDF-backed synchronization, screening, duplicate detection, and stable alphabetical entry sorting.

## Usage

```bash
uv run bib refresh
uv run bib refresh --dry-run
uv run bib refresh --no-sort
uv run bib refresh --input-bib papers/sources.bib --pdf-dir papers
uv run bib enrich papers/sources.bib --dry-run
uv run bib enrich papers/sources.bib --in-place
uv run bib enrich papers/sources.bib --in-place --sort
uv run bib screen papers/sources.bib --in-place
uv run bib screen papers/sources.bib --in-place --sort
uv run bib dedupe papers/sources.bib
uv run bib pdf-sync papers/sources.bib --pdf-dir papers --dry-run
uv run bib pdf-sync papers/sources.bib --out papers/sources.synced.bib --sort
```

`bib refresh` is the repo-oriented maintenance shortcut: it runs PDF sync, enrichment, duplicate reporting, and screening in one pass, then sorts entries alphabetically by BibTeX key before writing back to `papers/sources.bib` by default. Use `--no-sort` to keep the existing order. Duplicate handling remains report-only.

`bib enrich`, `bib screen`, and `bib pdf-sync` do not sort by default, but each supports `--sort` to sort entries before writing output.

The screening rubric treats strong venues as a positive signal. The default `top_tier_venues` list is non-exhaustive and includes:

- General ML and computer vision: NeurIPS, CVPR, ICML, ICLR, ICCV, ECCV, AISTATS
- Medical imaging: MICCAI, ISBI

## Screening fields

```bibtex
x_screening_bucket = {review}
x_screening_score = {0.75}
x_screening_reason = {review suggested: no resolvable external match found}
x_screening_details = {doi=None; matched=False; venue=OpenReview; ...}
```

## PDF sync handoff

`bib pdf-sync` is intentionally best-effort. It should extract and resolve metadata for the easy majority of PDFs, but it should not force low-confidence guesses for the remaining cases.

When a PDF cannot be matched or created confidently, the command reports it as `needs_review` and includes the strongest locally extracted metadata it has available, such as title, authors, year, DOI, or venue. The calling agent is expected to use that payload to finish the hard cases with direct PDF inspection, OCR, or web lookup.

## Config

Use `examples/config.yaml` as the starting point for provider toggles, cache settings, resolution thresholds, canonical field overwrite rules, PDF sync settings, and screening settings.
