# bib

Project-local BibTeX skill and CLI for metadata enrichment, PDF-backed synchronization, screening, and duplicate detection.

## Usage

```bash
uv run bib enrich papers/sources.bib --dry-run
uv run bib enrich papers/sources.bib --in-place
uv run bib screen papers/sources.bib --in-place
uv run bib dedupe papers/sources.bib
uv run bib pdf-sync papers/sources.bib --pdf-dir papers --dry-run
```

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
