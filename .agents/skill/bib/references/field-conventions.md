# Field Conventions

`bib screen` adds only custom `x_screening_*` fields and preserves unknown fields already present in the BibTeX entry. `bib enrich` updates canonical BibTeX fields directly when a confident match is found.

## Added fields

- `x_screening_bucket`: `keep`, `review`, or `exclude`
- `x_screening_score`: numeric composite score
- `x_screening_reason`: short human-readable rationale
- `x_screening_details`: compact JSON-like detail payload

## Canonical updates

`bib enrich` may update:

- `doi`
- `url`
- `journal` or `booktitle`
- `publisher`
- `volume`
- `number`
- `pages`
- `year`
- `title`
- `file`

## Reruns

On rerun, the tool replaces prior `x_screening_*` values instead of appending duplicates.
