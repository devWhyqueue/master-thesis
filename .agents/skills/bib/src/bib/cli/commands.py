"""CLI command implementations for BibTeX workflows."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..bibtex import (
    append_bibtex_entries,
    load_bibtex_file,
    parse_bibtex,
    strip_screening_updates,
    update_bibtex_fields,
    write_output,
)
from ..config import BibConfig, load_config
from ..metadata import (
    CrossrefProvider,
    OpenAlexProvider,
    ResponseCache,
    detect_duplicates,
    render_duplicates,
    render_enrichment_preview,
    render_enrichment_summary,
    render_pdf_sync_preview,
    render_pdf_sync_summary,
    render_screening_summary,
    resolve_cache_paths,
    resolve_entry,
    score_entry,
    screening_updates,
)
from ..models import EnrichmentPreview, ScreenedEntry
from ..pdf import (
    PdfSyncResult,
    build_partial_entry,
    creation_confidence,
    create_new_entry,
    discover_pdfs,
    extract_pdf_metadata,
    match_existing_entry,
)
from .parser import build_parser

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Run the bib CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in {"enrich", "screen", "pdf-sync"}:
        _validate_write_args(parser, args)
    if args.command == "refresh":
        args.input_bib = Path(args.input_bib)
    config = load_config(getattr(args, "config", None))
    if getattr(args, "disable_online_enrichment", False):
        config.crossref.enabled = False
        config.openalex.enabled = False
    if getattr(args, "cache_dir", None) is not None:
        config.cache.dir = str(args.cache_dir)
    if args.command == "refresh":
        return _run_refresh(args, config)
    if args.command == "enrich":
        return _run_enrich(args, config)
    if args.command == "screen":
        return _run_screen(args, config)
    if args.command == "pdf-sync":
        return _run_pdf_sync(args, config)
    return _run_dedupe(args)


def _validate_write_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.in_place and args.out is not None:
        parser.error("--in-place cannot be combined with --out")
    if getattr(args, "dry_run", False):
        return
    if not args.in_place and args.out is None and args.command != "dedupe":
        parser.error("either --out or --in-place is required")


def _canonical_updates(entry, resolved, config: BibConfig) -> dict[str, str | None]:
    updates: dict[str, str | None] = {}
    for field in config.enrichment.overwrite_fields:
        value = getattr(resolved, field, None)
        if value is not None:
            updates[field] = str(value)
    if config.enrichment.canonicalize_title and resolved.title:
        updates["title"] = resolved.title
    if not updates and entry.title and resolved.title and entry.title != resolved.title:
        updates["title"] = resolved.title
    return updates


def _build_runtime(config: BibConfig) -> tuple[ResponseCache, list[object]]:
    cache = ResponseCache(resolve_cache_paths(Path.cwd(), config.cache), config.cache)
    providers = [CrossrefProvider(config.crossref), OpenAlexProvider(config.openalex)]
    return cache, providers


def _enrich_document(
    text: str, entries, config: BibConfig
) -> tuple[str, list[EnrichmentPreview], int]:
    cache, providers = _build_runtime(config)
    previews: list[EnrichmentPreview] = []
    updates_by_key: dict[str, dict[str, str | None]] = {}
    changed_count = 0
    for entry in entries:
        resolution = resolve_entry(entry, providers, cache, config)
        updates = _canonical_updates(entry, resolution.metadata, config)
        preview_updates = {
            field: value for field, value in updates.items() if value is not None
        }
        previews.append(
            EnrichmentPreview(
                entry.key, resolution.confidence, preview_updates, resolution.reasons
            )
        )
        updates_by_key[entry.key] = updates
        changed_count += int(bool(updates))
    return update_bibtex_fields(text, updates_by_key), previews, changed_count


def _screen_document(text: str, entries, config: BibConfig) -> tuple[str, list[ScreenedEntry]]:
    duplicates = detect_duplicates(entries)
    cache, providers = _build_runtime(config)
    screened_entries: list[ScreenedEntry] = []
    updates_by_key: dict[str, dict[str, str | None]] = {}
    for entry in entries:
        resolution = resolve_entry(entry, providers, cache, config)
        score = score_entry(entry, resolution, duplicates, config)
        screened_entries.append(ScreenedEntry(entry, resolution, score))
        updates = strip_screening_updates()
        updates.update(
            screening_updates(
                score,
                config.screening.output.include_details,
                config.screening.output.details_format,
            )
        )
        updates_by_key[entry.key] = updates
    return update_bibtex_fields(text, updates_by_key), screened_entries


def _pdf_sync_document(
    text: str, entries, input_bib: Path, pdf_dir: Path, config: BibConfig
) -> tuple[str, list[PdfSyncResult]]:
    existing_keys = {entry.key for entry in entries}
    cache, providers = _build_runtime(config)
    results: list[PdfSyncResult] = []
    updates_by_key: dict[str, dict[str, str | None]] = {}
    new_entries: list[dict[str, object]] = []
    for pdf_path in discover_pdfs(pdf_dir):
        extracted = extract_pdf_metadata(pdf_path, input_bib, config)
        matched_entry, match_confidence, reasons = match_existing_entry(
            extracted, entries, config
        )
        if matched_entry is not None:
            partial_entry = build_partial_entry(extracted)
            resolution = resolve_entry(partial_entry, providers, cache, config)
            updates = _canonical_updates(matched_entry, resolution.metadata, config)
            if extracted.title and not matched_entry.title:
                updates["title"] = extracted.title
            if extracted.year is not None and not matched_entry.year:
                updates["year"] = str(extracted.year)
            if extracted.doi and not matched_entry.doi:
                updates["doi"] = extracted.doi
            if extracted.url and not matched_entry.url:
                updates["url"] = extracted.url
            if extracted.venue and not matched_entry.venue:
                updates[
                    "booktitle"
                    if matched_entry.entry_type == "inproceedings"
                    else "journal"
                ] = extracted.venue
            updates["file"] = extracted.relative_pdf_path
            updates_by_key.setdefault(matched_entry.key, {}).update(updates)
            results.append(
                PdfSyncResult(
                    extracted,
                    matched_entry.key,
                    match_confidence,
                    None,
                    updates,
                    None,
                    reasons,
                )
            )
            continue
        partial_entry = build_partial_entry(extracted)
        resolution = resolve_entry(partial_entry, providers, cache, config)
        new_entry = create_new_entry(
            extracted, resolution.metadata, existing_keys, config
        )
        created_key = None
        if new_entry is not None:
            created_key = str(new_entry["key"])
            existing_keys.add(created_key)
            new_entries.append(new_entry)
        confidence = (
            creation_confidence(extracted, resolution.metadata)
            if new_entry is not None
            else max(match_confidence, resolution.confidence)
        )
        results.append(
            PdfSyncResult(
                extracted,
                None,
                confidence,
                created_key,
                {},
                new_entry,
                reasons or resolution.reasons,
            )
        )
    return append_bibtex_entries(update_bibtex_fields(text, updates_by_key), new_entries), results


def _run_enrich(args: argparse.Namespace, config: BibConfig) -> int:
    text, entries = load_bibtex_file(args.input_bib)
    updated, previews, changed_count = _enrich_document(text, entries, config)
    logger.info(render_enrichment_summary(previews, changed_count))
    preview_text = render_enrichment_preview(previews)
    if preview_text:
        logger.info(preview_text)
    if args.dry_run:
        return 0
    if args.in_place:
        write_output(args.input_bib, updated, in_place_target=args.input_bib)
    else:
        write_output(args.out, updated)
    return 0


def _run_screen(args: argparse.Namespace, config: BibConfig) -> int:
    text, entries = load_bibtex_file(args.input_bib)
    updated, screened_entries = _screen_document(text, entries, config)
    logger.info(render_screening_summary(screened_entries))
    if args.dry_run:
        return 0
    if args.in_place:
        write_output(args.input_bib, updated, in_place_target=args.input_bib)
    else:
        write_output(args.out, updated)
    return 0


def _run_dedupe(args: argparse.Namespace) -> int:
    _, entries = load_bibtex_file(args.input_bib)
    logger.info(render_duplicates(detect_duplicates(entries)))
    return 0


def _run_pdf_sync(args: argparse.Namespace, config: BibConfig) -> int:
    text, entries = load_bibtex_file(args.input_bib)
    pdf_dir = args.pdf_dir or Path(config.pdf_sync.pdf_dir)
    updated, results = _pdf_sync_document(text, entries, args.input_bib, pdf_dir, config)
    logger.info(render_pdf_sync_summary(results))
    preview = render_pdf_sync_preview(results)
    if preview:
        logger.info(preview)
    if args.dry_run:
        return 0
    if args.in_place:
        write_output(args.input_bib, updated, in_place_target=args.input_bib)
    else:
        write_output(args.out, updated)
    return 0


def _run_refresh(args: argparse.Namespace, config: BibConfig) -> int:
    input_bib = args.input_bib
    pdf_dir = args.pdf_dir or Path(config.pdf_sync.pdf_dir)
    text, entries = load_bibtex_file(input_bib)

    logger.info(
        "refresh input=%s pdf_dir=%s mode=%s",
        input_bib,
        pdf_dir,
        "dry-run" if args.dry_run else "in-place",
    )

    text, pdf_results = _pdf_sync_document(text, entries, input_bib, pdf_dir, config)
    entries = parse_bibtex(text)
    logger.info(render_pdf_sync_summary(pdf_results))
    pdf_preview = render_pdf_sync_preview(pdf_results)
    if pdf_preview:
        logger.info(pdf_preview)

    text, previews, changed_count = _enrich_document(text, entries, config)
    entries = parse_bibtex(text)
    logger.info(render_enrichment_summary(previews, changed_count))
    enrich_preview = render_enrichment_preview(previews)
    if enrich_preview:
        logger.info(enrich_preview)

    duplicates = detect_duplicates(entries)
    logger.info(render_duplicates(duplicates))

    text, screened_entries = _screen_document(text, entries, config)
    logger.info(render_screening_summary(screened_entries))
    logger.info(
        "refresh complete steps=pdf-sync,enrich,dedupe,screen mode=%s",
        "dry-run" if args.dry_run else "in-place",
    )

    if args.dry_run:
        return 0

    write_output(input_bib, text, in_place_target=input_bib)
    return 0
