"""Argument parser helpers for the BibTeX CLI."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser."""

    parser = argparse.ArgumentParser(prog="bib")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("enrich", "screen"):
        _add_mutation_parser(subparsers.add_parser(name))
    _add_pdf_sync_parser(subparsers.add_parser("pdf-sync"))
    _add_dedupe_parser(subparsers.add_parser("dedupe"))
    return parser


def _add_mutation_parser(parser: argparse.ArgumentParser) -> None:
    _add_common_file_args(parser)


def _add_pdf_sync_parser(parser: argparse.ArgumentParser) -> None:
    _add_common_file_args(parser)
    parser.add_argument("--pdf-dir", type=Path, help="Directory containing local PDFs")


def _add_dedupe_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input_bib", type=Path, help="Path to the input BibTeX file")
    parser.add_argument("--config", type=Path, help="Optional YAML config file")
    parser.add_argument(
        "--cache-dir", type=Path, help="Override the provider cache directory"
    )
    parser.add_argument(
        "--disable-online-enrichment",
        action="store_true",
        help="Use only local metadata",
    )


def _add_common_file_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input_bib", type=Path, help="Path to the input BibTeX file")
    parser.add_argument("--config", type=Path, help="Optional YAML config file")
    parser.add_argument("--out", type=Path, help="Path to the output BibTeX file")
    parser.add_argument(
        "--cache-dir", type=Path, help="Override the provider cache directory"
    )
    parser.add_argument(
        "--in-place", action="store_true", help="Replace the input file in place"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the preview without writing"
    )
    parser.add_argument(
        "--disable-online-enrichment",
        action="store_true",
        help="Use only local metadata",
    )
