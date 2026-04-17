"""BibTeX parsing and rendering helpers."""

from __future__ import annotations

from .parse import (
    append_bibtex_entries,
    load_bibtex_file,
    parse_bibtex,
    strip_screening_updates,
    update_bibtex_fields,
    validate_bibtex,
    write_output,
)
from .render import detect_indent, normalize_field_line, render_entry, render_new_entry

__all__ = [
    "append_bibtex_entries",
    "detect_indent",
    "load_bibtex_file",
    "normalize_field_line",
    "parse_bibtex",
    "render_entry",
    "render_new_entry",
    "strip_screening_updates",
    "update_bibtex_fields",
    "validate_bibtex",
    "write_output",
]
