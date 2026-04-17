"""BibTeX rendering helpers."""

from __future__ import annotations

import re
from typing import cast

from ..models import BibEntry


def render_entry(entry: BibEntry, updates: dict[str, str | None]) -> str:
    """Render an existing BibTeX entry with field updates applied."""
    indent = detect_indent([segment.raw_text for segment in entry.field_segments])
    rendered: list[str] = []
    seen: set[str] = set()
    for segment in entry.field_segments:
        seen.add(segment.name)
        if segment.name in updates:
            value = updates[segment.name]
            if value is None:
                continue
            rendered.append(render_field(segment.name, value, indent))
        else:
            rendered.append(
                normalize_field_line(segment.raw_text.rstrip().rstrip(","), indent)
            )
    for name, value in updates.items():
        if name not in seen and value is not None:
            rendered.append(render_field(name, value, indent))
    header = (
        entry.header if entry.header.endswith(("\n", "\r")) else f"{entry.header}\n"
    )
    body = ",\n".join(line for line in rendered if line.strip())
    return f"{header}{body}\n}}"


def detect_indent(lines: list[str]) -> str:
    """Infer indentation from existing field lines."""
    pattern = re.compile(r"^(\s+)")
    for line in lines:
        match = pattern.match(line)
        if match:
            return match.group(1)
    return "  "


def normalize_field_line(line: str, indent: str) -> str:
    """Normalize an unchanged raw field line to the selected indent."""
    stripped = line.lstrip()
    return f"{indent}{stripped}" if stripped else stripped


def render_field(name: str, value: str, indent: str) -> str:
    """Render one BibTeX field line."""
    return f"{indent}{name} = {{{sanitize_value(value)}}}"


def sanitize_value(value: str) -> str:
    """Collapse newlines and trim a field value for BibTeX output."""
    return value.replace("\n", " ").strip()


def render_new_entry(entry: dict[str, object]) -> str:
    """Render a new BibTeX entry from a plain dictionary."""
    entry_type = str(entry["entry_type"])
    key = str(entry["key"])
    fields = cast(list[tuple[str, str]], entry["fields"])
    indent = "  "
    lines = [f"@{entry_type}{{{key},"]
    for name, value in fields:
        lines.append(f"{indent}{name} = {{{sanitize_value(str(value))}}},")
    if lines[-1].endswith(","):
        lines[-1] = lines[-1][:-1]
    lines.append("}")
    return "\n".join(lines)
