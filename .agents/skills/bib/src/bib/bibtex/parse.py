"""BibTeX parsing and safe update helpers."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from ..models import BibEntry, ParsedField
from .render import render_entry, render_new_entry

SCREENING_FIELDS = {
    "x_screening_bucket",
    "x_screening_score",
    "x_screening_reason",
    "x_screening_details",
}


def parse_bibtex(text: str) -> list[BibEntry]:
    """Parse a BibTeX document while keeping raw entry text."""

    entries: list[BibEntry] = []
    index = 0
    while True:
        at_sign = text.find("@", index)
        if at_sign == -1:
            break
        entry, cursor = _parse_entry(text, at_sign)
        entries.append(entry)
        index = cursor
    return entries


def load_bibtex_file(path: Path) -> tuple[str, list[BibEntry]]:
    """Read and parse a BibTeX file."""

    text = path.read_text(encoding="utf-8")
    return text, parse_bibtex(text)


def validate_bibtex(text: str) -> None:
    """Raise if the updated BibTeX cannot be parsed again."""

    parse_bibtex(text)


def update_bibtex_fields(
    text: str, updates_by_key: dict[str, dict[str, str | None]]
) -> str:
    """Apply field updates while preserving unchanged raw fields."""

    entries = parse_bibtex(text)
    parts: list[str] = []
    cursor = 0
    for entry in entries:
        parts.append(text[cursor : entry.start])
        parts.append(render_entry(entry, updates_by_key.get(entry.key, {})))
        cursor = entry.end
    parts.append(text[cursor:])
    return "".join(parts)


def append_bibtex_entries(text: str, new_entries: list[dict[str, object]]) -> str:
    """Append new BibTeX entries to the document in stable order."""

    if not new_entries:
        return text
    suffix = "\n" if text.endswith("\n") else "\n\n"
    rendered = "\n\n".join(render_new_entry(entry) for entry in new_entries)
    return f"{text}{suffix}{rendered}\n"


def sort_bibtex_entries(text: str) -> str:
    """Sort BibTeX entries alphabetically by key while preserving entry contents."""

    entries = parse_bibtex(text)
    if not entries:
        return text

    prefix = text[: entries[0].start]
    if prefix.strip():
        prefix = f"{prefix.rstrip()}\n\n"
    else:
        prefix = ""
    rendered_entries = [
        render_entry(entry, {})
        for entry in sorted(entries, key=lambda entry: (entry.key.lower(), entry.start))
    ]
    rendered_body = "\n\n".join(rendered_entries)
    return f"{prefix}{rendered_body}\n"


def strip_screening_updates() -> dict[str, str | None]:
    """Return an update map that removes existing screening fields."""

    return {field: None for field in SCREENING_FIELDS}


def write_output(path: Path, text: str, *, in_place_target: Path | None = None) -> None:
    """Write validated output, replacing the source atomically when requested."""

    validate_bibtex(text)
    target = in_place_target or path
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=target.parent
    ) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    try:
        validate_bibtex(temp_path.read_text(encoding="utf-8"))
        temp_path.replace(target)
    except OSError:
        temp_path.unlink(missing_ok=True)
        raise


def _parse_entry(text: str, at_sign: int) -> tuple[BibEntry, int]:
    brace_start = text.find("{", at_sign)
    if brace_start == -1:
        raise ValueError("invalid BibTeX entry: missing opening brace")
    cursor = _find_matching_brace(text, brace_start)
    raw_entry = text[at_sign:cursor]
    entry_type = text[at_sign + 1 : brace_start].strip()
    inner = text[brace_start + 1 : cursor - 1]
    comma = inner.find(",")
    if comma == -1:
        raise ValueError("invalid BibTeX entry: missing key separator")
    key = inner[:comma].strip()
    field_segments = _parse_fields(inner[comma + 1 :])
    fields = {field.name: field.value for field in field_segments}
    return (
        BibEntry(
            entry_type=entry_type,
            key=key,
            fields=fields,
            field_order=[field.name for field in field_segments],
            start=at_sign,
            end=cursor,
            raw_text=raw_entry,
            header=raw_entry[: raw_entry.find(key) + len(key) + 1],
            field_segments=field_segments,
        ),
        cursor,
    )


def _find_matching_brace(text: str, brace_start: int) -> int:
    depth = 1
    cursor = brace_start + 1
    while cursor < len(text) and depth:
        char = text[cursor]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        cursor += 1
    if depth != 0:
        raise ValueError("invalid BibTeX entry: unmatched brace")
    return cursor


def _parse_fields(fields_text: str) -> list[ParsedField]:
    fields: list[ParsedField] = []
    index = 0
    while index < len(fields_text):
        while index < len(fields_text) and fields_text[index] in " \t\r\n,":
            index += 1
        if index >= len(fields_text):
            break

        name_start = index
        while index < len(fields_text) and fields_text[index] != "=":
            index += 1
        name = fields_text[name_start:index].strip().lower()
        if not name:
            break
        index += 1
        while index < len(fields_text) and fields_text[index].isspace():
            index += 1

        value, index = _parse_value(fields_text, index)
        raw_end = index
        while raw_end < len(fields_text) and fields_text[raw_end] in " \t\r\n":
            raw_end += 1
        if raw_end < len(fields_text) and fields_text[raw_end] == ",":
            raw_end += 1

        raw_text = fields_text[name_start:raw_end]
        fields.append(ParsedField(name=name, raw_text=raw_text, value=value.strip()))
        index = raw_end
    return fields


def _parse_value(text: str, index: int) -> tuple[str, int]:
    if index >= len(text):
        return "", index
    opener = text[index]
    if opener == "{":
        depth = 1
        cursor = index + 1
        while cursor < len(text) and depth:
            char = text[cursor]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            cursor += 1
        return text[index + 1 : cursor - 1], cursor
    if opener == '"':
        cursor = index + 1
        escaped = False
        while cursor < len(text):
            char = text[cursor]
            if char == '"' and not escaped:
                break
            escaped = char == "\\" and not escaped
            if char != "\\":
                escaped = False
            cursor += 1
        return text[index + 1 : cursor], cursor + 1

    cursor = index
    while cursor < len(text) and text[cursor] not in ",\n\r":
        cursor += 1
    return text[index:cursor].strip(), cursor
