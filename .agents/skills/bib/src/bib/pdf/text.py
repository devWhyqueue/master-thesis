"""Shared private text heuristics for PDF metadata extraction."""

from __future__ import annotations

import re
from pathlib import Path

from ..config import BibConfig

DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


def _parse_filename(pdf_path: Path, config: BibConfig) -> tuple[str | None, list[str]]:
    if not config.pdf_sync.filename_parsing:
        return None, []
    stem = pdf_path.stem
    if ";" not in stem:
        return stem, []
    author_part, title_part = [part.strip() for part in stem.split(";", 1)]
    return title_part.strip(), _parse_author_filename(author_part)


def _parse_author_filename(text: str) -> list[str]:
    if "et al." in text:
        return [text.replace("et al.", "").strip().rstrip(",")]
    if "," in text and " and " not in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text.strip()] if text.strip() else []


def _parse_author_line(text: str) -> list[str]:
    cleaned = re.sub(r"[*†∗]", "", re.sub(r"\d+", "", text))
    return [
        part.strip(" ,")
        for part in cleaned.replace(" and ", ",").split(",")
        if part.strip(" ,")
    ]


def _clean_text(value: object) -> str | None:
    return " ".join(value.split()) or None if isinstance(value, str) else None


def _extract_doi(text: str) -> str | None:
    match = DOI_PATTERN.search(text)
    return match.group(0).rstrip(".,;)") if match else None


def _extract_year(text: str) -> int | None:
    for line in [line.strip() for line in text.splitlines() if line.strip()][:15]:
        match = YEAR_PATTERN.search(line)
        if match:
            year = int(match.group(0))
            if 1990 <= year <= 2100:
                return year
    years = [int(match.group(0)) for match in YEAR_PATTERN.finditer(text)]
    years = [year for year in years if 1990 <= year <= 2100]
    return years[0] if years else None


def _extract_venue(text: str) -> str | None:
    for line in [line.strip() for line in text.splitlines() if line.strip()][:8]:
        if "conference paper at" in line.casefold():
            return line
    return None


def _is_non_content_line(line: str) -> bool:
    lower = line.casefold()
    if (
        len(line) < 8
        or any(token in lower for token in ("doi:", "arxiv:"))
        or "@" in line
        or lower.startswith(("abstract", "correspondence"))
    ):
        return True
    if lower.startswith(
        (
            "department of",
            "laboratory of",
            "school of",
            "institute of",
            "faculty of",
            "college of",
        )
    ):
        return True
    return bool(line and line[0].isdigit() and any(char.isalpha() for char in line))


def _looks_like_person_name(author: str) -> bool:
    words = [word for word in re.findall(r"[A-Za-z][A-Za-z.'-]*", author) if word]
    if not 2 <= len(words) <= 5:
        return False
    if {word.casefold() for word in words} & {
        "survey",
        "pathology",
        "foundation",
        "model",
        "models",
    }:
        return False
    return sum(1 for word in words if word[0].isupper()) >= 2


def _leading_person_names(authors: list[str]) -> list[str]:
    names: list[str] = []
    for author in authors:
        if not _looks_like_person_name(author):
            break
        names.append(author)
    return names


def _is_probable_title_line(line: str) -> bool:
    if _is_non_content_line(line):
        return False
    words = re.findall(r"[A-Za-z0-9]+", line)
    if len(words) < 4 or len(_leading_person_names(_parse_author_line(line))) >= 2:
        return False
    alpha_words = [word for word in words if any(char.isalpha() for char in word)]
    if not alpha_words:
        return False
    capitalized = sum(1 for word in alpha_words if word[0].isupper() or word.isupper())
    return line.isupper() or ":" in line or (capitalized / len(alpha_words)) >= 0.7


def _is_probable_title_continuation(line: str) -> bool:
    if _is_non_content_line(line) or line.count(",") >= 2:
        return False
    return (
        len(_leading_person_names(_parse_author_line(line))) < 2
        and len(re.findall(r"[A-Za-z0-9]+", line)) >= 3
    )


def _extract_title(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for index, line in enumerate(lines[:12]):
        if (
            "published as a conference paper at" in line.casefold()
            or not _is_probable_title_line(line)
        ):
            continue
        collected = [line]
        for next_line in lines[index + 1 : index + 3]:
            if not _is_probable_title_continuation(next_line):
                break
            collected.append(next_line)
        return " ".join(collected)
    return None


def _extract_authors(text: str) -> list[str]:
    for line in [line.strip() for line in text.splitlines() if line.strip()][:15]:
        if _is_non_content_line(line):
            continue
        if ("," in line or " and " in line.casefold()) and any(
            char.isalpha() for char in line
        ):
            leading = _leading_person_names(_parse_author_line(line))
            if len(leading) >= 2:
                return leading[:6]
    return []


def _infer_entry_type(venue: str | None) -> str:
    return (
        "inproceedings"
        if venue and "conference paper" in venue.casefold()
        else "article"
    )
