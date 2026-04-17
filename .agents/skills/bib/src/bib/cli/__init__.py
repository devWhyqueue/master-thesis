"""CLI operations for the BibTeX skill."""

from __future__ import annotations

from .commands import main
from .parser import build_parser

__all__ = ["build_parser", "main"]
