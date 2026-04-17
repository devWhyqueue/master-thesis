from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from bib.bibtex import append_bibtex_entries, parse_bibtex, update_bibtex_fields
from bib.cli import main
from bib.config import load_config
from bib.metadata import (
    detect_duplicates,
    normalize_title,
    render_pdf_sync_preview,
    render_pdf_sync_summary,
    score_candidate,
    score_entry,
    screening_updates,
)
from bib.models import EnrichmentData
from bib.pdf import (
    MergedPdfMetadata,
    PdfSyncResult,
    build_partial_entry,
    create_new_entry,
    extract_pdf_metadata,
    generate_key,
    match_existing_entry,
)


SAMPLE_BIB = """@article{paper1,
  title = {A Review of Something},
  year = {2020},
  journal = {Nature Medicine},
  doi = {10.1000/review}
}

@inproceedings{paper2,
  author = {Lukas Muttenthaler and Robert A. Vandermeulen},
  title = {Set Learning for Accurate and Calibrated Models},
  booktitle = {The Twelfth International Conference on Learning Representations (ICLR 2024)},
  year = {2024},
  url = {https://openreview.net/forum?id=HZ3S17EI0o}
}
"""


class BibTests(unittest.TestCase):
    def test_parse_bibtex_preserves_entries(self) -> None:
        entries = parse_bibtex(SAMPLE_BIB)
        self.assertEqual(2, len(entries))
        self.assertEqual("paper1", entries[0].key)
        self.assertEqual("10.1000/review", entries[0].doi)

    def test_update_bibtex_fields_updates_canonical_field(self) -> None:
        updated = update_bibtex_fields(SAMPLE_BIB, {"paper2": {"doi": "10.1000/iclr", "publisher": "OpenReview"}})
        self.assertIn("doi = {10.1000/iclr}", updated)
        self.assertIn("publisher = {OpenReview}", updated)

    def test_append_bibtex_entries_adds_new_entry(self) -> None:
        updated = append_bibtex_entries(
            SAMPLE_BIB,
            [{"entry_type": "article", "key": "paper3", "fields": [("title", "Third Paper"), ("year", "2024")]}],
        )
        self.assertIn("@article{paper3,", updated)

    def test_normalize_title(self) -> None:
        self.assertEqual("a review of something", normalize_title("A Review of Something!"))

    def test_detect_duplicates(self) -> None:
        entries = parse_bibtex(SAMPLE_BIB + SAMPLE_BIB.replace("paper1", "paper3"))
        duplicates = detect_duplicates(entries)
        self.assertIn("10.1000/review", duplicates.doi_keys)

    def test_score_candidate_prefers_title_year_venue_match(self) -> None:
        entry = parse_bibtex(SAMPLE_BIB)[1]
        candidate = score_candidate(
            entry,
            EnrichmentData(
                provider="crossref",
                data={
                    "title": "Set Learning for Accurate and Calibrated Models",
                    "year": 2024,
                    "container-title": "ICLR",
                    "doi": "10.1000/iclr",
                    "type": "proceedings-article",
                    "url": "https://openreview.net/forum?id=HZ3S17EI0o",
                },
            ),
        )
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertGreaterEqual(candidate.confidence, 0.74)

    def test_screening_updates_include_bucket(self) -> None:
        entry = parse_bibtex(SAMPLE_BIB)[1]
        resolution = load_config(None)
        from bib.models import CitationStats, ResolutionResult, ResolvedMetadata

        score = score_entry(
            entry,
            ResolutionResult(
                matched=False,
                confidence=0.0,
                metadata=ResolvedMetadata(
                    title=entry.title,
                    booktitle=entry.fields.get("booktitle"),
                    year=entry.year,
                    url=entry.url,
                    publication_type="inproceedings",
                    citation_stats=CitationStats(),
                    provider_notes=["no resolvable external match found"],
                ),
                reasons=["no resolvable external match found"],
            ),
            detect_duplicates([entry]),
            resolution,
        )
        updates = screening_updates(score, include_details=True, details_format="compact")
        self.assertIn("x_screening_bucket", updates)

    def test_extract_pdf_metadata_from_filename_and_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pdf_path = root / "papers" / "calibration" / "Muttenthaler et al.; Set Learning for accurate, calibrated models.pdf"
            pdf_path.parent.mkdir(parents=True)
            _make_pdf(
                pdf_path,
                [
                    "Published as a conference paper at ICLR 2024",
                    "SET LEARNING FOR ACCURATE AND CALIBRATED MODELS",
                    "Lukas Muttenthaler, Robert A. Vandermeulen",
                    "doi: 10.48550/arXiv.2307.02245",
                ],
            )
            bib_path = root / "papers" / "sources.bib"
            bib_path.write_text("", encoding="utf-8")
            extracted = extract_pdf_metadata(pdf_path, bib_path, load_config(None))
            self.assertEqual("Set Learning for accurate, calibrated models", extracted.title)
            self.assertEqual(2024, extracted.year)
            self.assertEqual("10.48550/arXiv.2307.02245", extracted.doi)
            self.assertEqual("calibration/Muttenthaler et al.; Set Learning for accurate, calibrated models.pdf", extracted.relative_pdf_path)

    def test_extract_pdf_metadata_prefers_more_specific_title_page_title(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pdf_path = root / "papers" / "current reviews" / "Li et al.; Survey on CPath Foundation Models.pdf"
            pdf_path.parent.mkdir(parents=True)
            _make_pdf(
                pdf_path,
                [
                    "A Survey on Computational Pathology Foundation Models: Datasets, Adaptation Strategies, and Evaluation Tasks",
                    "Dong Li, Guihong Wan, Xintao Wu, Xinyu Wu",
                    "Abstract",
                ],
            )
            bib_path = root / "papers" / "sources.bib"
            bib_path.write_text("", encoding="utf-8")
            extracted = extract_pdf_metadata(pdf_path, bib_path, load_config(None))
            self.assertEqual(
                "A Survey on Computational Pathology Foundation Models: Datasets, Adaptation Strategies, and Evaluation Tasks",
                extracted.title,
            )

    def test_extract_pdf_metadata_prefers_title_page_authors_when_filename_is_weak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pdf_path = root / "papers" / "basics" / "Hong et al.; AI for Digital Pathology.pdf"
            pdf_path.parent.mkdir(parents=True)
            _make_pdf(
                pdf_path,
                [
                    "Artificial Intelligence for Digital and Computational Pathology",
                    "Andrew H. Song, Guillaume Jaume, Drew F.K. Williamson",
                    "Abstract",
                ],
            )
            bib_path = root / "papers" / "sources.bib"
            bib_path.write_text("", encoding="utf-8")
            extracted = extract_pdf_metadata(pdf_path, bib_path, load_config(None))
            self.assertEqual(
                ["Andrew H. Song", "Guillaume Jaume", "Drew F.K. Williamson"],
                extracted.authors,
            )

    def test_extract_pdf_metadata_keeps_filename_when_title_page_is_unrelated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pdf_path = root / "papers" / "basics" / "Doe et al.; Example Paper.pdf"
            pdf_path.parent.mkdir(parents=True)
            _make_pdf(
                pdf_path,
                [
                    "Supplementary Material",
                    "Noise only, not the paper title",
                    "Abstract",
                ],
            )
            bib_path = root / "papers" / "sources.bib"
            bib_path.write_text("", encoding="utf-8")
            extracted = extract_pdf_metadata(pdf_path, bib_path, load_config(None))
            self.assertEqual("Example Paper", extracted.title)

    def test_extract_pdf_metadata_hong_style_pdf_uses_full_canonical_title(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pdf_path = root / "papers" / "basics" / "Hong et al.; AI for Digital Pathology.pdf"
            pdf_path.parent.mkdir(parents=True)
            _make_pdf(
                pdf_path,
                [
                    "Artificial Intelligence for Digital and Computational Pathology",
                    "Andrew H. Song, Guillaume Jaume, Drew F.K. Williamson",
                    "arXiv:2401.06148v1  [eess.IV]  13 Dec 2023",
                    "Abstract",
                ],
            )
            bib_path = root / "papers" / "sources.bib"
            bib_path.write_text("", encoding="utf-8")
            extracted = extract_pdf_metadata(pdf_path, bib_path, load_config(None))
            self.assertEqual(
                "Artificial Intelligence for Digital and Computational Pathology",
                extracted.title,
            )
            self.assertEqual(2023, extracted.year)

    def test_match_existing_entry_by_title_year(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pdf_path = root / "papers" / "calibration" / "Muttenthaler et al.; Set Learning for accurate, calibrated models.pdf"
            pdf_path.parent.mkdir(parents=True)
            _make_pdf(pdf_path, ["Published as a conference paper at ICLR 2024"])
            bib_path = root / "papers" / "sources.bib"
            bib_path.write_text(SAMPLE_BIB, encoding="utf-8")
            extracted = extract_pdf_metadata(pdf_path, bib_path, load_config(None))
            entries = parse_bibtex(SAMPLE_BIB)
            match, confidence, _ = match_existing_entry(extracted, entries, load_config(None))
            self.assertIsNotNone(match)
            assert match is not None
            self.assertEqual("paper2", match.key)
            self.assertGreaterEqual(confidence, 0.82)

    def test_generate_key_author_year_slug(self) -> None:
        key = generate_key(
            MergedPdfMetadata(
                title="Set Learning for Accurate and Calibrated Models",
                authors=["Lukas Muttenthaler"],
                year=2024,
                journal=None,
                booktitle="ICLR",
                publisher=None,
                volume=None,
                number=None,
                pages=None,
                doi=None,
                url=None,
                file_field=None,
                publication_type="inproceedings",
            ),
            {"muttenthaler2024setLearning"},
        )
        self.assertEqual("muttenthaler2024setLearning-2", key)

    def test_create_new_entry_with_relative_file_field(self) -> None:
        from bib.models import CitationStats, ResolvedMetadata

        config = load_config(None)
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pdf_path = root / "papers" / "basics" / "Doe et al.; Example Paper.pdf"
            pdf_path.parent.mkdir(parents=True)
            _make_pdf(pdf_path, ["EXAMPLE PAPER", "Jane Doe", "doi: 10.1000/example"])
            bib_path = root / "papers" / "sources.bib"
            bib_path.write_text("", encoding="utf-8")
            extracted = extract_pdf_metadata(pdf_path, bib_path, config)
            entry = create_new_entry(
                extracted,
                ResolvedMetadata(
                    title="Example Paper",
                    year=2024,
                    journal="Example Journal",
                    doi="10.1000/example",
                    url="https://doi.org/10.1000/example",
                    publication_type="journal-article",
                    citation_stats=CitationStats(),
                ),
                set(),
                config,
            )
            self.assertIsNotNone(entry)
            assert entry is not None
            field_map = dict(entry["fields"])
            self.assertEqual("basics/Doe et al.; Example Paper.pdf", field_map["file"])

    def test_cli_enrich_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "input.bib"
            input_path.write_text(SAMPLE_BIB, encoding="utf-8")
            result = main(["enrich", str(input_path), "--dry-run", "--disable-online-enrichment"])
            self.assertEqual(0, result)

    def test_cli_screen_out_writes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "input.bib"
            out_path = root / "output.bib"
            input_path.write_text(SAMPLE_BIB, encoding="utf-8")
            result = main(["screen", str(input_path), "--out", str(out_path), "--disable-online-enrichment"])
            self.assertEqual(0, result)
            self.assertTrue(out_path.exists())
            self.assertIn("x_screening_bucket", out_path.read_text(encoding="utf-8"))

    def test_cli_pdf_sync_out_writes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            papers_dir = root / "papers" / "calibration"
            papers_dir.mkdir(parents=True)
            pdf_path = papers_dir / "Muttenthaler et al.; Set Learning for accurate, calibrated models.pdf"
            _make_pdf(
                pdf_path,
                [
                    "Published as a conference paper at ICLR 2024",
                    "SET LEARNING FOR ACCURATE AND CALIBRATED MODELS",
                    "Lukas Muttenthaler",
                    "doi: 10.48550/arXiv.2307.02245",
                ],
            )
            input_path = root / "papers" / "sources.bib"
            input_path.write_text(SAMPLE_BIB, encoding="utf-8")
            out_path = root / "output.bib"
            result = main(
                [
                    "pdf-sync",
                    str(input_path),
                    "--pdf-dir",
                    str(root / "papers"),
                    "--out",
                    str(out_path),
                    "--disable-online-enrichment",
                ]
            )
            self.assertEqual(0, result)
            content = out_path.read_text(encoding="utf-8")
            self.assertIn("file = {calibration/Muttenthaler et al.; Set Learning for accurate, calibrated models.pdf}", content)

    def test_cli_pdf_sync_resolves_new_entry_from_title_page_title(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            papers_dir = root / "papers" / "current reviews"
            papers_dir.mkdir(parents=True)
            pdf_path = papers_dir / "Li et al.; Survey on CPath Foundation Models.pdf"
            _make_pdf(
                pdf_path,
                [
                    "A Survey on Computational Pathology Foundation Models: Datasets, Adaptation Strategies, and Evaluation Tasks",
                    "Dong Li, Guihong Wan, Xintao Wu",
                    "Abstract",
                ],
            )
            input_path = root / "papers" / "sources.bib"
            input_path.write_text("", encoding="utf-8")
            out_path = root / "output.bib"

            def _fake_resolve(entry, providers, cache, config):
                from bib.models import CitationStats, ResolutionResult, ResolvedMetadata

                self.assertEqual(
                    "A Survey on Computational Pathology Foundation Models: Datasets, Adaptation Strategies, and Evaluation Tasks",
                    entry.title,
                )
                return ResolutionResult(
                    matched=True,
                    confidence=0.95,
                    metadata=ResolvedMetadata(
                        title=entry.title,
                        year=2025,
                        journal="arXiv",
                        doi="10.48550/arXiv.2501.15724",
                        url="https://doi.org/10.48550/arXiv.2501.15724",
                        publication_type="preprint",
                        citation_stats=CitationStats(),
                    ),
                )

            with patch("bib.cli.commands.resolve_entry", new=_fake_resolve):
                result = main(
                    [
                        "pdf-sync",
                        str(input_path),
                        "--pdf-dir",
                        str(root / "papers"),
                        "--out",
                        str(out_path),
                    ]
                )

            self.assertEqual(0, result)
            content = out_path.read_text(encoding="utf-8")
            self.assertIn(
                "title = {A Survey on Computational Pathology Foundation Models: Datasets, Adaptation Strategies, and Evaluation Tasks}",
                content,
            )
            self.assertIn("doi = {10.48550/arXiv.2501.15724}", content)

    def test_cli_dedupe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "input.bib"
            input_path.write_text(SAMPLE_BIB + SAMPLE_BIB.replace("paper1", "paper3"), encoding="utf-8")
            result = main(["dedupe", str(input_path)])
            self.assertEqual(0, result)

    def test_render_pdf_sync_summary_uses_needs_review_bucket(self) -> None:
        from bib.models import ExtractedPdfMetadata

        pdf = ExtractedPdfMetadata(
            pdf_path=Path("papers/example.pdf"),
            relative_pdf_path="example.pdf",
            title="Example Paper",
        )
        summary = render_pdf_sync_summary(
            [
                PdfSyncResult(
                    pdf=pdf,
                    matched_key="paper1",
                    match_confidence=1.0,
                    created_key=None,
                    updates={"file": "example.pdf"},
                    new_entry=None,
                    reasons=[],
                ),
                PdfSyncResult(
                    pdf=pdf,
                    matched_key=None,
                    match_confidence=0.45,
                    created_key=None,
                    updates={},
                    new_entry=None,
                    reasons=["match candidate below confidence threshold"],
                ),
            ]
        )
        self.assertEqual("processed=2 matched=1 created=0 needs_review=1", summary)

    def test_render_pdf_sync_preview_includes_agent_review_payload(self) -> None:
        from bib.models import ExtractedPdfMetadata

        pdf = ExtractedPdfMetadata(
            pdf_path=Path("papers/example.pdf"),
            relative_pdf_path="current reviews/example.pdf",
            title="A Survey of Example Models",
            authors=["Jane Doe", "John Smith"],
            year=2025,
            doi="10.1000/example",
        )
        preview = render_pdf_sync_preview(
            [
                PdfSyncResult(
                    pdf=pdf,
                    matched_key=None,
                    match_confidence=0.45,
                    created_key=None,
                    updates={},
                    new_entry=None,
                    reasons=["match candidate below confidence threshold"],
                )
            ]
        )
        self.assertIn("needs_review", preview)
        self.assertIn("title=A Survey of Example Models", preview)
        self.assertIn("authors=Jane Doe | John Smith", preview)
        self.assertIn("year=2025", preview)
        self.assertIn("doi=10.1000/example", preview)


def _make_pdf(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    y = height - 72
    for line in lines:
        c.drawString(72, y, line)
        y -= 16
    c.save()


if __name__ == "__main__":
    unittest.main()
