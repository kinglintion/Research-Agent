"""Offline regression checks for content-based duplicate warnings and version replacement."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf
from sqlalchemy import create_engine, select
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.models import Base, Paper, PaperVersion, ParagraphTranslation
from backend.app.services import paper_ingestion as ingestion
from backend.app.services.pdf_parsing import ParsedPdf


@compiles(LONGTEXT, "sqlite")
def _sqlite_longtext(_type, _compiler, **_kwargs):
    return "TEXT"


class _TinyEmbedder:
    def get_text_embedding(self, text: str) -> list[float]:
        return [float(len(text) % 97 + 1), 1.0]


class DuplicateReviewTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                                    poolclass=StaticPool)
        Base.metadata.create_all(self.engine)
        folder = Path(__file__).resolve().parents[2] / "科研小助手" / "pdf"
        self.v1_bytes = (folder / "1706.03762v1.pdf").read_bytes()
        self.v7_bytes = (folder / "1706.03762v7.pdf").read_bytes()
        with pymupdf.open(stream=self.v1_bytes, filetype="pdf") as pdf:
            self.v1_text = "\n".join(page.get_text() for page in pdf)
        with pymupdf.open(stream=self.v7_bytes, filetype="pdf") as pdf:
            self.v7_text = "\n".join(page.get_text() for page in pdf)
        with Session(self.engine) as session:
            session.add(Paper(id="paper-1", arxiv_id="1706.03762", canonical_title="Attention Is All You Need"))
            session.add(PaperVersion(id="version-1", paper_id="paper-1", version_label="v1",
                source_filename="1706.03762v1.pdf", source_sha256="a" * 64, stored_pdf_path="",
                title="Attention Is All You Need", authors=[], abstract="", parser_name="pymupdf",
                needs_ocr=False, structured_json={}, markdown=self.v1_text))
            session.add(ParagraphTranslation(version_id="version-1", paragraph_id="p001", target_language="zh",
                source_sha256="a" * 64, translated_text="旧译文"))
            session.commit()

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_v7_triggers_content_warning_before_ingestion(self) -> None:
        with patch.object(ingestion, "mysql_engine", return_value=self.engine), \
             patch.object(ingestion, "ensure_paper_tables"), \
             patch("backend.app.services.model_adapters.get_embedder", return_value=_TinyEmbedder()):
            result = ingestion.preview_pdf_duplicates("1706.03762v7.pdf", self.v7_bytes)
        self.assertTrue(result["has_warning"])
        self.assertEqual(result["matches"][0]["version_id"], "version-1")
        self.assertEqual(result["matches"][0]["kind"], "content_overlap")
        self.assertGreater(result["matches"][0]["content_similarity"], 0.6)

    def test_batch_warns_about_v1_v7_overlap(self) -> None:
        with patch.object(ingestion, "mysql_engine", return_value=self.engine), \
             patch.object(ingestion, "ensure_paper_tables"), \
             patch("backend.app.services.model_adapters.get_embedder", return_value=_TinyEmbedder()):
            previews = ingestion.preview_pdf_batch([
                ("1706.03762v1.pdf", self.v1_bytes), ("1706.03762v7.pdf", self.v7_bytes)])
        self.assertTrue(all(item["has_warning"] for item in previews))
        self.assertGreater(previews[0]["batch_matches"][0]["content_similarity"], 0.6)
        self.assertGreater(previews[1]["batch_matches"][0]["content_similarity"], 0.6)

    def test_filename_alone_does_not_trigger_warning(self) -> None:
        pdf = pymupdf.open()
        page = pdf.new_page()
        page.insert_text((40, 50), "A study of forest soil moisture and rainfall patterns")
        page.insert_text((40, 80), "Tree roots and seasonal water balance are measured in field plots.")
        unrelated = pdf.tobytes()
        pdf.close()
        with patch.object(ingestion, "mysql_engine", return_value=self.engine), \
             patch.object(ingestion, "ensure_paper_tables"), \
             patch("backend.app.services.model_adapters.get_embedder", return_value=_TinyEmbedder()):
            result = ingestion.preview_pdf_duplicates("1706.03762v1.pdf", unrelated)
        self.assertFalse(result["has_warning"])
        self.assertEqual(result["matches"], [])

    def test_replace_updates_only_selected_version_and_removes_old_translation(self) -> None:
        parsed = ParsedPdf(parser_name="pymupdf", markdown=self.v7_text, full_text=self.v7_text,
                           pages=[], sections=[], title="Attention Is All You Need", authors=[])
        bands, shingles = ingestion._minhash_bands(self.v7_text)
        with patch.object(ingestion, "mysql_engine", return_value=self.engine), \
             patch.object(ingestion, "index_paper_content", return_value={"status": "indexed"}):
            result = ingestion._replace_pdf_version(
                "version-1", "1706.03762v7.pdf", "b" * 64,
                Path(".data/papers/new.pdf"), parsed, {"warnings": []},
                bands, shingles, "1706.03762", None, "v7", None)
        self.assertEqual(result["replaced_version_id"], "version-1")
        with Session(self.engine) as session:
            version = session.get(PaperVersion, "version-1")
            self.assertEqual(version.version_label, "v7")
            self.assertEqual(version.source_sha256, "b" * 64)
            self.assertEqual(session.scalars(select(ParagraphTranslation)).all(), [])


if __name__ == "__main__":
    unittest.main()
