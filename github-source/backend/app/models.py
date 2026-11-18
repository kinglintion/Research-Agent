from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    arxiv_id: Mapped[str | None] = mapped_column(String(32), unique=True, nullable=True)
    doi: Mapped[str | None] = mapped_column(String(255), index=True, nullable=True)
    canonical_title: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class PaperVersion(Base):
    __tablename__ = "paper_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    parent_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("paper_versions.id", ondelete="SET NULL"), nullable=True
    )
    version_label: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_filename: Mapped[str] = mapped_column(String(512))
    source_sha256: Mapped[str] = mapped_column(String(64), unique=True)
    stored_pdf_path: Mapped[str] = mapped_column(String(2048))
    title: Mapped[str] = mapped_column(Text, default="")
    authors: Mapped[list] = mapped_column(JSON, default=list)
    abstract: Mapped[str] = mapped_column(LONGTEXT, default="")
    parser_name: Mapped[str] = mapped_column(String(64))
    needs_ocr: Mapped[bool] = mapped_column(default=False)
    structured_json: Mapped[dict] = mapped_column(JSON)
    markdown: Mapped[str] = mapped_column(LONGTEXT)
    change_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class PaperFingerprintBand(Base):
    __tablename__ = "paper_fingerprint_bands"
    __table_args__ = (
        UniqueConstraint("version_id", "bucket_key", name="uq_fingerprint_version_bucket"),
        Index("ix_fingerprint_bucket_key", "bucket_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_id: Mapped[str] = mapped_column(ForeignKey("paper_versions.id", ondelete="CASCADE"))
    bucket_key: Mapped[str] = mapped_column(String(80))


class ParagraphTranslation(Base):
    __tablename__ = "paragraph_translations"
    __table_args__ = (
        UniqueConstraint("version_id", "paragraph_id", "target_language", name="uq_paragraph_translation"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_id: Mapped[str] = mapped_column(ForeignKey("paper_versions.id", ondelete="CASCADE"), index=True)
    paragraph_id: Mapped[str] = mapped_column(String(80))
    target_language: Mapped[str] = mapped_column(String(16))
    source_sha256: Mapped[str] = mapped_column(String(64))
    translated_text: Mapped[str] = mapped_column(LONGTEXT)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ExternalPaperRecord(Base):
    __tablename__ = "external_paper_records"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    canonical_title: Mapped[str] = mapped_column(Text, default="")
    doi: Mapped[str | None] = mapped_column(String(255), index=True, nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    sources_json: Mapped[list] = mapped_column(JSON, default=list)
    repositories_json: Mapped[list] = mapped_column(JSON, default=list)
    fulltext_candidates_json: Mapped[list] = mapped_column(JSON, default=list)
    metadata_checks_json: Mapped[list] = mapped_column(JSON, default=list)
    recommendation_reason: Mapped[str] = mapped_column(Text, default="")
    download_status: Mapped[str] = mapped_column(String(64), default="not_attempted")
    acquired_format: Mapped[str | None] = mapped_column(String(16), nullable=True)
    status_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    paper_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class UserResearchProfile(Base):
    __tablename__ = "user_research_profiles"

    user_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    interests: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class RecommendationFeedback(Base):
    __tablename__ = "recommendation_feedback"
    __table_args__ = (UniqueConstraint("user_id", "record_id", name="uq_recommendation_user_record"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("user_research_profiles.user_id", ondelete="CASCADE"), index=True)
    record_id: Mapped[str] = mapped_column(ForeignKey("external_paper_records.id", ondelete="CASCADE"), index=True)
    vote: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ReadingPreference(Base):
    __tablename__ = "reading_preferences"
    __table_args__ = (UniqueConstraint("user_id", "paper_id", name="uq_reading_preference_user_paper"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("user_research_profiles.user_id", ondelete="CASCADE"), index=True)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    last_opened_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    open_count: Mapped[int] = mapped_column(Integer, default=1)


class ArxivFollow(Base):
    __tablename__ = "arxiv_follows"
    __table_args__ = (UniqueConstraint("user_id", "query", name="uq_arxiv_follow_user_query"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("user_research_profiles.user_id", ondelete="CASCADE"), index=True)
    query: Mapped[str] = mapped_column(String(500))
    enabled: Mapped[bool] = mapped_column(default=True)
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
