"""Local preference ranking, recommendation feedback, and periodic arXiv tracking."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import ArxivFollow, ExternalPaperRecord, Paper, PaperVersion, ReadingPreference, RecommendationFeedback, UserResearchProfile
from .connections import mysql_engine
from .paper_ingestion import ensure_paper_tables


def _tokens(text: str) -> set[str]:
    return {item for item in re.findall(r"[\w-]+", (text or "").casefold()) if len(item) > 1}


def _profile(session: Session, user_id: str) -> UserResearchProfile:
    profile = session.get(UserResearchProfile, user_id)
    if profile is None:
        profile = UserResearchProfile(user_id=user_id, interests="")
        session.add(profile)
        session.flush()
    return profile


def save_profile(user_id: str, interests: str) -> dict:
    ensure_paper_tables()
    with Session(mysql_engine(), expire_on_commit=False) as session:
        profile = _profile(session, user_id)
        profile.interests = interests.strip()
        session.commit()
        return {"user_id": user_id, "interests": profile.interests, "updated_at": profile.updated_at.isoformat()}


def save_recommendation_feedback(user_id: str, record_id: str, vote: int) -> dict:
    ensure_paper_tables()
    with Session(mysql_engine(), expire_on_commit=False) as session:
        if session.get(ExternalPaperRecord, record_id) is None:
            return {"saved": False, "reason": "recommendation_not_found"}
        _profile(session, user_id)
        row = session.scalar(select(RecommendationFeedback).where(
            RecommendationFeedback.user_id == user_id, RecommendationFeedback.record_id == record_id))
        if row is None:
            row = RecommendationFeedback(user_id=user_id, record_id=record_id, vote=vote)
            session.add(row)
        else:
            row.vote = vote
        session.commit()
        return {"saved": True, "user_id": user_id, "record_id": record_id, "vote": vote}


def record_reading_preference(user_id: str, paper_id: str, version_id: str) -> dict:
    ensure_paper_tables()
    with Session(mysql_engine(), expire_on_commit=False) as session:
        version = session.get(PaperVersion, version_id)
        if version is None or version.paper_id != paper_id or session.get(Paper, paper_id) is None:
            return {"saved": False, "reason": "paper_version_not_found"}
        _profile(session, user_id)
        row = session.scalar(select(ReadingPreference).where(
            ReadingPreference.user_id == user_id, ReadingPreference.paper_id == paper_id))
        if row is None:
            row = ReadingPreference(user_id=user_id, paper_id=paper_id, open_count=1)
            session.add(row)
        else:
            row.open_count += 1
            row.last_opened_at = datetime.utcnow()
        session.commit()
        return {"saved": True, "paper_id": paper_id, "open_count": row.open_count}


def get_recommendations(user_id: str, limit: int = 20) -> dict:
    ensure_paper_tables()
    with Session(mysql_engine()) as session:
        profile = session.get(UserResearchProfile, user_id)
        interests = _tokens(profile.interests if profile else "")
        reading_ids = session.scalars(select(ReadingPreference.paper_id).where(ReadingPreference.user_id == user_id)).all()
        for paper_id in reading_ids:
            paper = session.get(Paper, paper_id)
            if paper:
                version = session.scalar(select(PaperVersion).where(PaperVersion.paper_id == paper_id).order_by(PaperVersion.created_at.desc()).limit(1))
                if version:
                    interests |= _tokens(" ".join([paper.canonical_title or version.title or "", version.abstract or ""]))
        feedback = {row.record_id: row.vote for row in session.scalars(
            select(RecommendationFeedback).where(RecommendationFeedback.user_id == user_id)).all()}
        records = session.scalars(select(ExternalPaperRecord)).all()
        ranked = []
        for record in records:
            metadata = record.metadata_json or {}
            haystack = _tokens(" ".join([record.canonical_title, metadata.get("abstract", ""),
                                         " ".join(repo.get("description", "") for repo in record.repositories_json or [])]))
            overlap = len(interests & haystack) / max(1, len(interests))
            vote = feedback.get(record.id, 0)
            score = overlap + (0.15 if vote > 0 else -0.2 if vote < 0 else 0)
            if record.repositories_json:
                score += 0.05
            if len(record.sources_json or []) > 1:
                score += 0.05
            reasons = []
            if overlap:
                reasons.append(f"与本地兴趣词匹配 {overlap:.0%}")
            if vote > 0:
                reasons.append("符合你的点赞反馈")
            elif vote < 0:
                reasons.append("已按你的踩反馈降序")
            if record.repositories_json:
                reasons.append("附有代码仓库")
            if len(record.sources_json or []) > 1:
                reasons.append("元数据有多个来源")
            result = {"record_id": record.id, "metadata": metadata, "sources": record.sources_json or [],
                      "repositories": record.repositories_json or [], "fulltext_candidates": record.fulltext_candidates_json or [],
                      "metadata_checks": record.metadata_checks_json or [], "recommendation_reason": "；".join(reasons) or "来自已缓存学术记录",
                      "download_status": record.download_status, "status_detail": record.status_detail,
                      "paper_id": record.paper_id, "version_id": record.version_id,
                      "personalization_score": round(score, 4), "vote": vote}
            ranked.append((score, result))
        ranked.sort(key=lambda row: row[0], reverse=True)
        return {"user_id": user_id, "interests": profile.interests if profile else "",
                "results": [result for _, result in ranked[:limit]]}


def save_arxiv_follow(user_id: str, query: str) -> dict:
    ensure_paper_tables()
    query = query.strip()
    with Session(mysql_engine(), expire_on_commit=False) as session:
        _profile(session, user_id)
        row = session.scalar(select(ArxivFollow).where(ArxivFollow.user_id == user_id, ArxivFollow.query == query))
        if row is None:
            row = ArxivFollow(user_id=user_id, query=query, enabled=True)
            session.add(row)
        else:
            row.enabled = True
        session.commit()
        result = {"id": row.id, "user_id": user_id, "query": query, "enabled": row.enabled,
                  "last_checked_at": row.last_checked_at.isoformat() if row.last_checked_at else None}
    # New and re-enabled follows are refreshed immediately; recurring refresh remains as a fallback.
    result["refresh"] = refresh_arxiv_query(query)
    result["last_checked_at"] = datetime.utcnow().isoformat()
    return result


def get_arxiv_follows(user_id: str) -> list[dict]:
    ensure_paper_tables()
    with Session(mysql_engine()) as session:
        rows = session.scalars(select(ArxivFollow).where(ArxivFollow.user_id == user_id).order_by(ArxivFollow.id)).all()
        return [{"id": row.id, "query": row.query, "enabled": row.enabled,
                 "last_checked_at": row.last_checked_at.isoformat() if row.last_checked_at else None} for row in rows]


def remove_arxiv_follow(user_id: str, follow_id: int) -> bool:
    ensure_paper_tables()
    with Session(mysql_engine()) as session:
        row = session.scalar(select(ArxivFollow).where(ArxivFollow.id == follow_id, ArxivFollow.user_id == user_id))
        if row is None:
            return False
        session.delete(row)
        session.commit()
        return True


def refresh_arxiv_query(query: str) -> dict:
    """Immediately fetch one followed query; cache metadata, never auto-download PDFs."""
    from .scholarly_discovery import _metadata_checks, _record_key, _search_arxiv

    ensure_paper_tables()
    try:
        papers = _search_arxiv(query, 10)
        with Session(mysql_engine(), expire_on_commit=False) as session:
            for metadata in papers:
                key = _record_key(metadata)
                sources = [{key: value for key, value in metadata.items() if key != "fulltext_candidates"}]
                record = session.get(ExternalPaperRecord, key)
                if record is None:
                    record = ExternalPaperRecord(id=key)
                    session.add(record)
                record.canonical_title = metadata.get("title") or "Untitled work"
                record.doi = metadata.get("doi")
                record.metadata_json = metadata
                record.sources_json = sources
                record.fulltext_candidates_json = metadata.get("fulltext_candidates", [])
                record.metadata_checks_json = _metadata_checks(sources)
                record.recommendation_reason = f"arXiv 跟踪：匹配订阅主题「{query}」；可查看公开预印本"
            checked_at = datetime.utcnow()
            session.query(ArxivFollow).filter(ArxivFollow.enabled.is_(True), ArxivFollow.query == query).update(
                {ArxivFollow.last_checked_at: checked_at}, synchronize_session=False)
            session.commit()
        return {"status": "refreshed", "papers": len(papers)}
    except Exception as exc:
        checked_at = datetime.utcnow()
        with Session(mysql_engine()) as session:
            session.query(ArxivFollow).filter(ArxivFollow.enabled.is_(True), ArxivFollow.query == query).update(
                {ArxivFollow.last_checked_at: checked_at}, synchronize_session=False)
            session.commit()
        return {"status": "failed", "error_type": exc.__class__.__name__, "message": str(exc)[:300]}


def refresh_arxiv_follows() -> int:
    """Scheduled refresh of active queries; only metadata is cached."""
    ensure_paper_tables()
    with Session(mysql_engine()) as session:
        queries = sorted({row.query for row in session.scalars(
            select(ArxivFollow).where(ArxivFollow.enabled.is_(True))).all()})
    return sum(int(refresh_arxiv_query(query).get("papers", 0)) for query in queries)
