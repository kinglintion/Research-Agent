"""Conservative, source-located draft-to-library candidate alignment."""
from __future__ import annotations

import re

from .paper_reader import reader_document
from .research_workbench import _is_research_paper, _latest_versions

_STOP = set("the and for with from this that using based study paper result results method model models to of in on by is are a an we our it 研究 论文 方法 结果 本文 基于 使用 提出".split())


def _tokens(text: str) -> set[str]:
    english = {word.casefold() for word in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text)
               if word.casefold() not in _STOP}
    chinese = {run[i:i + 2] for run in re.findall(r"[\u4e00-\u9fff]{2,}", text)
               for i in range(len(run) - 1)}
    return english | chinese


def _sections(draft: str) -> list[tuple[str, str]]:
    parts = re.split(r"(?m)^#{1,4}\s+(.+)\s*$", draft)
    if len(parts) == 1:
        return [("草稿", draft)]
    sections = [("前言", parts[0])] if parts[0].strip() else []
    sections += [(parts[i].strip(), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]
    return [(title, body[:3000]) for title, body in sections if body.strip()][:15]


def align_draft_to_library(draft_text: str, paper_ids: list[str] | None = None, limit: int = 12) -> dict:
    requested = set(paper_ids or [])
    papers = []
    for paper, version in _latest_versions(500):
        if not _is_research_paper(paper, version) or (requested and paper.id not in requested):
            continue
        papers.append((paper, version))
        if len(papers) >= limit:
            break
    sources = []
    for paper, version in papers:
        document = reader_document(paper.id, version.id)
        if not document:
            continue
        for paragraph in document.get("paragraphs", []):
            quote = re.sub(r"\s+", " ", str(paragraph.get("text") or "")).strip()
            if len(quote) < 45 or len(quote) > 1800:
                continue
            sources.append({"paper_id": paper.id, "version_id": version.id,
                            "title": paper.canonical_title or version.title,
                            "page": paragraph["page"], "paragraph_id": paragraph["id"],
                            "quote": quote[:750], "tokens": _tokens(quote)})
    output = []
    for title, body in _sections(draft_text):
        wanted = _tokens(title + " " + body)
        candidates = []
        for source in sources:
            shared = wanted & source["tokens"]
            if len(shared) < 2:
                continue
            score = len(shared) / max(1, len(wanted))
            candidates.append((score, source))
        candidates.sort(key=lambda item: item[0], reverse=True)
        seen: set[str] = set()
        matches = []
        for score, source in candidates:
            if source["paper_id"] in seen:
                continue
            seen.add(source["paper_id"])
            matches.append({key: value for key, value in source.items() if key != "tokens"} | {"overlap_score": round(score, 3), "quote_verified": True})
            if len(matches) >= 3:
                break
        output.append({"section": title, "matches": matches})
    return {"sections": output, "paper_count": len(papers),
            "method_note": "仅按草稿与原文段落的术语重合推荐候选；引句来自已入库 PDF，可跳回原文。支持、挑战或替代关系尚未自动断定，需人工核查。"}
