"""Source-backed topic graph for the local corpus.

This is a deterministic companion to the directed citation graph.  It uses
phrases that actually occur in titles, abstracts or section headings and does
not label co-occurrence as a citation or a scientific relationship.
"""
from __future__ import annotations

import re
from collections import defaultdict

from .research_workbench import _is_research_paper, _latest_versions

_STOP = set("the and for with from this that using based study studies approach method methods model models paper proposed novel results analysis research learning deep neural network networks large language llm llms a an of in on to by is are at as we our their into through via new towards toward can could should would all you need substantially significantly exactly existing previously further more most less than authors introduction background conclusion references discussion experiments experimental setup related work result future limitations appendix ablation task tasks".split())
_WORDS = re.compile(r"[A-Za-z][A-Za-z0-9+-]*")
_GENERIC_HEADING = re.compile(r"^(?:abstract|introduction|background|related work|method|methods|results?|experiments?|discussion|conclusions?|references|appendix|future work|limitations)(?:\s+and\s+\w+)?$", re.I)


def _phrases(text: str) -> set[str]:
    words = [w.casefold() for w in _WORDS.findall(text)]
    out: set[str] = set()
    for size in (2, 3):
        for pos in range(len(words) - size + 1):
            part = words[pos:pos + size]
            if any(len(w) < 3 or w in _STOP for w in part):
                continue
            out.add(" ".join(part))
    return out


def _heading_terms(text: str) -> set[str]:
    clean = re.sub(r"^\s*\d+(?:\.\d+)*\.?\s*", "", text).strip(" .:-")
    if not clean or _GENERIC_HEADING.fullmatch(clean):
        return set()
    words = _WORDS.findall(clean)
    terms = _phrases(clean)
    terms.update(word.casefold() for word in words
                 if len(word) >= 5 and word.casefold() not in _STOP)
    if 2 <= len(words) <= 6 and all(w.casefold() not in _STOP for w in words):
        terms.add(" ".join(w.casefold() for w in words))
    return terms


def _outline_titles(path: str) -> list[tuple[str, int]]:
    try:
        import pymupdf
        with pymupdf.open(path) as document:
            return [(title, page) for _, title, page, *_ in document.get_toc()[:100]
                    if title and page > 0]
    except (ImportError, OSError, ValueError, RuntimeError):
        return []


def _page_for_phrase(pages: list, phrase: str) -> int | None:
    for index, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            continue
        if phrase.casefold() in str(page.get("text") or "").casefold():
            number = page.get("page") or page.get("page_number") or index
            return number if isinstance(number, int) and number > 0 else index
    return None


def build_concept_graph(paper_ids: list[str] | None = None, limit: int = 20) -> dict:
    selected = []
    requested = set(paper_ids or [])
    seen: set[str] = set()
    for paper, version in _latest_versions(500):
        if paper.id in seen or not _is_research_paper(paper, version):
            continue
        if requested and paper.id not in requested:
            continue
        selected.append((paper, version))
        seen.add(paper.id)
        if len(selected) >= limit:
            break

    papers = []
    phrase_sources: dict[str, list[dict]] = defaultdict(list)
    for paper, version in selected:
        title = paper.canonical_title or version.title or version.source_filename
        metadata = (version.structured_json or {}).get("metadata") or {}
        pages = (version.structured_json or {}).get("pages") or []
        abstract = (version.abstract or metadata.get("abstract") or "")[:4500]
        sections = (version.structured_json or {}).get("sections") or []
        headings = [(str(s.get("title") or ""), s.get("page") or s.get("page_number") or 1)
                    for s in sections[:80] if isinstance(s, dict)]
        if not headings and version.stored_pdf_path.lower().endswith(".pdf"):
            headings = _outline_titles(version.stored_pdf_path)
        papers.append({"id": paper.id, "version_id": version.id, "title": title, "year": metadata.get("year")})
        evidence = [(title, "title", 1)]
        evidence += [(heading, "heading", page) for heading, page in headings]
        local: dict[str, dict] = {}
        for source_text, source_type, page in evidence:
            for phrase in _heading_terms(source_text):
                if phrase not in local:
                    local[phrase] = {"paper_id": paper.id, "version_id": version.id,
                                     "source_type": source_type, "page": page if isinstance(page, int) and page > 0 else 1,
                                     "quote": source_text[:450]}
        # Upper-case technical identifiers in an abstract are useful, unlike
        # arbitrary adjacent words which create many meaningless graph nodes.
        for acronym in set(re.findall(r"\b[A-Z][A-Z0-9-]{2,12}\b", abstract)):
            if acronym.casefold() in _STOP or acronym in {"PDF", "API", "URL"}:
                continue
            position = abstract.find(acronym)
            local.setdefault(acronym.casefold(), {"paper_id": paper.id, "version_id": version.id,
                "source_type": "abstract", "page": _page_for_phrase(pages, acronym),
                "quote": abstract[max(0, position - 160):position + len(acronym) + 160]})
        for phrase, source in local.items():
            phrase_sources[phrase].append(source)

    # Terms shared by multiple papers are especially useful, but retain
    # strong single-paper terms when the corpus is still small.
    ranked = sorted(phrase_sources, key=lambda term: (len(phrase_sources[term]), len(term)), reverse=True)[:70]
    nodes = [{"id": f"term:{term}", "label": term, "kind": "term",
              "paper_count": len(phrase_sources[term]), "evidence": phrase_sources[term]} for term in ranked]
    edges = [{"source": item["paper_id"], "target": node["id"], "relation": "mentions"}
             for node in nodes for item in node["evidence"]]
    term_papers = {node["id"]: {item["paper_id"] for item in node["evidence"]} for node in nodes}
    for i, left in enumerate(nodes):
        for right in nodes[i + 1:]:
            shared = term_papers[left["id"]] & term_papers[right["id"]]
            if len(shared) >= 2:
                edges.append({"source": left["id"], "target": right["id"],
                              "relation": "co_mentioned", "weight": len(shared)})
    return {"papers": papers, "nodes": nodes, "edges": edges,
            "summary": {"paper_count": len(papers), "term_count": len(nodes),
                        "shared_term_count": sum(node["paper_count"] > 1 for node in nodes)},
            "method_note": "术语取自本地论文标题、摘要和章节标题；同名按字面合并，共现只表示同批论文提及，不证明学术关系。"}
