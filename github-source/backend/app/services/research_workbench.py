from __future__ import annotations

import csv
import io
import json
import math
import re
import statistics
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

import networkx as nx
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import ExternalPaperRecord, Paper, PaperVersion
from .connections import mysql_engine
from .hybrid_retrieval import _llm_json
from .paper_ingestion import ensure_paper_tables


def _version_priority(version: PaperVersion) -> tuple[int, int, int, datetime]:
    label = version.version_label or ""
    number = re.fullmatch(r"v(\d+)", label, re.I)
    usable = int(bool((version.markdown or "").strip() and not version.needs_ocr))
    return (usable, int(number.group(1)) if number else -1,
            len(version.markdown or ""), version.created_at)


def _latest_versions(limit: int = 200) -> list[tuple[Paper, PaperVersion]]:
    with Session(mysql_engine()) as session:
        rows = session.execute(
            select(Paper, PaperVersion)
            .join(PaperVersion, PaperVersion.paper_id == Paper.id)
            .order_by(PaperVersion.created_at.desc())
            .limit(limit)
        ).all()
        chosen: dict[str, tuple[Paper, PaperVersion]] = {}
        for paper, version in rows:
            previous = chosen.get(paper.id)
            if previous is None or _version_priority(version) > _version_priority(previous[1]):
                chosen[paper.id] = (paper, version)
        return list(chosen.values())


def _is_research_paper(paper: Paper, version: PaperVersion) -> bool:
    """Keep administrative PDFs out of the literature graph without requiring a DOI."""
    structured = version.structured_json or {}
    metadata = structured.get("metadata") or {}
    title = (paper.canonical_title or version.title or "").strip()
    if re.search(r"签章页|签字页|封面页|申请表|身份证|certificate of registration", title, re.I):
        return False
    if paper.doi or paper.arxiv_id or (version.abstract or metadata.get("abstract")):
        return True
    if structured.get("references"):
        return True
    return len(version.markdown or "") >= 5000 and len(structured.get("sections") or []) >= 2


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[\w]+", text.casefold()))


def _year(paper: Paper, version: PaperVersion) -> int | None:
    metadata = (version.structured_json or {}).get("metadata", {})
    raw = metadata.get("year") or metadata.get("published") or ""
    match = re.search(r"\b(19|20)\d{2}\b", str(raw))
    if match:
        return int(match.group(0))
    arxiv = paper.arxiv_id or ""
    if not arxiv:
        found = re.search(r"(?i)arxiv\s*[:.]\s*(\d{4}\.\d{4,5})", (version.markdown or "")[:8000])
        arxiv = found.group(1) if found else ""
    match = re.match(r"(\d{2})(0\d|1[0-2])\.", arxiv)
    if match:
        return 2000 + int(match.group(1))
    return None


def _identifiers(text: str) -> tuple[str | None, str | None]:
    doi = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", text, re.I)
    arxiv = re.search(r"(?:arXiv:|arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5}(?:v\d+)?)|(?:arXiv:|arxiv\.org/(?:abs|pdf)/)([a-z-]+(?:\.[A-Z]{2})?/\d{7})", text, re.I)
    arxiv_raw = next((group for group in arxiv.groups() if group), None) if arxiv else None
    arxiv_id = re.sub(r"v\d+$", "", arxiv_raw.lower()) if arxiv_raw else None
    return (doi.group(0).rstrip(".,;)]}").lower() if doi else None, arxiv_id)


def _citation_strength(markdown: str, reference: dict) -> dict[str, Any]:
    """Heuristic citation-context score; expose evidence and avoid presenting it as ground truth."""
    explicit_number = re.match(r"^\s*(?:\[(\d+)\]|(\d+)[.)])\s*", str(reference.get("raw") or ""))
    number = int(explicit_number.group(1) or explicit_number.group(2)) if explicit_number else None
    raw = str(reference.get("raw") or reference.get("title") or "")
    pattern = rf"\[\s*{number}\s*\]" if number is not None else None
    body = re.split(r"(?im)^\s*(?:#{1,6}\s*)?(?:references|bibliography)\s*$", markdown, maxsplit=1)[0]
    hits = list(re.finditer(pattern, body)) if pattern else []
    contexts = [body[max(0, match.start() - 180):min(len(body), match.end() + 180)] for match in hits[:20]]
    substantive = re.compile(r"\b(we (?:adopt|follow|build|extend|use)|based on|introduced by|proposed by|as in|derive[sd]? from|outperform|improv(?:e|ed))\b", re.I)
    perfunctory = re.compile(r"\b(e\.g\.|for example|see also|such as|including|among others)\b", re.I)
    hits_substantive = sum(bool(substantive.search(context)) for context in contexts)
    hits_perfunctory = sum(bool(perfunctory.search(context)) for context in contexts)
    score = min(1.0, (0.35 if contexts else 0.0) + min(0.2, len(contexts) * 0.05) + (0.3 if hits_substantive else 0) - (0.2 if hits_perfunctory and not hits_substantive else 0))
    strength = "unverified" if not contexts else "strong" if score >= 0.65 else "medium" if score >= 0.35 else "weak"
    return {"score": round(score, 3), "strength": strength, "citation_count": len(contexts),
            "substantive_contexts": hits_substantive, "perfunctory_contexts": hits_perfunctory,
            "evidence": [" ".join(context.split())[:320] for context in contexts[:3]],
            "method": "启发式引文上下文评分，仅用于筛选，需人工复核"}


def build_citation_graph(paper_ids: list[str] | None = None, limit: int = 20, citation_strength: str = "all") -> dict[str, Any]:
    ensure_paper_tables()
    rows = _latest_versions(500)
    chosen: list[tuple[Paper, PaperVersion]] = []
    seen: set[str] = set()
    requested = set(paper_ids or [])
    for paper, version in rows:
        if not _is_research_paper(paper, version):
            continue
        if paper.id in requested or (not requested and paper.id not in seen and len(chosen) < limit):
            if paper.id not in seen:
                chosen.append((paper, version))
                seen.add(paper.id)
        if requested and requested <= seen:
            break
    title_index: dict[str, str] = {}
    identifier_index: dict[str, str] = {}
    papers: dict[str, dict[str, Any]] = {}
    for paper, version in chosen:
        key = paper.id
        title = paper.canonical_title or version.title or "Untitled paper"
        title_index[_norm(title)] = key
        for identifier in (paper.doi, paper.arxiv_id):
            if identifier:
                identifier_index[identifier.casefold()] = key
        metadata = (version.structured_json or {}).get("metadata", {})
        papers[key] = {"id": key, "title": title, "doi": paper.doi, "arxiv_id": paper.arxiv_id,
                       "year": _year(paper, version), "version_id": version.id,
                       "abstract": version.abstract or metadata.get("abstract", ""),
                       "references": (version.structured_json or {}).get("references", []) or []}

    graph = nx.DiGraph()
    for key, item in papers.items():
        graph.add_node(key, title=item["title"], year=item["year"], doi=item["doi"], arxiv_id=item["arxiv_id"])
    edges = []
    edge_pairs: set[tuple[str, str]] = set()
    unresolved: list[dict[str, Any]] = []
    for source_id, source in papers.items():
        for reference in source["references"][:500]:
            raw = str(reference.get("raw") or reference.get("title") or "").strip()
            ref_title = str(reference.get("title") or "").strip()
            doi, arxiv = _identifiers(" ".join([raw, str(reference.get("doi") or "")]))
            identifier_target = identifier_index.get((doi or "").casefold()) or identifier_index.get((arxiv or "").casefold())
            norm_title = _norm(ref_title)
            if not norm_title and raw:
                raw_norm = _norm(raw)
                ref_title = next((title for title in title_index if len(title) > 24 and title in raw_norm), "")
                norm_title = _norm(ref_title)
            title_matches = {pid for title, pid in title_index.items()
                             if len(title) > 24 and title in _norm(raw)}
            if norm_title in title_index:
                title_matches.add(title_index[norm_title])
            # A malformed reference can contain multiple bibliography entries.
            # Never assign its DOI or arXiv ID to an unrelated title.
            target = None
            if len(title_matches) == 1:
                title_target = next(iter(title_matches))
                if not identifier_target or identifier_target == title_target:
                    target = title_target
            elif not title_matches and identifier_target and len(raw) <= 350:
                target = identifier_target
            if target and target != source_id:
                if (source_id, target) in edge_pairs:
                    continue
                strength = _citation_strength(str(next((v.markdown for p, v in chosen if p.id == source_id), "")), reference)
                if not strength["citation_count"]:
                    unresolved.append({"source": source_id, "title": ref_title or raw[:200], "doi": doi,
                                       "arxiv_id": arxiv, "raw": raw[:600],
                                       "reason": "参考条目可能匹配，但正文引文尚未定位；未绘制连线"})
                    continue
                edge = {"source": source_id, "target": target, "evidence": raw[:600], "match": "identifier/title", "citation_strength": strength}
                if citation_strength == "all" or (citation_strength == "medium" and strength["strength"] in {"medium", "strong"}) or (citation_strength == "strong" and strength["strength"] == "strong"):
                    graph.add_edge(source_id, target)
                    edges.append(edge)
                    edge_pairs.add((source_id, target))
            elif raw:
                unresolved.append({"source": source_id, "title": ref_title or raw[:200], "doi": doi,
                                   "arxiv_id": arxiv, "raw": raw[:600]})

    centrality = nx.pagerank(graph) if graph.number_of_nodes() else {}
    degree = nx.degree_centrality(graph) if graph.number_of_nodes() > 1 else {node: 0.0 for node in graph}
    betweenness = nx.betweenness_centrality(graph) if graph.number_of_nodes() > 2 else {node: 0.0 for node in graph}
    undirected = graph.to_undirected()
    communities = list(nx.algorithms.community.greedy_modularity_communities(undirected)) if undirected.number_of_edges() else [{node} for node in undirected]
    community_for = {node: index + 1 for index, community in enumerate(communities) for node in community}
    ranked_by_pagerank = sorted(papers, key=lambda key: (centrality.get(key, 0), graph.in_degree(key)), reverse=True)
    core_cutoff = max(1, int(len(ranked_by_pagerank) * 0.2 + 0.999))
    core_candidates = {key for key in ranked_by_pagerank[:core_cutoff] if graph.in_degree(key) > 0}
    core_candidates.update(key for key in papers if graph.in_degree(key) >= 2)
    nonzero_in = [graph.in_degree(key) for key in papers]
    median_in = sorted(nonzero_in)[len(nonzero_in) // 2] if nonzero_in else 0
    nodes = []
    for key, item in papers.items():
        item = {name: value for name, value in item.items() if name != "references"}
        if key in core_candidates:
            role = "基石候选"
            role_reason = "局部被引次数或 PageRank 较高；仅表示当前样本文献网络中的结构中心性。"
        elif graph.out_degree(key) > 0 and graph.in_degree(key) <= median_in:
            role = "边缘衍生候选"
            role_reason = "在局部图中引用其他论文但被引较少；可能是应用或衍生研究，需结合全文核查。"
        else:
            role = "一般节点"
            role_reason = "当前网络中尚不满足基石或边缘衍生候选规则。"
        item.update({"pagerank": round(float(centrality.get(key, 0)), 6),
                     "degree_centrality": round(float(degree.get(key, 0)), 6),
                     "betweenness": round(float(betweenness.get(key, 0)), 6),
                     "in_degree": graph.in_degree(key), "out_degree": graph.out_degree(key),
                     "community_id": community_for.get(key, 0), "role": role, "role_reason": role_reason})
        nodes.append(item)
    nodes.sort(key=lambda item: (item["pagerank"], item["degree_centrality"]), reverse=True)

    # Evidence-based candidate overlap: compare source Future Work excerpts with later local abstracts/titles.
    timeline = []
    future_work = []
    for paper, version in chosen:
        markdown = version.markdown or ""
        excerpt = _extract_future_excerpt(markdown)
        if excerpt:
            tokens = _future_topic_tokens(excerpt)
            later_candidates = []
            src_year = _year(paper, version)
            if tokens:
                for other_paper, other_version in chosen:
                    other_year = _year(other_paper, other_version)
                    if other_paper.id == paper.id or not src_year or not other_year or other_year <= src_year:
                        continue
                    other_text = " ".join([other_paper.canonical_title or other_version.title or "", other_version.abstract or ""])
                    other_tokens = _future_topic_tokens(other_text)
                    shared = tokens & other_tokens
                    overlap = len(shared) / max(1, len(tokens | other_tokens))
                    if len(shared) >= 2 and overlap >= 0.1:
                        later_candidates.append({"paper_id": other_paper.id,
                            "title": other_paper.canonical_title or other_version.title,
                            "year": other_year, "doi": other_paper.doi, "arxiv_id": other_paper.arxiv_id,
                            "abstract_excerpt": (other_version.abstract or other_text)[:600],
                            "overlap_score": round(overlap, 3)})
                later_candidates.sort(key=lambda item: (item["overlap_score"], item["year"]), reverse=True)
            future_work.append({"paper_id": paper.id, "title": paper.canonical_title or version.title,
                "year": src_year, "excerpt": excerpt,
                "assessment": ("Future Work 表述过于笼统，无法定位具体研究问题" if not tokens else
                    "可能与后续研究主题重叠，需人工核查" if later_candidates else
                    "当前库中未找到足够相似的后续证据"),
                "later_candidates": later_candidates[:5]})
    by_year = Counter(str(item["year"] or "年份未知") for item in papers.values())
    timeline = [{"year": year, "papers": count} for year, count in sorted(by_year.items())]
    return {"nodes": nodes, "edges": edges, "unresolved_references": unresolved[:500],
            "timeline": timeline, "future_work": future_work,
            "summary": {"paper_count": len(nodes), "edge_count": len(edges),
                        "unresolved_count": len(unresolved), "requested_limit": limit,
                        "community_count": len(communities),
                        "matching_note": "仅按 DOI、arXiv ID 或规范化标题匹配；未匹配项保留为待核查。",
                        "citation_strength_filter": citation_strength}}


def generate_graph_review(paper_ids: list[str] | None = None, limit: int = 20) -> dict[str, Any]:
    """Summarize only inspectable graph facts, without invented method claims."""
    graph_data = build_citation_graph(paper_ids, limit, "all")
    nodes = graph_data["nodes"][:20]
    if not nodes:
        return {"status": "insufficient_sources", "overview": "文献库中还没有可综述的已解析论文。",
                "paragraphs": [], "sources": []}
    by_id = {node["id"]: node for node in nodes}
    sources = [{key: node.get(key) for key in ("id", "version_id", "title", "year", "doi", "arxiv_id", "role", "community_id")}
               for node in nodes]
    summary = graph_data["summary"]
    overview = (f"所选 {len(nodes)} 篇论文形成 {summary['edge_count']} 条已匹配的局部引用关系，"
                f"分为 {summary.get('community_count', 0)} 个连通或弱连接的社区。"
                "以下仅描述题名、参考文献匹配和图论指标能直接支持的现象。")
    paragraphs = []
    cores = [node for node in nodes if node["role"] == "基石候选"][:5]
    if cores:
        paragraphs.append({"text": "按本批次的被引边和 PageRank，较居中的论文为：" + "；".join(
            f"《{node['title']}》（被引边 {node['in_degree']}，PageRank {node['pagerank']}）" for node in cores) +
            "。这是局部网络排名，不表示全领域影响力。", "source_ids": [node["id"] for node in cores]})
    connected = [edge for edge in graph_data["edges"] if edge["source"] in by_id and edge["target"] in by_id][:6]
    if connected:
        paragraphs.append({"text": "本批次匹配到的引用关系包括：" + "；".join(
            f"《{by_id[edge['source']]['title']}》的参考文献列出《{by_id[edge['target']]['title']}》"
            for edge in connected) + "。引用关系本身不能证明方法承袭或效果优劣。",
            "source_ids": list(dict.fromkeys(id for edge in connected for id in (edge["source"], edge["target"])))})
    groups: dict[int, list[dict]] = defaultdict(list)
    for node in nodes:
        groups[node["community_id"]].append(node)
    for community_id, members in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))[:3]:
        if len(members) < 2:
            continue
        paragraphs.append({"text": f"社区 {community_id} 包含 " + "、".join(
            f"《{node['title']}》" for node in members[:5]) + "；该分组来自本批次的引用连接，不代表相同研究主题。",
            "source_ids": [node["id"] for node in members[:5]]})
    return {"status": "generated", "overview": overview, "paragraphs": paragraphs,
            "sources": sources, "graph_summary": summary}


def _extract_future_excerpt(markdown: str) -> str:
    markdown = re.split(r"(?im)^\s*(?:#{1,6}\s*)?(?:references|bibliography)\s*$", markdown, maxsplit=1)[0]
    lines = markdown.splitlines()
    heading = re.compile(r"(?i)^\s*(?:#{1,6}\s*)?(?:\d+(?:\.\d+)*\s+)?(?:future work|future directions|limitations and future work|conclusion, limitation, and future work|未来工作|未来研究|未来方向)\s*[.:]?\s*$")
    forward = re.compile(r"(?i)\b(?:future work|future research|future direction|we (?:plan|intend|aim|hope) to|remains? (?:open|unexplored)|could (?:be|also)|will (?:explore|investigate|study|focus))\b|未来(?:工作|研究|方向)")
    for index, line in enumerate(lines):
        if not heading.match(line):
            continue
        body = []
        for following in lines[index + 1:index + 35]:
            if re.match(r"(?i)^\s*(?:#{1,6}\s*)?(?:references|bibliography|appendix)\b", following):
                break
            if re.match(r"^\s*## Page \d+\s*$", following):
                continue
            body.append(following.strip())
        excerpt = " ".join(" ".join(body).split())[:1800]
        if len(excerpt) >= 80 and forward.search(excerpt):
            return excerpt
    text = " ".join(markdown.split())
    sentences = re.split(r"(?<=[.!?。！？])\s+", text)
    selected = [sentence for sentence in sentences if len(sentence) >= 80 and forward.search(sentence)]
    return " ".join(selected[:2])[:1800]


def assess_future_work_coverage(paper_ids: list[str] | None = None, limit: int = 20) -> dict[str, Any]:
    """Assess later-work evidence; only exact, explicit solution claims are filtered from open directions."""
    ensure_paper_tables()
    chosen = [row for row in _latest_versions(500) if _is_research_paper(*row)]
    requested = set(paper_ids or [])
    selected = [(paper, version) for paper, version in chosen if not requested or paper.id in requested][:limit]
    ideas = []
    for paper, version in selected:
        excerpt = _extract_future_excerpt(version.markdown or "")
        if excerpt:
            ideas.append({"paper_id": paper.id, "title": paper.canonical_title or version.title,
                          "year": _year(paper, version), "excerpt": excerpt})
    if not ideas:
        return {"status": "insufficient_sources", "assessments": [], "open_directions": [],
                "note": "所选论文中未提取到清晰的 Future Work / 未来工作段落。"}

    local_sources = []
    for paper, version in chosen:
        body = "\n".join([version.abstract or "", (version.markdown or "")[:16000]])
        local_sources.append({"id": paper.id, "title": paper.canonical_title or version.title,
            "year": _year(paper, version), "doi": paper.doi, "arxiv_id": paper.arxiv_id,
            "text": body, "url": None})
    external_sources = []
    with Session(mysql_engine()) as session:
        records = session.scalars(select(ExternalPaperRecord).order_by(ExternalPaperRecord.updated_at.desc()).limit(300)).all()
        for record in records:
            metadata = record.metadata_json or {}
            raw_year = str(metadata.get("year") or metadata.get("published") or metadata.get("updated") or "")
            match = re.search(r"\b(?:19|20)\d{2}\b", raw_year)
            abstract = str(metadata.get("abstract") or "")
            if not abstract:
                continue
            external_sources.append({"id": "external:" + record.id,
                "title": str(metadata.get("title") or record.canonical_title),
                "year": int(match.group(0)) if match else None,
                "doi": metadata.get("doi") or record.doi,
                "arxiv_id": metadata.get("arxiv_id") or metadata.get("source_id"),
                "text": abstract, "url": metadata.get("url")})

    # Keep a small, inspectable set of later works per direction.
    candidates_by_idea: dict[str, list[dict[str, Any]]] = {}
    llm_payload = []
    for idea in ideas[:12]:
        topic = _future_topic_tokens(idea["excerpt"])
        ranked = []
        for candidate in [*local_sources, *external_sources]:
            if candidate["id"] == idea["paper_id"] or not idea["year"] or not candidate["year"] or candidate["year"] <= idea["year"]:
                continue
            shared = topic & _future_topic_tokens(candidate["title"] + " " + candidate["text"][:5000])
            overlap = len(shared) / max(1, len(topic))
            if len(shared) < 2 or overlap < 0.15:
                continue
            sentences = re.split(r"(?<=[.!?。！？])\s+", candidate["text"])
            snippets = sorted(sentences, key=lambda text: len(topic & _topic_tokens(text)), reverse=True)
            evidence_text = " ".join(text.strip() for text in snippets[:4] if text.strip())[:1400]
            ranked.append({**candidate, "overlap": round(overlap, 3), "evidence_text": evidence_text})
        ranked.sort(key=lambda item: (item["overlap"], item["year"] or 0), reverse=True)
        selected_candidates = ranked[:3]
        candidates_by_idea[idea["paper_id"]] = selected_candidates
        llm_payload.append({"future_work": idea, "later_candidates": [
            {key: candidate.get(key) for key in ("id", "title", "year", "doi", "arxiv_id", "url", "overlap", "evidence_text")}
            for candidate in selected_candidates]})

    model_assessments: dict[str, dict] = {}
    model_error = None
    model_attempted = any(item["later_candidates"] for item in llm_payload)
    if model_attempted:
        prompt = (
            "比较论文 Future Work 与给定的后续研究证据。只可基于提供文本。返回严格 JSON："
            "{assessments:[{future_paper_id:string,status:'direct_coverage'|'possible_progress'|'no_evidence',reason:string,evidence:[{source_id:string,quote:string}]}]}。"
            "direct_coverage 仅当后续来源明确表示其直接解决/处理了同一个具体问题时使用，并复制逐字原文作为 quote；"
            "一般主题相似、提出相关方法、性能提升或引用关系都不能判为 direct_coverage。"
            "没有清楚逐字证据时用 possible_progress 或 no_evidence。不得声称某问题已被彻底解决。\n数据(JSON):\n" +
            json.dumps(llm_payload, ensure_ascii=False))
        try:
            parsed = _llm_json(
                "You are a cautious research evidence assessor. Return valid JSON only.",
                prompt, max_tokens=2600)
            model_assessments = {str(item.get("future_paper_id")): item for item in parsed.get("assessments", [])}
        except Exception as exc:
            model_error = exc.__class__.__name__

    explicit_resolution = re.compile(r"\b(?:solve[sd]?|address(?:ed|es)?|resolve[sd]?|overcome|eliminate[sd]?|mitigate[sd]?)\b|解决了|解决该|解决此|克服了|消除了|处理了该问题", re.I)
    assessments = []
    open_directions = []
    for idea in ideas[:12]:
        model_item = model_assessments.get(idea["paper_id"], {})
        candidate_map = {item["id"]: item for item in candidates_by_idea.get(idea["paper_id"], [])}
        verified_evidence = []
        for evidence in model_item.get("evidence", [])[:5]:
            candidate = candidate_map.get(str(evidence.get("source_id")))
            quote = " ".join(str(evidence.get("quote", "")).split())
            if candidate and quote and _norm(quote) in _norm(candidate["text"]):
                verified_evidence.append({"source_id": candidate["id"], "title": candidate["title"],
                    "year": candidate["year"], "doi": candidate["doi"], "arxiv_id": candidate["arxiv_id"],
                    "url": candidate["url"], "quote": quote, "overlap_score": candidate["overlap"]})
        model_status = model_item.get("status")
        if model_status == "direct_coverage" and any(explicit_resolution.search(item["quote"]) for item in verified_evidence):
            status = "direct_coverage_evidence"
            label = "后续论文存在直接处理该问题的原文证据；建议人工核对适用范围"
        elif not _future_topic_tokens(idea["excerpt"]):
            status = "unspecified_direction"
            label = "原文只提出笼统计划，未说明具体研究问题，无法判断后续覆盖"
        elif candidate_map:
            status = "possible_progress"
            label = "发现较新主题相关研究，但没有足够直接证据判定该方向已被覆盖"
        else:
            status = "no_later_evidence"
            label = "当前本地库与已缓存元数据中未找到足够的后续证据"
        item = {**idea, "status": status, "assessment": label,
                "model_reason": str(model_item.get("reason", ""))[:600],
                "candidates": [{key: candidate.get(key) for key in ("id", "title", "year", "doi", "arxiv_id", "url", "overlap")}
                    for candidate in candidates_by_idea.get(idea["paper_id"], [])],
                "verified_evidence": verified_evidence}
        assessments.append(item)
        if status not in {"direct_coverage_evidence", "unspecified_direction"}:
            open_directions.append(item)
    return {"status": "assessed" if model_attempted and model_error is None else "evidence_only",
            "assessments": assessments, "open_directions": open_directions,
            "note": "只有后续来源中的直接原文证据被定位后，才从待探索方向中过滤；此结果不等同于宣布问题已彻底解决。" +
                (f" 模型判断不可用（{model_error}），仅显示主题候选。" if model_error else
                 " 当前未找到可供模型核查的后续候选。" if not model_attempted else ""),
            "source_scope": "已选本地论文全文/摘要 + 已缓存外部论文摘要；覆盖范围受本地库和跟踪缓存限制。"}


_QWEN_CODE_RL_REFERENCES = [
    {"title": "Qwen2.5-Coder Technical Report", "year": 2024, "arxiv_id": "2409.12186",
     "authors": ["Binyuan Hui", "Jian Yang", "Zeyu Cui", "Jiaxi Yang", "Dayiheng Liu", "Lei Zhang", "Tianyu Liu", "Jiajun Zhang", "Bowen Yu", "Keming Lu", "Kai Dang", "Yang Fan", "Yichang Zhang", "An Yang", "Rui Men", "Fei Huang", "Bo Zheng", "Yibo Miao", "Shanghaoran Quan", "Yunlong Feng", "Xingzhang Ren", "Xuancheng Ren", "Jingren Zhou", "Junyang Lin"],
     "scope": "介绍 Qwen2.5-Coder 系列及其代码生成、补全、推理和修复评估；系列包含 0.5B 与 1.5B 等较小参数规模。"},
    {"title": "CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning", "year": 2022, "arxiv_id": "2207.01780",
     "authors": ["Hung Le", "Yue Wang", "Akhilesh Deepak Gotmare", "Silvio Savarese", "Steven C. H. Hoi"],
     "scope": "研究在预训练代码生成模型上利用强化学习与测试反馈进行程序合成。"},
    {"title": "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", "year": 2024, "arxiv_id": "2402.03300",
     "authors": ["Zhihong Shao", "Peiyi Wang", "Qihao Zhu", "Runxin Xu", "Junxiao Song", "Xiao Bi", "Haowei Zhang", "Mingchuan Zhang", "Y. K. Li", "Y. Wu", "Daya Guo"],
     "scope": "提出 Group Relative Policy Optimization（GRPO）用于数学推理；迁移到代码任务仍需独立验证。"},
    {"title": "Measuring Coding Challenge Competence With APPS", "year": 2021, "arxiv_id": "2105.09938",
     "authors": ["Dan Hendrycks", "Steven Basart", "Saurav Kadavath", "Mantas Mazeika", "Akul Arora", "Ethan Guo", "Collin Burns", "Samir Puranik", "Horace He", "Dawn Song", "Jacob Steinhardt"],
     "scope": "提出 APPS 代码生成基准，按测试用例评估由自然语言问题生成的 Python 程序。"},
]


def _is_qwen_code_rl_idea(idea: str) -> bool:
    return (bool(re.search(r"qwen|通义", idea, re.I))
            and bool(re.search(r"强化学习|\brl\b|reinforcement learning", idea, re.I))
            and bool(re.search(r"代码|编程|coding|code|program", idea, re.I)))


def _writing_terms(text: str) -> set[str]:
    terms = set(re.findall(r"[a-z][a-z0-9]+", text.casefold()))
    for pattern, words in (
        (r"强化学习", {"reinforcement", "learning", "rl", "policy"}),
        (r"代码|编程", {"code", "coding", "program", "programming", "generation"}),
        (r"开放式", {"open", "ended", "openended"}),
        (r"轻量|小模型", {"small", "lightweight", "efficient"}),
        (r"测试", {"test", "tests", "testing"}),
    ):
        if re.search(pattern, text, re.I):
            terms.update(words)
    return terms - {"the", "and", "for", "with", "from", "this", "that", "using", "paper"}


def _source_records(idea: str, maximum: int) -> list[dict[str, Any]]:
    ensure_paper_tables()
    terms = _writing_terms(idea)
    ranked = []
    seen_keys: set[str] = set()
    with Session(mysql_engine()) as session:
        versions = session.execute(select(PaperVersion).order_by(PaperVersion.created_at.desc()).limit(500)).scalars().all()
        preferred: dict[str, PaperVersion] = {}
        for version in versions:
            previous = preferred.get(version.paper_id)
            if previous is None or _version_priority(version) > _version_priority(previous):
                preferred[version.paper_id] = version
        for version in preferred.values():
            paper = session.get(Paper, version.paper_id)
            if not paper or not _is_research_paper(paper, version):
                continue
            title = paper.canonical_title or version.title or ""
            unique_key = (paper.doi or paper.arxiv_id or _norm(title)).casefold()
            if unique_key in seen_keys:
                continue
            seen_keys.add(unique_key)
            body = " ".join([title, version.abstract or "", version.markdown[:6000] if version.markdown else ""])
            overlap = len(terms & _writing_terms(body))
            metadata = (version.structured_json or {}).get("metadata", {})
            ranked.append((overlap, {"ref_id": f"P{len(ranked)+1}", "paper_id": paper.id,
                "title": title, "authors": version.authors or metadata.get("authors", []),
                "year": _year(paper, version), "doi": paper.doi or metadata.get("doi"),
                "arxiv_id": paper.arxiv_id or metadata.get("arxiv_id"),
                "abstract": version.abstract or metadata.get("abstract", ""),
                "source": "文献库 MySQL 元数据", "source_url": f"https://doi.org/{paper.doi}" if paper.doi else f"https://arxiv.org/abs/{paper.arxiv_id}" if paper.arxiv_id else None,
                "verified": bool(title and (paper.doi or paper.arxiv_id)),
                "version_id": version.id}))
        external_records = session.execute(
            select(ExternalPaperRecord).order_by(ExternalPaperRecord.updated_at.desc()).limit(300)
        ).scalars().all()
        for record in external_records:
            metadata = record.metadata_json or {}
            title = str(metadata.get("title") or record.canonical_title or "").strip()
            sources = record.sources_json or []
            if not title or not sources:
                continue
            doi = metadata.get("doi") or record.doi
            arxiv_id = metadata.get("arxiv_id") or metadata.get("arxiv")
            unique_key = str(doi or arxiv_id or _norm(title)).casefold()
            if unique_key in seen_keys:
                continue
            seen_keys.add(unique_key)
            authors = metadata.get("authors") or []
            if isinstance(authors, str):
                authors = [authors]
            abstract = str(metadata.get("abstract") or "")
            overlap = len(terms & _writing_terms(" ".join([title, abstract])))
            ranked.append((overlap, {"ref_id": f"P{len(ranked)+1}", "paper_id": record.paper_id,
                "external_record_id": record.id, "title": title, "authors": authors,
                "year": metadata.get("year"), "doi": doi, "arxiv_id": arxiv_id,
                "abstract": abstract, "source": ", ".join(str(item.get("source", "metadata")) for item in sources if isinstance(item, dict)),
                "source_url": metadata.get("url") or (f"https://doi.org/{doi}" if doi else f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None),
                "verified": bool(doi or arxiv_id), "version_id": record.version_id}))
    if _is_qwen_code_rl_idea(idea):
        for seed in _QWEN_CODE_RL_REFERENCES:
            unique_key = seed["arxiv_id"].casefold()
            ranked = [(score, record) for score, record in ranked
                      if str(record.get("arxiv_id") or "").casefold() != unique_key]
            seen_keys.add(unique_key)
            ranked.append((10 + len(terms & _writing_terms(seed["title"])), {
                "ref_id": "", "paper_id": None, "title": seed["title"], "authors": seed["authors"],
                "year": seed["year"], "doi": None, "arxiv_id": seed["arxiv_id"],
                "abstract": seed["scope"], "source": "arXiv 官方条目（预核验元数据）",
                "source_url": f"https://arxiv.org/abs/{seed['arxiv_id']}",
                "verified": True, "version_id": None, "scope": seed["scope"]}))
    ranked.sort(key=lambda item: item[0], reverse=True)
    for rank, (_, record) in enumerate(ranked[:maximum], start=1):
        record["ref_id"] = f"P{rank}"
    return [record for score, record in ranked[:maximum] if score > 0 or not terms]


def generate_research_draft(idea: str, max_sources: int = 8) -> dict[str, Any]:
    sources = _source_records(idea, max_sources)
    verified = [item for item in sources if item["verified"]]
    identifiers = {str(item.get("arxiv_id") or ""): item["ref_id"] for item in verified}
    qwen_code_rl = _is_qwen_code_rl_idea(idea)

    def block(text: str, ref_ids: list[str] | None = None, status: str = "研究方案，待实验验证") -> dict[str, Any]:
        return {"text": text, "ref_ids": [ref_id for ref_id in (ref_ids or []) if ref_id in {item["ref_id"] for item in verified}],
                "evidence_status": status}

    def anchor(arxiv_id: str) -> list[str]:
        return [identifiers[arxiv_id]] if arxiv_id in identifiers else []

    if qwen_code_rl:
        title = "面向开放式代码问题的轻量化 Qwen 模型强化学习后训练框架"
        abstract = [block("开放式代码问题需要生成可执行、可验证且能适应未见问题的程序。本文拟选用轻量化开源 Qwen 代码模型作为策略网络，"
                          "利用隔离执行环境中的测试反馈构建奖励，并研究强化学习后训练对代码正确性与训练成本的影响。"
                          "评估计划包括与原始模型及监督微调基线比较，并报告通过率、推理开销和失败类型。"
                          "[待填：数据集、具体模型尺寸、训练配置及真实实验结果。]"),
                    ]
        introduction = [
            block("自然语言描述的编程题可以通过执行测试检查生成程序的行为；因此，开放式代码问题适合作为可验证生成任务。",
                  anchor("2105.09938"), "来源支持任务与评估方式；具体研究动机仍需论证"),
            block("Qwen2.5-Coder 系列提供多个代码专用模型尺寸，为选择较小参数规模的策略模型提供候选。",
                  anchor("2409.12186"), "来源支持模型系列与尺寸；本研究所选检查点待确定"),
            block("已有 CodeRL 工作研究了将强化学习与代码生成模型、测试反馈结合。本研究拟进一步关注轻量化 Qwen、开放式题目和计算预算的共同约束；差异性与新颖性仍需逐篇核查。",
                  anchor("2207.01780"), "前句关联已核验来源；研究空白为待验证假设"),
            block("拟构建“题目采样 → Qwen 生成 → 沙箱执行 → 奖励计算 → 策略更新”的训练闭环，"
                  "并在固定计算预算下比较测试通过率与成本。[待填：奖励公式、RL 算法、数据划分和统计方案。]")]
        related = [
            block("模型基础：Qwen2.5-Coder Technical Report 介绍了代码模型系列；本研究应据具体权重、许可和资源预算选定轻量化检查点。",
                  anchor("2409.12186"), "文献题名与系列信息可回查"),
            block("代码强化学习：CodeRL 讨论了预训练模型、强化学习和测试反馈的组合；需要比较其训练目标与本方案的区别。",
                  anchor("2207.01780"), "文献主题可回查；方法优劣尚未判断"),
            block("优化方法：DeepSeekMath 提出了 GRPO，但研究对象是数学推理；将该方法用于代码任务仅是候选设计，不能据此声称有效。",
                  anchor("2402.03300"), "来源支持 GRPO 的出处；迁移效果待实验"),
            block("评价基准：APPS 提供自然语言代码问题及基于测试用例的评价；是否适用于目标“开放式”场景需明确任务边界。",
                  anchor("2105.09938"), "来源支持基准信息；适用性待核查")]
    else:
        title = f"研究框架：{idea.strip()[:100]}"
        abstract = [block(f"本研究拟围绕“{idea.strip()[:220]}”建立方法与评估框架。"
                          "[待填：明确研究问题、数据来源、方法细节、比较基线与实测结果。] 当前未写入任何未经验证的结果。")]
        introduction = [block("界定研究对象、任务边界与读者需要解决的实际问题；背景事实应由可核验来源支持。"),
                        block("比较现有方法，并将拟解决的差距标为研究假设，不能直接写成已证实结论。"),
                        block(f"提出针对“{idea.strip()[:180]}”的拟议方法与验证路线，列明关键假设和失败条件。")]
        related = [block(f"候选来源《{item['title']}》可用于相关工作筛选；目前仅确认其元数据与标识符，具体方法和结论须回查摘要及全文。",
                         [item["ref_id"]], "元数据可定位；正文论断待核查") for item in verified[:6]]
    if not verified:
        related = [block("尚无带 DOI 或 arXiv ID 的可核验文献；References 留空，先导入或检索来源。", status="证据不足")]

    sections = {"Abstract": abstract, "Introduction": introduction, "Related Work": related}
    references = []
    for item in verified:
        reference = {key: item.get(key) for key in ("ref_id", "title", "authors", "year", "doi", "arxiv_id", "source", "source_url", "version_id")}
        reference["authors"] = reference["authors"] if isinstance(reference["authors"], list) else []
        references.append(reference)
    markdown_lines = [f"# {title}"]
    for name, blocks in sections.items():
        markdown_lines.extend(["", f"## {name}"])
        for item in blocks:
            citations = " " + " ".join(f"[{ref_id}]" for ref_id in item["ref_ids"]) if item["ref_ids"] else ""
            markdown_lines.extend(["", item["text"] + citations])
    markdown_lines.extend(["", "## References", ""])
    if references:
        for item in references:
            authors = item.get("authors") or []
            author_text = ", ".join(authors[:3]) + (", et al." if len(authors) > 3 else "") if authors else "作者待核查"
            identifier = f"doi:{item['doi']}" if item.get("doi") else f"arXiv:{item['arxiv_id']}"
            markdown_lines.append(f"[{item['ref_id']}] {author_text}. {item['title']}. {item.get('year') or '年份待核查'}. {identifier}. {item.get('source_url') or ''}")
    else:
        markdown_lines.append("[待补：可核验来源。]")
    return {"idea": idea, "title": title, "sections": sections, "sources": verified,
            "outline": ["Abstract：研究问题、拟议方法、评估计划、结果占位", "Introduction：任务背景、现有工作、研究假设、贡献计划",
                        "Related Work：模型基础、代码强化学习、优化算法、评价基准", "Method：模型、奖励、策略更新与沙箱约束（待细化）",
                        "Experiments：基线、划分、通过率、成本和消融（待实测）"],
            "review_paragraphs": related, "references": references, "markdown": "\n".join(markdown_lines),
            "missing_inputs": ["具体 Qwen 检查点及许可证", "开放式代码问题的定义与数据集", "奖励函数与 RL 算法",
                               "隔离执行环境和预算", "基线、评估指标及实际实验结果"] if qwen_code_rl else
                              ["任务定义与数据集", "方法细节", "基线与真实实验结果"],
            "evidence_status": "参考文献仅含可定位 DOI/arXiv 条目；研究结果保留占位" if references else
                               "暂无可核验参考文献；已生成无结果断言的框架",
            "generation_note": "框架由受控模板生成；元数据可核查，研究假设、方法建议和未来实验结果仍需作者验证。"}


def _csv_number(value: Any) -> float | None:
    raw = str(value or "").strip().replace(",", "")
    if not raw:
        return None
    try:
        number = float(raw)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _csv_aggregate(values: list[float], method: str) -> float:
    if method == "count":
        return float(len(values))
    if method == "sum":
        return float(sum(values))
    if method == "median":
        return float(statistics.median(values))
    return float(statistics.mean(values))


def normalize_tabular_upload(filename: str, content: bytes) -> dict[str, str]:
    if not content or len(content) > 2_000_000:
        raise ValueError("表格文件不能为空且不能超过 2 MB")
    if content.startswith(b"PK\x03\x04"):
        from openpyxl import load_workbook
        try:
            workbook = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
            sheet = workbook.worksheets[0]
            output = io.StringIO(newline="")
            writer = csv.writer(output)
            for index, row in enumerate(sheet.values):
                if index > 100_000:
                    raise ValueError("工作表最多支持 100,000 行")
                writer.writerow(["" if value is None else str(value) for value in row])
            workbook.close()
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("文件包含 Excel 数据，但工作簿无法解析") from exc
        return {"filename": filename, "csv_text": output.getvalue(), "detected_format": "xlsx", "sheet": sheet.title}
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return {"filename": filename, "csv_text": content.decode(encoding), "detected_format": "csv", "sheet": ""}
        except UnicodeDecodeError:
            continue
    raise ValueError("无法识别文件编码；请另存为 UTF-8 CSV 或 XLSX")


def analyze_csv(filename: str, csv_text: str, x_column: str | None = None,
                y_column: str | None = None, chart_type: str = "bar", aggregation: str = "mean",
                radar_columns: list[str] | None = None) -> dict[str, Any]:
    """Calculate all figures and captions from the same parsed CSV rows, without LLM estimates."""
    try:
        reader = csv.DictReader(io.StringIO(csv_text.lstrip("\ufeff")))
        columns = list(reader.fieldnames or [])
        if not columns or any(not name or not name.strip() for name in columns) or len(set(columns)) != len(columns):
            raise ValueError("CSV 需要非空且不重复的表头")
        rows = list(reader)
    except csv.Error as exc:
        raise ValueError(f"CSV 格式无效：{exc}") from exc
    if not rows:
        raise ValueError("CSV 中没有数据行")
    if len(rows) > 100_000:
        raise ValueError("CSV 最多支持 100,000 行")
    if any(None in row for row in rows):
        raise ValueError("CSV 数据行的列数超过表头，请检查分隔符与引号")

    profiles = []
    numeric_columns = []
    numeric_fallback = []
    for column in columns:
        present = [str(row.get(column) or "").strip() for row in rows if str(row.get(column) or "").strip()]
        parsed = [_csv_number(value) for value in present]
        numeric_ratio = sum(value is not None for value in parsed) / len(present) if present else 0
        is_numeric = numeric_ratio >= 0.95
        distinct = len(set(present))
        identifier = bool(re.search(r"(^id$|(?:^|_)id$|identifier|编号$)", column, re.I))
        category = not is_numeric or (distinct <= max(12, int(math.sqrt(len(rows)))) and distinct < len(rows) / 2)
        kind = "identifier" if identifier else "category" if category else "numeric"
        profiles.append({"name": column, "kind": kind, "nonempty": len(present), "distinct": distinct,
                         "numeric_ratio": round(numeric_ratio, 3)})
        if is_numeric and not identifier:
            numeric_fallback.append(column)
            if not category:
                numeric_columns.append(column)
    if not numeric_columns:
        numeric_columns = numeric_fallback
    if not numeric_columns:
        raise ValueError("CSV 中没有可绘图的数值指标")

    profile_map = {item["name"]: item for item in profiles}
    categorical = [item["name"] for item in profiles if item["kind"] == "category"]
    preferred = next((name for name in categorical if re.search(r"niche|group|category|class|类别|分组", name, re.I)), None)
    if chart_type == "stacked_bar" and x_column is None:
        x_column = next((name for name in columns if re.search(r"^time$|^时间$|^date$", name, re.I)), None)
    x_column = x_column if x_column in columns else (preferred or (categorical[0] if categorical else columns[0]))
    y_column = y_column or next((name for name in numeric_columns if name != x_column), numeric_columns[0])
    if y_column not in numeric_columns:
        raise ValueError(f"指标列“{y_column}”不是可计算的数值列")
    if x_column == y_column:
        raise ValueError("分组/横轴列和指标列不能相同")
    if aggregation not in {"mean", "sum", "median", "count"}:
        raise ValueError("不支持的汇总方式")
    if chart_type not in {"bar", "line", "scatter", "table", "radar3d", "stacked_bar"}:
        raise ValueError("不支持的图表类型")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    valid_values = []
    for row in rows:
        value = _csv_number(row.get(y_column))
        if value is None:
            continue
        label = str(row.get(x_column) or "").strip() or "缺失标签"
        grouped[label].append(row)
        valid_values.append(value)
    if not valid_values:
        raise ValueError(f"指标列“{y_column}”没有有效数值")
    ordered_labels = list(grouped)
    if profile_map[x_column]["numeric_ratio"] >= 0.95:
        ordered_labels.sort(key=lambda label: _csv_number(label) if _csv_number(label) is not None else float("inf"))
    elif re.search(r"date|time|year|month|日期|时间|年份", x_column, re.I):
        ordered_labels.sort()
    else:
        ordered_labels.sort(key=lambda label: (-len(grouped[label]), label))

    table_rows = []
    for label in ordered_labels:
        values = [_csv_number(row.get(y_column)) for row in grouped[label]]
        clean = [value for value in values if value is not None]
        table_rows.append({"label": label[:160], "count": len(clean), "value": _csv_aggregate(clean, aggregation),
                           "mean": statistics.mean(clean), "median": statistics.median(clean),
                           "min": min(clean), "max": max(clean),
                           "stdev": statistics.stdev(clean) if len(clean) > 1 else None})
    chart_points = [{"x": item["label"], "y": item["value"], "count": item["count"]} for item in table_rows[:500]]
    stats = {"count": len(valid_values), "missing": len(rows) - len(valid_values), "sum": sum(valid_values),
             "mean": statistics.mean(valid_values), "median": statistics.median(valid_values),
             "min": min(valid_values), "max": max(valid_values),
             "stdev": statistics.stdev(valid_values) if len(valid_values) > 1 else 0.0}

    allowed_radar = [name for name in numeric_columns if name != x_column and profile_map[name]["kind"] == "numeric"]
    chosen_radar = list(dict.fromkeys(name for name in (radar_columns or []) if name in allowed_radar))[:6]
    if not chosen_radar:
        chosen_radar = allowed_radar[:5]
    radar_groups = []
    if len(chosen_radar) >= 3:
        for label in sorted(ordered_labels, key=lambda name: (-len(grouped[name]), name))[:6]:
            metrics = {}
            for column in chosen_radar:
                values = [_csv_number(row.get(column)) for row in grouped[label]]
                clean = [value for value in values if value is not None]
                metrics[column] = statistics.mean(clean) if clean else None
            if all(value is not None for value in metrics.values()):
                radar_groups.append({"label": label[:160], "count": len(grouped[label]), "metrics": metrics})
        for column in chosen_radar:
            values = [group["metrics"][column] for group in radar_groups if group["metrics"][column] is not None]
            lower, upper = (min(values), max(values)) if values else (0.0, 0.0)
            for group in radar_groups:
                raw = group["metrics"][column]
                group.setdefault("normalized", {})[column] = (raw - lower) / (upper - lower) if upper > lower else 0.5

    high = max(table_rows, key=lambda item: item["value"])
    low = min(table_rows, key=lambda item: item["value"])
    method_name = {"mean": "均值", "sum": "总和", "median": "中位数", "count": "有效记录数"}[aggregation]
    base = f"{filename} 中 {len(rows)} 行记录，{y_column} 有效值 {len(valid_values)} 个、缺失或无效值 {stats['missing']} 个。"
    stacked_series = []
    if chart_type == "stacked_bar":
        species_columns = [name for name in numeric_columns if name != x_column and re.search(r"species|物种", name, re.I)][:5]
        if len(species_columns) < 2:
            raise ValueError("相对丰度堆叠柱状图需要至少两个物种数值列")
        for label in ordered_labels:
            totals = {name: sum(_csv_number(row.get(name)) or 0 for row in grouped[label]) for name in species_columns}
            total = sum(totals.values())
            if total > 0:
                stacked_series.append({"label": label, "values": {name: value / total * 100 for name, value in totals.items()},
                                       "sample_count": len(grouped[label])})
        if not stacked_series:
            raise ValueError("所选物种列在各时间点没有可计算的丰度")
        caption = (f"图 1. {len(species_columns)} 个物种按 {x_column} 分组的相对丰度堆叠柱状图（{len(stacked_series)} 个时间点）。"
                   f"各时间点先汇总 {', '.join(species_columns)} 的丰度，再以该时间点这 {len(species_columns)} 个物种的总丰度为分母归一化至 100%；"
                   "颜色代表不同物种。该图仅描述所选物种的组成变化，不代表全部物种群落，也未进行显著性检验。")
    elif chart_type == "radar3d":
        if len(chosen_radar) < 3 or len(radar_groups) < 2:
            raise ValueError("三维雷达图需要至少 3 个数值指标和 2 个分组")
        focus_metric = y_column if y_column in chosen_radar else chosen_radar[0]
        focus_groups = [group for group in radar_groups if group["metrics"][focus_metric] is not None]
        focus_high = max(focus_groups, key=lambda group: group["metrics"][focus_metric]) if focus_groups else None
        focus_low = min(focus_groups, key=lambda group: group["metrics"][focus_metric]) if focus_groups else None
        comparison = (f"其中 {focus_metric} 的组均值以 {focus_high['label']} 最高（{focus_high['metrics'][focus_metric]:.4g}），"
                      f"{focus_low['label']} 最低（{focus_low['metrics'][focus_metric]:.4g}）。") if focus_high and focus_low else ""
        caption = (f"图 1. 按 {x_column} 分组的多指标均值轮廓（展示所选指标均有数值的 {len(radar_groups)} 组）。"
                   f"{comparison}各轴基于展示组的指标均值独立进行最小－最大归一化；图形面积不能用于比较不同指标的绝对量。"
                   f"{base}仅展示描述性差异，未进行显著性检验。")
    elif chart_type == "table":
        caption = (f"表 1. {y_column} 按 {x_column} 的分组描述性统计（{method_name}）。"
                   f"共 {len(table_rows)} 组；最高为 {high['label']}（{high['value']:.4g}），最低为 {low['label']}（{low['value']:.4g}）。{base}")
    else:
        chart_name = {"line": "折线图", "scatter": "散点图", "bar": "柱状图"}[chart_type]
        ordered_axis = profile_map[x_column]["numeric_ratio"] >= 0.95 or bool(re.search(r"date|time|year|month|日期|时间|年份", x_column, re.I))
        endpoint_note = (f"按横轴顺序，首组 {table_rows[0]['label']} 为 {table_rows[0]['value']:.4g}，"
                         f"末组 {table_rows[-1]['label']} 为 {table_rows[-1]['value']:.4g}；区间内可能有波动。") if chart_type == "line" and ordered_axis and len(table_rows) > 1 else ""
        caption = (f"图 1. {y_column} 按 {x_column} 分组的{method_name}{chart_name}。"
                   f"共 {len(table_rows)} 组，最高为 {high['label']}（{high['value']:.4g}），最低为 {low['label']}（{low['value']:.4g}）。"
                   f"{endpoint_note}{base}"
                   + ("横轴为类别，连线仅辅助比较，不表示时间趋势。" if chart_type == "line" and profile_map[x_column]["kind"] == "category" else ""))
    return {"filename": filename, "columns": columns, "column_profiles": profiles,
            "numeric_columns": numeric_columns, "row_count": len(rows), "x_column": x_column,
            "y_column": y_column, "chart_type": chart_type, "aggregation": aggregation,
            "points": chart_points, "table_rows": table_rows[:500], "statistics": stats,
            "radar_columns": chosen_radar, "radar_groups": radar_groups,
            "stacked_columns": species_columns if chart_type == "stacked_bar" else [], "stacked_series": stacked_series,
            "caption": caption, "truncated_for_chart": len(table_rows) > 500,
            "preview_rows": [{column: str(row.get(column) or "")[:160] for column in columns} for row in rows[:20]],
            "method_note": "图注与图形由同一批有效 CSV 数值计算；不作因果或统计显著性推断。"}


def _topic_tokens(text: str) -> set[str]:
    result: set[str] = set()
    for match in re.findall(r"[a-z]{4,}|[\u3400-\u9fff]+", text.casefold()):
        if re.fullmatch(r"[\u3400-\u9fff]+", match):
            width = min(2, len(match))
            result.update(match[index:index + width] for index in range(len(match) - width + 1))
        else:
            result.add(match)
    return result


_FUTURE_GENERIC_WORDS = {
    "about", "after", "also", "among", "another", "approach", "based", "been", "before",
    "could", "direction", "directions", "further", "future", "have", "into", "investigate",
    "more", "need", "next", "other", "paper", "possible", "potential", "research", "should",
    "some", "study", "than", "that", "their", "there", "these", "this", "those", "through",
    "using", "with", "work", "would", "will", "plan", "plans", "intend", "intends", "hope",
    "我们的", "未来", "研究", "工作", "方向", "进一步", "方法", "可能", "进行",
}


def _future_topic_tokens(text: str) -> set[str]:
    tokens = _topic_tokens(text) - _FUTURE_GENERIC_WORDS
    return tokens if len(tokens) >= 3 else set()


def mermaid_architecture(idea: str) -> dict[str, Any]:
    inferred_pairs: set[tuple[int, int]] = set()
    if _is_qwen_code_rl_idea(idea):
        labels = ["Open coding tasks", "Lightweight Qwen policy", "Sample candidate code",
                  "Isolated sandbox + tests", "Executable reward", "RL policy update",
                  "Held-out evaluation", "Failure analysis"]
        coordinates = [(0, 2), (2.3, 2), (4.6, 2), (6.9, 2), (9.2, 2),
                       (9.2, 0), (11.5, 2), (13.8, 2)]
        pairs = [(0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (1, 6), (6, 7)]
        title = "Lightweight Qwen RL for open-ended code problems"
    else:
        from .model_adapters import get_llm_client
        from ..config import get_settings
        error = ""
        for attempt in range(3):
            response = get_llm_client().with_options(timeout=180).chat.completions.create(
                model=get_settings().llm_model,
                messages=[{"role": "system", "content": "你是科研技术路线图设计师。仅依据用户方案提取具体流程，不补造实验结果或疗效。输出严格 JSON：{\"title\":\"中文标题\",\"nodes\":[{\"label\":\"中文节点\",\"layer\":0}],\"edges\":[[0,1]]}。设计 10–24 个节点和有效有向边；layer 为 0–6。节点须体现原文关键术语与分支。edges 最好用从 0 开始的节点下标；也接受节点 ID 或完整名称。"},
                          {"role": "user", "content": idea[:11000] + (f"\n上次结构校验错误：{error}。请修正全部连线。" if error else "")}],
                temperature=0.1, max_tokens=5000, extra_body={"thinking": {"type": "disabled"}},
            )
            try:
                raw = (response.choices[0].message.content or "").strip()
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
                design = json.loads(raw)
                items = design.get("nodes", [])
                links = design.get("edges", [])
                if not 8 <= len(items) <= 28 or not len(items) - 1 <= len(links) <= 70:
                    raise ValueError("节点或连线数量不符合要求")
                labels = [str(item.get("label") or item.get("name") or "").strip()[:40] for item in items]
                layers = [int(item.get("layer", 0)) for item in items]
                if any(not label for label in labels) or any(layer < 0 or layer > 6 for layer in layers):
                    raise ValueError("存在空节点或无效层级")
                ids = {str(item.get("id", f"N{index + 1}")): index for index, item in enumerate(items)}
                names = {label: index for index, label in enumerate(labels)}
                endpoints = [(edge.get("source", edge.get("from")), edge.get("target", edge.get("to")))
                             if isinstance(edge, dict) else (edge[0], edge[1]) for edge in links]
                numeric = [int(value) for edge in endpoints for value in edge if isinstance(value, int) or str(value).isdigit()]
                one_based = bool(numeric) and 0 not in numeric and max(numeric) == len(labels)
                def resolve(value: Any) -> int:
                    key = str(value).strip()
                    if key in names:
                        return names[key]
                    if key in ids:
                        return ids[key]
                    if key.isdigit():
                        return int(key) - (1 if one_based else 0)
                    raise ValueError(f"无法识别的节点：{key[:40]}")
                pairs = []
                rejected = 0
                for source, target in endpoints:
                    try:
                        start, end = resolve(source), resolve(target)
                        if start < 0 or end < 0 or start >= len(labels) or end >= len(labels) or start == end:
                            raise ValueError("节点下标超出范围或指向自身")
                        if (start, end) not in pairs:
                            pairs.append((start, end))
                    except (ValueError, TypeError):
                        rejected += 1
                if (rejected or len(pairs) < len(labels) - 1) and attempt < 2:
                    raise ValueError(f"{rejected} 条连线无法定位；仅保留 {len(pairs)} 条，请用节点 ID 修正")
                if len(pairs) < len(labels) - 1:
                    # Keep the model's valid links. Complete only the visual flow and mark inferred links.
                    ordered = sorted(range(len(labels)), key=lambda index: (layers[index], index))
                    for start, end in zip(ordered, ordered[1:]):
                        if (start, end) not in pairs:
                            pairs.append((start, end))
                            inferred_pairs.add((start, end))
                break
            except (ValueError, TypeError, KeyError, IndexError, AttributeError, json.JSONDecodeError) as exc:
                error = str(exc)[:300]
        else:
            raise ValueError(f"模型拓扑结构连续 3 次未通过校验：{error}")
        counts: dict[int, int] = {}
        coordinates = []
        for layer in layers:
            position = counts.get(layer, 0)
            coordinates.append((layer * 2.6, position * 1.15))
            counts[layer] = position + 1
        title = str(design.get("title") or "研究技术路线图")[:80]
    nodes = [{"id": f"N{index + 1}", "label": label, "x": x, "y": y}
             for index, (label, (x, y)) in enumerate(zip(labels, coordinates))]
    edges = [{"source": nodes[source]["id"], "target": nodes[target]["id"],
              "inferred": (source, target) in inferred_pairs}
             for source, target in pairs]
    def tex_label(value: str) -> str:
        return "".join({"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "^": r"\textasciicircum{}"}.get(char, char)
                       for char in value)

    mermaid_lines = ["flowchart LR"]
    mermaid_lines += [f'  {node["id"]}["{node["label"].replace(chr(34), "＂")}"]' for node in nodes]
    mermaid_lines += [f'  {edge["source"]} --> {edge["target"]}' for edge in edges]
    mermaid = "\n".join(mermaid_lines)

    dot_lines = ['digraph ResearchTopology {', '  graph [label="' + title + '", labelloc=t, rankdir=LR, bgcolor="white", pad=0.3];',
                 '  node [shape=box, style="rounded,filled", fillcolor="#eaf5ee", color="#267c5b", fontname="Microsoft YaHei", fontsize=11];',
                 '  edge [color="#638875", arrowsize=0.75];']
    dot_lines += [f'  {node["id"]} [label={json.dumps(node["label"], ensure_ascii=False)}];' for node in nodes]
    dot_lines += [f'  {edge["source"]} -> {edge["target"]};' for edge in edges]
    dot_lines.append('}')
    graphviz = "\n".join(dot_lines)

    tikz_lines = [r"\documentclass[tikz,border=8pt]{standalone}", r"\usepackage{ctex}",
                  r"\usetikzlibrary{arrows.meta,positioning}", r"\begin{document}",
                  r"\begin{tikzpicture}[>=Stealth, every node/.style={draw=green!45!black,fill=green!7,rounded corners,align=center,font=\small,minimum height=8mm,text width=27mm},x=1.6cm,y=1.5cm]"]
    tikz_lines += [f'\\node ({node["id"]}) at ({node["x"]:.2f},{node["y"]:.2f}) {{{tex_label(node["label"])}}};' for node in nodes]
    tikz_lines += [f'\\draw[->,thick,green!45!black] ({edge["source"]}) -- ({edge["target"]});' for edge in edges]
    tikz_lines += [r"\end{tikzpicture}", r"\end{document}"]
    tikz = "\n".join(tikz_lines)

    matplotlib = ("import matplotlib.pyplot as plt\n"
                  "from matplotlib.patches import FancyBboxPatch, FancyArrowPatch\n\n"
                  "plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']\n"
                  "plt.rcParams['axes.unicode_minus'] = False\n"
                  f"nodes = {json.dumps(nodes, ensure_ascii=False)}\n"
                  f"edges = {json.dumps(edges, ensure_ascii=False)}\n"
                  f"title = {json.dumps(title, ensure_ascii=False)}\n\n"
                  "fig, ax = plt.subplots(figsize=(max(12, max(node['x'] for node in nodes) * 1.2), max(5, max(node['y'] for node in nodes) * 1.2)))\n"
                  "positions = {node['id']: (node['x'], node['y']) for node in nodes}\n"
                  "for node in nodes:\n"
                  "    x, y = positions[node['id']]\n"
                  "    ax.add_patch(FancyBboxPatch((x - .88, y - .34), 1.76, .68, boxstyle='round,pad=0.04',\n"
                  "                                facecolor='#eaf5ee', edgecolor='#267c5b', linewidth=1.5))\n"
                  "    ax.text(x, y, node['label'], ha='center', va='center', fontsize=8.5, wrap=True)\n"
                  "for edge in edges:\n"
                  "    start, end = positions[edge['source']], positions[edge['target']]\n"
                  "    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=12,\n"
                  "                                 color='#638875', linewidth=1.4, connectionstyle='arc3,rad=0.12',\n"
                  "                                 shrinkA=34, shrinkB=34))\n"
                  "ax.set_xlim(-1.2, max(node['x'] for node in nodes) + 1.2)\n"
                  "ax.set_ylim(-.8, max(node['y'] for node in nodes) + .8)\nax.set_aspect('equal')\nax.axis('off')\n"
                  "ax.set_title(title, fontsize=13, pad=18)\nfig.tight_layout()\n"
                  "fig.savefig('research_topology.svg', bbox_inches='tight')\n"
                  "fig.savefig('research_topology.pdf', bbox_inches='tight')\n"
                  "plt.show()\n")
    return {"title": title, "mermaid": mermaid, "graphviz": graphviz, "tikz": tikz,
            "matplotlib": matplotlib, "scripts": {"mermaid": mermaid, "graphviz": graphviz,
                                                 "tikz": tikz, "matplotlib": matplotlib},
            "nodes": nodes, "edges": edges,
            "note": ("部分连线由节点层级顺序补齐，仅供绘图排版，须人工核对；" if inferred_pairs else "")
                    + "四种脚本由同一组节点和边生成，表示拟议方法流程，不代表已完成实验。"}
