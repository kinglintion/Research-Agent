from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import socket
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import ExternalPaperRecord
from .connections import mysql_engine, redis_client
from .paper_ingestion import ensure_paper_tables, ingest_external_jats, ingest_pdf


DOI_RE = re.compile(r"(?:https?://(?:dx\.)?doi\.org/)?\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.I)
ARXIV_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5}(?:v\d+)?)", re.I)
USER_AGENT = "ResearchCopilot/0.1 (academic metadata lookup; contact: local development app)"
SAFE_FULLTEXT_HOSTS = {"arxiv.org", "export.arxiv.org", "europepmc.org", "www.ebi.ac.uk", "pmc.ncbi.nlm.nih.gov"}


def _http() -> httpx.Client:
    return httpx.Client(timeout=get_settings().scholarly_http_timeout_seconds,
                        headers={"User-Agent": USER_AGENT, "Accept": "application/json, application/atom+xml, application/xml"})


def _normal(value: str | None) -> str:
    return " ".join(re.findall(r"[\w]+", (value or "").casefold()))


def _doi(value: str | None) -> str | None:
    match = DOI_RE.search(value or "")
    return match.group(1).rstrip(".,;)]}").lower() if match else None


def _is_doi_query(query: str) -> bool:
    return _doi(query) is not None


def _request_json(client: httpx.Client, url: str, params: dict | None = None) -> dict:
    response = client.get(url, params=params)
    response.raise_for_status()
    return response.json()


def _search_arxiv(query: str, limit: int) -> list[dict]:
    clean_doi = _doi(query)
    arxiv_match = ARXIV_RE.search(query)
    params: dict[str, str | int] = {"start": 0, "max_results": limit, "sortBy": "relevance", "sortOrder": "descending"}
    if arxiv_match:
        params["id_list"] = arxiv_match.group(1)
    elif clean_doi:
        params["search_query"] = f'doi:"{clean_doi}"'
    else:
        params["search_query"] = f'all:"{query.strip().replace(chr(34), "")}"'
    with _http() as client:
        response = client.get("https://export.arxiv.org/api/query", params=params)
        response.raise_for_status()
    root = ET.fromstring(response.content)
    ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    papers = []
    for entry in root.findall("a:entry", ns):
        abs_url = entry.findtext("a:id", default="", namespaces=ns)
        arxiv_id = abs_url.rsplit("/", 1)[-1]
        title = " ".join(entry.findtext("a:title", default="", namespaces=ns).split())
        authors = [item.findtext("a:name", default="", namespaces=ns) for item in entry.findall("a:author", ns)]
        doi = _doi(entry.findtext("arxiv:doi", default="", namespaces=ns))
        updated = entry.findtext("a:updated", default="", namespaces=ns)
        published = entry.findtext("a:published", default="", namespaces=ns)
        summary = " ".join(entry.findtext("a:summary", default="", namespaces=ns).split())
        categories = [c.attrib.get("term", "") for c in entry.findall("a:category", ns)]
        papers.append({
            "source": "arXiv", "source_id": arxiv_id, "title": title, "doi": doi,
            "authors": authors, "abstract": summary, "year": (published[:4] or None),
            "published": published, "updated": updated, "url": abs_url,
            "categories": categories,
            "fulltext_candidates": [{"url": f"https://arxiv.org/pdf/{arxiv_id}", "format": "pdf",
                                      "legal_basis": "arXiv公开预印本，可按arXiv使用条款获取"}],
        })
    return papers


def _crossref_work(query: str, limit: int) -> list[dict]:
    doi = _doi(query)
    with _http() as client:
        if doi:
            response = client.get(f"https://api.crossref.org/works/{quote(doi, safe='')}")
            if response.status_code == 404:
                return []
            response.raise_for_status()
            items = [response.json().get("message", {})]
        else:
            data = _request_json(client, "https://api.crossref.org/works", {
                "query.bibliographic": query, "rows": limit, "select": "DOI,title,author,abstract,published,issued,publisher,container-title,type,URL,link,license,score"
            })
            items = data.get("message", {}).get("items", [])
    output = []
    for item in items:
        titles = item.get("title") or []
        if not titles:
            continue
        authors = [" ".join(filter(None, [a.get("given"), a.get("family")])) for a in item.get("author", [])]
        date_parts = (item.get("published") or item.get("issued") or {}).get("date-parts", [[]])[0]
        licenses = item.get("license", [])
        links = item.get("link", [])
        candidates = []
        for link in links:
            content_type = (link.get("content-type") or "").lower()
            license_urls = [str(x.get("URL", "")).lower() for x in licenses]
            if ("pdf" in content_type
                    and urlparse(str(link.get("URL", ""))).hostname in SAFE_FULLTEXT_HOSTS
                    and any("creativecommons.org/licenses/" in lic or "creativecommons.org/publicdomain/" in lic for lic in license_urls)):
                candidates.append({"url": link.get("URL"), "format": "pdf", "legal_basis": "Crossref登记的Creative Commons许可",
                                   "license": licenses[0].get("URL") if licenses else None})
        output.append({
            "source": "Crossref", "source_id": item.get("DOI", ""), "title": titles[0],
            "doi": _doi(item.get("DOI")), "authors": authors, "abstract": _strip_jats(item.get("abstract", "")),
            "year": str(date_parts[0]) if date_parts else None, "publisher": item.get("publisher"),
            "container_title": (item.get("container-title") or [None])[0], "type": item.get("type"),
            "url": item.get("URL"), "score": item.get("score"),
            "licenses": licenses, "fulltext_candidates": candidates,
        })
    return output


def _strip_jats(value: str) -> str:
    if not value:
        return ""
    try:
        root = ET.fromstring(f"<root>{value}</root>")
        return " ".join(" ".join(root.itertext()).split())
    except ET.ParseError:
        return re.sub(r"<[^>]+>", " ", value).strip()


def _search_pubmed(query: str, limit: int) -> list[dict]:
    doi = _doi(query)
    term = f'"{doi}"[AID]' if doi else f'"{query.strip()}"[Title]'
    with _http() as client:
        search = _request_json(client, "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", {
            "db": "pubmed", "term": term, "retmax": limit, "retmode": "json"
        })
        ids = search.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        summaries = _request_json(client, "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi", {
            "db": "pubmed", "id": ",".join(ids), "retmode": "json"
        }).get("result", {})
        out = []
        for pmid in ids:
            item = summaries.get(pmid, {})
            if not item:
                continue
            article_ids = {str(entry.get("idtype", "")).lower(): entry.get("value") for entry in item.get("articleids", [])}
            pub_doi = _doi(article_ids.get("doi"))
            authors = [a.get("name", "") for a in item.get("authors", [])]
            out.append({"source": "PubMed", "source_id": pmid, "pmid": pmid, "pmcid": article_ids.get("pmc"),
                        "title": item.get("title", "").rstrip("."), "doi": pub_doi, "authors": authors,
                        "abstract": "", "year": (item.get("pubdate") or "")[:4] or None,
                        "container_title": item.get("fulljournalname") or item.get("source"),
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", "fulltext_candidates": []})
        # One Europe PMC request enriches PubMed IDs with OA status and licensed full-text links.
        if out:
            id_query = " OR ".join(f"EXT_ID:{item['pmid']}" for item in out)
            try:
                data = _request_json(client, "https://www.ebi.ac.uk/europepmc/webservices/rest/search", {
                    "query": f"({id_query}) AND SRC:MED", "format": "json", "resultType": "core", "pageSize": len(out)
                })
                epmc_by_pmid = {str(item.get("pmid")): item for item in data.get("resultList", {}).get("result", [])}
                for item in out:
                    epmc = epmc_by_pmid.get(str(item["pmid"]), {})
                    item["abstract"] = _strip_jats(epmc.get("abstractText", "")) or item["abstract"]
                    item["pmcid"] = epmc.get("pmcid") or item["pmcid"]
                    item["is_open_access"] = epmc.get("isOpenAccess") == "Y"
                    for fulltext in epmc.get("fullTextUrlList", {}).get("fullTextUrl", []):
                        style = (fulltext.get("documentStyle") or "").lower()
                        availability = (fulltext.get("availability") or "").lower()
                        link = fulltext.get("url") or ""
                        if (style == "pdf" and "open access" in availability and item["is_open_access"]
                                and urlparse(link).scheme == "https" and urlparse(link).hostname in SAFE_FULLTEXT_HOSTS):
                            item["fulltext_candidates"].append({"url": link, "format": "pdf",
                                "legal_basis": "Europe PMC标记为Open access的全文PDF", "license": epmc.get("license"),
                                "provider": fulltext.get("site", "Europe PMC")})
                    if item["pmcid"] and item["is_open_access"] and not item["fulltext_candidates"]:
                        item["fulltext_candidates"].append({"url": f"https://www.ebi.ac.uk/europepmc/webservices/rest/{item['pmcid']}/fullTextXML",
                            "format": "xml", "legal_basis": "Europe PMC OA subset全文XML", "license": epmc.get("license"),
                            "provider": "Europe PMC"})
            except Exception:
                for item in out:
                    item["fulltext_lookup_status"] = "unavailable"
        return out


def _search_github(query: str, limit: int) -> list[dict]:
    settings = get_settings()
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    terms = _doi(query) or query.strip()
    if len(terms) > 180:
        terms = terms[:180]
    with httpx.Client(timeout=settings.scholarly_http_timeout_seconds, headers={**headers, "User-Agent": USER_AGENT}) as client:
        response = client.get("https://api.github.com/search/repositories", params={"q": terms, "sort": "stars", "order": "desc", "per_page": min(limit, 10)})
        response.raise_for_status()
    repos = []
    for item in response.json().get("items", []):
        repos.append({"name": item.get("full_name"), "description": item.get("description"),
                      "url": item.get("html_url"), "stars": item.get("stargazers_count", 0),
                      "language": item.get("language"), "updated_at": item.get("pushed_at"),
                      "topics": item.get("topics", []), "archived": item.get("archived", False),
                      "fork": item.get("fork", False)})
    return repos


def _records_for_query(query: str, limit: int) -> tuple[list[dict], dict[str, str]]:
    providers = {
        "arxiv": lambda: _search_arxiv(query, limit),
        "pubmed": lambda: _search_pubmed(query, limit),
        "crossref": lambda: _crossref_work(query, limit),
        "github": lambda: _search_github(query, limit),
    }
    raw: dict[str, list] = {}
    status: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fn): name for name, fn in providers.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                raw[name] = future.result()
                status[name] = "ok"
            except Exception as exc:
                raw[name] = []
                status[name] = f"unavailable:{exc.__class__.__name__}"

    source_entries = raw["arxiv"] + raw["pubmed"] + raw["crossref"]
    merged: list[dict] = []
    for item in source_entries:
        doi = _doi(item.get("doi"))
        title = item.get("title", "")
        match = next((row for row in merged if doi and row.get("doi") == doi), None)
        if match is None:
            normalized = _normal(title)
            match = next((row for row in merged if normalized and _normal(row["metadata"].get("title")) == normalized
                          and (not item.get("year") or not row["metadata"].get("year") or str(item.get("year"))[:4] == str(row["metadata"].get("year"))[:4])), None)
        if match is None:
            match = {"metadata": {}, "sources": [], "fulltext_candidates": [], "repositories": []}
            merged.append(match)
        match["sources"].append(item)
        if doi:
            match["doi"] = doi
        match["fulltext_candidates"].extend(item.get("fulltext_candidates", []))
        # Prefer Crossref's bibliographic fields while filling missing values from other sources.
        priority = {"Crossref": 0, "PubMed": 1, "arXiv": 2}.get(item["source"], 9)
        current_priority = match.get("metadata_priority", 99)
        for field in ("title", "authors", "abstract", "year", "doi", "publisher", "container_title"):
            value = item.get(field)
            if value in (None, "", []):
                continue
            if priority <= current_priority or match["metadata"].get(field) in (None, "", []):
                match["metadata"][field] = value
        match["metadata_priority"] = min(priority, current_priority)
        for field in ("arxiv_id", "pmid", "pmcid"):
            if item.get(field):
                match["metadata"][field] = item[field]

    github_repos = raw["github"]
    if _is_doi_query(query) and merged:
        try:
            github_repos = _search_github(merged[0]["metadata"].get("title", ""), min(limit, 8))
            status["github"] = "ok"
        except Exception as exc:
            status["github"] = f"unavailable:{exc.__class__.__name__}"
    for row in merged:
        title_terms = set(_normal(row["metadata"].get("title")).split())
        matched_repos = []
        for repo in github_repos:
            repo_terms = set(_normal(f"{repo['name']} {repo.get('description') or ''} {' '.join(repo.get('topics', []))}").split())
            overlap = len(title_terms & repo_terms) / max(1, len(title_terms))
            if overlap >= 0.2:
                repo["title_match"] = round(overlap, 3)
                matched_repos.append(repo)
        row["repositories"] = sorted(matched_repos, key=lambda repo: (repo["title_match"], repo["stars"]), reverse=True)[:3]
    return merged, status


def _metadata_checks(sources: list[dict]) -> list[dict]:
    checks = []
    for field in ("title", "doi", "year"):
        values = []
        for source in sources:
            value = source.get(field)
            if value not in (None, "", []):
                values.append({"source": source["source"], "value": value})
        normalized = {_normal(str(item["value"])) if field != "doi" else str(item["value"]).casefold() for item in values}
        checks.append({"field": field, "status": "not_provided" if not values else "consistent" if len(normalized) == 1 else "mismatch",
                       "values": values})
    return checks


def _record_key(metadata: dict) -> str:
    identity = _doi(metadata.get("doi")) or f"{_normal(metadata.get('title'))}:{metadata.get('year') or ''}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _record_json(record: ExternalPaperRecord) -> dict:
    return {"record_id": record.id, "metadata": record.metadata_json, "sources": record.sources_json,
            "repositories": record.repositories_json, "fulltext_candidates": record.fulltext_candidates_json,
            "metadata_checks": record.metadata_checks_json, "recommendation_reason": record.recommendation_reason,
            "download_status": record.download_status, "status_detail": record.status_detail,
            "source_format": record.acquired_format, "has_pdf": record.acquired_format == "pdf",
            "paper_id": record.paper_id, "version_id": record.version_id}


def search_external_papers(query: str, limit: int = 8) -> dict:
    query = query.strip()
    cache_key = "scholarly-search:v1:" + hashlib.sha256(f"{query.casefold()}:{limit}".encode("utf-8")).hexdigest()
    try:
        cached = redis_client().get(cache_key)
        if cached:
            payload = json.loads(cached)
            payload["cached"] = True
            return payload
    except Exception:
        pass

    ensure_paper_tables()
    merged, provider_status = _records_for_query(query, limit)
    result_rows = []
    with Session(mysql_engine(), expire_on_commit=False) as session:
        for row in merged:
            metadata = row["metadata"]
            title = str(metadata.get("title") or "Untitled work")
            key = _record_key(metadata)
            sources = row["sources"]
            checks = _metadata_checks(sources)
            candidates = row["fulltext_candidates"]
            repositories = row["repositories"]
            reasons = []
            if _is_doi_query(query) and _doi(query) == _doi(metadata.get("doi")):
                reasons.append("DOI 精确匹配")
            elif _normal(query) in _normal(title):
                reasons.append("标题与检索词匹配")
            if len({source["source"] for source in sources}) > 1:
                reasons.append(f"{len({source['source'] for source in sources})} 个学术来源可交叉核对")
            if any(item.get("format") == "pdf" and item.get("legal_basis") for item in candidates):
                reasons.append("发现明确开放许可的 PDF 全文")
            if repositories:
                reasons.append(f"发现 {len(repositories)} 个 GitHub 候选仓库")
            reason = "；".join(reasons) or "来自学术索引的相关记录，建议核对题名和作者"
            status = "full_text_available" if candidates else "metadata_only"
            row["record_id"] = key
            row["metadata_checks"] = checks
            row["recommendation_reason"] = reason
            with session.begin_nested():
                stored = session.get(ExternalPaperRecord, key)
                if stored is None:
                    stored = ExternalPaperRecord(id=key, canonical_title=title, doi=_doi(metadata.get("doi")),
                        metadata_json=metadata, sources_json=sources, repositories_json=repositories,
                        fulltext_candidates_json=candidates, metadata_checks_json=checks,
                        recommendation_reason=reason,
                        download_status="not_attempted" if candidates else "metadata_only")
                    session.add(stored)
                else:
                    stored.canonical_title = title
                    stored.doi = _doi(metadata.get("doi"))
                    stored.metadata_json = metadata
                    stored.sources_json = sources
                    stored.repositories_json = repositories
                    stored.fulltext_candidates_json = candidates
                    stored.metadata_checks_json = checks
                    stored.recommendation_reason = reason
                    if stored.download_status == "not_attempted" and not candidates:
                        stored.download_status = "metadata_only"
                result = _record_json(stored)
                result["relevance_status"] = status
                result_rows.append(result)
        session.commit()
    result_rows.sort(key=lambda item: (item["relevance_status"] == "full_text_available", len(item["sources"]), len(item["repositories"])), reverse=True)
    response = {"query": query, "results": result_rows[:limit], "providers": provider_status, "cached": False}
    try:
        redis_client().setex(cache_key, get_settings().scholarly_search_cache_seconds, json.dumps(response, ensure_ascii=False, default=str))
    except Exception:
        pass
    return response


def _safe_public_https_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in SAFE_FULLTEXT_HOSTS:
        return False
    try:
        addresses = socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
        return bool(addresses) and all(ipaddress.ip_address(address[4][0]).is_global for address in addresses)
    except (OSError, ValueError):
        return False


def _download_candidate(candidate: dict, max_bytes: int) -> tuple[bytes | None, str | None]:
    if candidate.get("format") != "pdf" or not candidate.get("legal_basis"):
        return None, "该全文不是已支持的 PDF 格式或许可依据不完整"
    current_url = str(candidate.get("url") or "")
    headers = {"User-Agent": USER_AGENT, "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.1"}
    with httpx.Client(timeout=get_settings().scholarly_http_timeout_seconds, headers=headers, follow_redirects=False) as client:
        for _ in range(5):
            if not _safe_public_https_url(current_url):
                return None, "全文链接不是允许的公开学术来源地址"
            with client.stream("GET", current_url) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        return None, "全文服务返回无目标的重定向"
                    current_url = urljoin(current_url, location)
                    continue
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                data = bytearray()
                for chunk in response.iter_bytes(64 * 1024):
                    data.extend(chunk)
                    if len(data) > max_bytes:
                        return None, "PDF 超过系统上传大小限制"
                if not bytes(data).startswith(b"%PDF"):
                    return None, f"下载内容不是 PDF（{content_type or '未知内容类型'}）"
                return bytes(data), None
    return None, "全文服务重定向次数过多"


def _download_xml_candidate(candidate: dict, max_bytes: int) -> tuple[bytes | None, str | None]:
    if candidate.get("format") != "xml" or not candidate.get("legal_basis"):
        return None, "来源没有提供明确开放许可的全文XML"
    current_url = str(candidate.get("url") or "")
    with httpx.Client(timeout=get_settings().scholarly_http_timeout_seconds,
                      headers={"User-Agent": USER_AGENT, "Accept": "application/xml,text/xml"},
                      follow_redirects=False) as client:
        for _ in range(5):
            if not _safe_public_https_url(current_url):
                return None, "全文链接不是允许的公开学术来源地址"
            with client.stream("GET", current_url) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        return None, "全文服务返回无目标的重定向"
                    current_url = urljoin(current_url, location)
                    continue
                response.raise_for_status()
                data = bytearray()
                for chunk in response.iter_bytes(64 * 1024):
                    data.extend(chunk)
                    if len(data) > max_bytes:
                        return None, "全文XML超过系统上传大小限制"
                try:
                    ET.fromstring(bytes(data))
                except ET.ParseError:
                    return None, "下载内容不是可解析的XML全文"
                return bytes(data), None
    return None, "全文服务重定向次数过多"


def _invalidate_search_cache() -> None:
    try:
        client = redis_client()
        keys = list(client.scan_iter(match="scholarly-search:v1:*", count=100))
        if keys:
            client.delete(*keys)
    except Exception:
        pass


def acquire_external_paper(record_id: str) -> dict | None:
    ensure_paper_tables()
    with Session(mysql_engine(), expire_on_commit=False) as session:
        record = session.get(ExternalPaperRecord, record_id)
        if record is None:
            return None
        if record.download_status == "ingested" and record.paper_id and record.version_id:
            return {**_record_json(record), "message": "全文已入库"}
        candidates = list(record.fulltext_candidates_json or [])
        if not candidates:
            record.download_status = "metadata_only"
            record.status_detail = "未发现可由当前来源接口自动获取且许可明确的开放全文；已保存元数据，可从来源页面查看获取条件。"
            session.commit()
            _invalidate_search_cache()
            return {**_record_json(record), "message": record.status_detail}
        record.download_status = "downloading"
        record.status_detail = None
        session.commit()
        title = record.canonical_title
        metadata = dict(record.metadata_json or {})

    max_bytes = get_settings().max_pdf_bytes
    last_error = None
    for candidate in candidates:
        try:
            if candidate.get("format") == "xml":
                content, error = _download_xml_candidate(candidate, max_bytes)
            elif candidate.get("format") == "pdf":
                content, error = _download_candidate(candidate, max_bytes)
            else:
                last_error = "该全文格式暂不支持。"
                continue
            if error:
                last_error = error
                continue
            if content is None:
                continue
            if candidate.get("format") == "xml":
                outcome = ingest_external_jats(title=title, doi=_doi(metadata.get("doi")),
                    authors=metadata.get("authors", []), source_name=candidate.get("provider") or "Europe PMC",
                    source_url=candidate["url"], license_name=candidate.get("license"), content=content)
            else:
                suffix = Path(urlparse(str(candidate["url"])).path).name
                if not suffix.lower().endswith(".pdf"):
                    suffix = re.sub(r"[^\w.-]+", "_", title)[:100] + ".pdf"
                outcome = ingest_pdf(suffix, content)
            paper_id = outcome.get("paper_id")
            if outcome.get("version_id"):
                version_id = outcome["version_id"]
            else:
                version_id = None
            versions = outcome.get("versions", [])
            version_id = version_id or next((item.get("version_id") for item in versions if item.get("source_sha256") == hashlib.sha256(content).hexdigest()), None)
            if version_id is None and versions:
                version_id = versions[0].get("version_id")
            with Session(mysql_engine(), expire_on_commit=False) as session:
                stored = session.get(ExternalPaperRecord, record_id)
                stored.download_status = "ingested"
                vector_index = outcome.get("vector_index") or (outcome.get("structured") or {}).get("vector_index") or {}
                index_note = (
                    "已匹配到已有版本，未重复执行向量索引。" if outcome.get("duplicate") else
                    "Qdrant 向量索引已完成。" if vector_index.get("status") == "indexed" else
                    "Qdrant 向量索引未完成，请查看入库详情并重试索引。"
                )
                stored.status_detail = f"已从 {candidate.get('provider') or candidate.get('url')} 获取并解析开放全文，已保存至 MySQL；{index_note}"
                stored.paper_id = paper_id
                stored.version_id = version_id
                stored.acquired_format = candidate.get("format")
                session.commit()
                _invalidate_search_cache()
                return {**_record_json(stored), "ingestion": outcome, "message": stored.status_detail}
        except Exception as exc:
            last_error = f"{exc.__class__.__name__}: 获取或解析全文失败"
    status = "full_text_unavailable" if last_error and "PDF" in last_error else "download_failed"
    with Session(mysql_engine(), expire_on_commit=False) as session:
        stored = session.get(ExternalPaperRecord, record_id)
        stored.download_status = status
        stored.status_detail = last_error or "未能从提供的公开链接获取 PDF 全文。"
        session.commit()
        _invalidate_search_cache()
        return {**_record_json(stored), "message": stored.status_detail}


def get_external_paper(record_id: str) -> dict | None:
    ensure_paper_tables()
    with Session(mysql_engine(), expire_on_commit=False) as session:
        record = session.get(ExternalPaperRecord, record_id)
        return _record_json(record) if record else None
