from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Callable

from qdrant_client import models
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import PaperVersion
from .connections import mysql_engine, qdrant_client
from .model_adapters import get_embedder, get_llm_client, get_reranker


class ModelJSONOutputError(ValueError):
    def __init__(self, reason: str, finish_reason: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.finish_reason = finish_reason


_TOKEN_RE = re.compile(r"[a-zA-Z]+(?:[._'-][a-zA-Z0-9]+)*|\d+(?:[.,]\d+)*(?:e[+-]?\d+)?|[\u4e00-\u9fff]+|[^\W\s]", re.I)
_NUMBER_RE = re.compile(r"(?<![\w.])([+-]?\d+(?:,\d{3})*(?:\.\d+)?(?:e[+-]?\d+)?)", re.I)
_UNIT_RE = re.compile(
    r"^\s*(%|％|‰|×|°\s*[CFK]|(?:k|M|G|m|μ|µ|n)?"
    r"(?:km|cm|mm|μm|nm|ms|min|hr|day|year|kHz|MHz|GHz|Hz|kPa|MPa|Pa|mV|mA|kW|kJ|kg|mg|keV|MeV|eV|mol|dB|m|s|h|V|A|W|J|N|g)"
    r"|cycles|cycle|evaluations|evaluation|evals|eval|tasks|task|agents|agent"
    r"|iterations|iteration|samples|sample|pages|page|tokens|token"
    r"|倍|年|月|天|小时|分钟|秒|百分点)",
    re.I,
)


def _tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text.casefold()):
        value = match.group(0)
        if re.fullmatch(r"[\u4e00-\u9fff]+", value):
            if len(value) == 1:
                tokens.append(value)
            else:
                tokens.extend(value[index : index + 2] for index in range(len(value) - 1))
        else:
            tokens.append(value)
    return tokens


def _qdrant_filter(paper_id: str | None, version_id: str | None) -> models.Filter | None:
    conditions = []
    if paper_id:
        conditions.append(models.FieldCondition(key="paper_id", match=models.MatchValue(value=paper_id)))
    if version_id:
        conditions.append(models.FieldCondition(key="version_id", match=models.MatchValue(value=version_id)))
    return models.Filter(must=conditions) if conditions else None


def _scroll_corpus(client, collection: str, query_filter: models.Filter | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=query_filter,
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            text = str(payload.get("text") or "").strip()
            if text:
                rows.append({"id": str(point.id), "payload": payload, "text": text})
        if offset is None:
            break
    return rows


def _bm25_scores(query: str, documents: list[dict[str, Any]]) -> list[tuple[int, float]]:
    query_terms = _tokens(query)
    if not query_terms or not documents:
        return []
    term_lists = [_tokens(document["text"]) for document in documents]
    frequencies = [Counter(terms) for terms in term_lists]
    lengths = [len(terms) for terms in term_lists]
    average_length = sum(lengths) / max(len(lengths), 1)
    document_frequency = Counter()
    for counts in frequencies:
        document_frequency.update(counts.keys())

    k1, b = 1.5, 0.75
    scored = []
    for index, counts in enumerate(frequencies):
        score = 0.0
        for term in query_terms:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            df = document_frequency[term]
            inverse_frequency = math.log(1 + (len(documents) - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (1 - b + b * lengths[index] / max(average_length, 1))
            score += inverse_frequency * frequency * (k1 + 1) / denominator
        if score > 0:
            scored.append((index, score))
    return sorted(scored, key=lambda item: item[1], reverse=True)


def hybrid_search(
    query: str,
    *,
    top_k: int = 6,
    paper_id: str | None = None,
    version_id: str | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    collection = settings.qdrant_papers_collection
    client = qdrant_client()
    if not client.collection_exists(collection):
        return {"query": query, "results": [], "retrieval": {"vector": 0, "bm25": 0, "rrf": 0}}

    query_filter = _qdrant_filter(paper_id, version_id)
    candidate_k = min(max(top_k * 5, 20), 100)
    vector = get_embedder().get_query_embedding(query)
    vector_response = client.query_points(
        collection_name=collection,
        query=list(vector),
        query_filter=query_filter,
        limit=candidate_k,
        with_payload=True,
    )
    vector_hits = list(vector_response.points)
    corpus = _scroll_corpus(client, collection, query_filter)
    bm25_hits = _bm25_scores(query, corpus)[:candidate_k]

    merged: dict[str, dict[str, Any]] = {}
    for rank, point in enumerate(vector_hits, start=1):
        payload = point.payload or {}
        item_id = str(point.id)
        merged[item_id] = {
            "source_id": item_id,
            "payload": payload,
            "text": str(payload.get("text") or ""),
            "vector_score": float(point.score),
            "vector_rank": rank,
            "bm25_score": 0.0,
            "bm25_rank": None,
            "rrf_score": 1 / (60 + rank),
        }
    for rank, (index, score) in enumerate(bm25_hits, start=1):
        row = corpus[index]
        item_id = row["id"]
        item = merged.setdefault(item_id, {
            "source_id": item_id,
            "payload": row["payload"],
            "text": row["text"],
            "vector_score": None,
            "vector_rank": None,
            "bm25_score": 0.0,
            "bm25_rank": None,
            "rrf_score": 0.0,
        })
        item["bm25_score"] = float(score)
        item["bm25_rank"] = rank
        item["rrf_score"] += 1 / (60 + rank)

    # A deleted paper must never be returned if Qdrant cleanup was temporarily unavailable.
    version_ids = {str(item["payload"].get("version_id") or "") for item in merged.values()}
    with Session(mysql_engine()) as session:
        active_versions = set(session.scalars(select(PaperVersion.id).where(PaperVersion.id.in_(version_ids))))
    rrf_ranked = sorted((item for item in merged.values()
                         if str(item["payload"].get("version_id") or "") in active_versions),
                        key=lambda item: item["rrf_score"], reverse=True)[:candidate_k]
    if not rrf_ranked:
        return {"query": query, "results": [], "retrieval": {"vector": len(vector_hits), "bm25": len(bm25_hits), "rrf": 0}}
    rerank_scores = get_reranker().predict([(query, item["text"]) for item in rrf_ranked])
    reranked = sorted(
        zip(rrf_ranked, rerank_scores), key=lambda pair: float(pair[1]), reverse=True
    )[:top_k]
    results = []
    for item, score in reranked:
        payload = item["payload"]
        results.append({
            "source_id": item["source_id"],
            "paper_id": payload.get("paper_id"),
            "version_id": payload.get("version_id"),
            "title": payload.get("title"),
            "source_filename": payload.get("source_filename"),
            "page": payload.get("page"),
            "locator": payload.get("locator"),
            "text": item["text"],
            "scores": {
                "vector": item["vector_score"],
                "bm25": item["bm25_score"],
                "rrf": item["rrf_score"],
                "reranker": float(score),
                "vector_rank": item["vector_rank"],
                "bm25_rank": item["bm25_rank"],
            },
        })
    return {
        "query": query,
        "results": results,
        "retrieval": {"vector": len(vector_hits), "bm25": len(bm25_hits), "rrf": len(merged)},
    }


def _json_from_model(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Model response must be a JSON object")
    return parsed


def _llm_json(system: str, user: str, max_tokens: int = 3000) -> dict[str, Any]:
    settings = get_settings()
    deepseek = settings.llm_api_base.rstrip("/").lower() in {
        "https://api.deepseek.com", "https://api.deepseek.com/v1"
    }
    request = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if deepseek:
        # DeepSeek defaults to thinking mode. Its reasoning can consume the
        # completion budget before the JSON answer appears in message.content.
        request["response_format"] = {"type": "json_object"}
        request["extra_body"] = {"thinking": {"type": "disabled"}}

    reason = "invalid_json"
    finish_reason = None
    for attempt in range(2):
        request["max_tokens"] = max_tokens * (attempt + 1)
        response = get_llm_client().chat.completions.create(**request)
        if not response.choices:
            reason = "no_choices"
            continue
        choice = response.choices[0]
        finish_reason = choice.finish_reason
        content = choice.message.content or ""
        if finish_reason == "length":
            reason = "output_truncated"
            continue
        if not content.strip():
            reason = "empty_content"
            continue
        try:
            return _json_from_model(content)
        except (json.JSONDecodeError, ValueError):
            reason = "invalid_json"
    raise ModelJSONOutputError(reason, finish_reason)


def _normalized_quote_pattern(quote: str) -> str:
    parts = [part for part in re.split(r"\s+", quote.strip()) if part]
    # PDF extraction often wraps a hyphenated word across two lines. The
    # quote may have the same visible hyphen without the intervening newline.
    return r"\s+".join(re.escape(part).replace(r"\-", r"\-\s*") for part in parts)


def _locate_quote(quote: str, source: str) -> dict[str, Any]:
    if not quote.strip():
        return {"found": False, "start_char": None, "end_char": None}
    match = re.search(_normalized_quote_pattern(quote), source, flags=re.I)
    if not match:
        return {"found": False, "start_char": None, "end_char": None}
    return {"found": True, "start_char": match.start(), "end_char": match.end()}


def _numeric_unit_tokens(text: str) -> list[tuple[str, str]]:
    tokens = []
    aliases = {
        "cycle": "cycles", "evaluation": "evaluations", "eval": "evaluations",
        "evals": "evaluations", "task": "tasks", "agent": "agents",
        "iteration": "iterations", "sample": "samples", "page": "pages", "token": "tokens",
    }
    # PDFs can insert a space after a thousands comma: "1, 363" means 1363.
    normalized_text = re.sub(r"(?<=\d),\s+(?=\d{3}(?:\D|$))", ",", text)
    for match in _NUMBER_RE.finditer(normalized_text):
        number = match.group(1).replace(",", "")
        try:
            number = f"{float(number):.12g}"
        except ValueError:
            pass
        suffix = normalized_text[match.end():]
        unit_match = _UNIT_RE.match(suffix)
        unit = ""
        if unit_match:
            candidate = unit_match.group(1)
            remainder = suffix[unit_match.end():]
            # "20 evaluations" must not be read as the unit eV.
            if not (candidate[-1].isascii() and candidate[-1].isalpha()
                    and remainder[:1].isascii() and remainder[:1].isalpha()):
                unit = re.sub(r"\s+", "", candidate).casefold().replace("％", "%")
                unit = aliases.get(unit, unit)
        tokens.append((number, unit))
    return tokens


def _semantic_judgments(claims: list[dict[str, Any]]) -> dict[int, dict[str, str]]:
    if not claims:
        return {}
    prompt_claims = []
    for index, claim in enumerate(claims):
        prompt_claims.append({
            "claim_index": index,
            "claim": claim["text"],
            "evidence": [
                {"source_id": item["source_id"], "page": item["page"], "quote": item["quote"]}
                for item in claim["_valid_quotes"]
            ],
        })
    result = _llm_json(
        "You are a strict evidence entailment verifier. Judge only the exact supplied quotes. "
        "Do not use outside knowledge. Return JSON: {\"judgments\":[{\"claim_index\":0,"
        "\"verdict\":\"supported|partial|unsupported|uncertain\",\"reason\":\"brief reason\"}]}.",
        json.dumps(prompt_claims, ensure_ascii=False),
        max_tokens=1800,
    )
    verdicts: dict[int, dict[str, str]] = {}
    allowed = {"supported", "partial", "unsupported", "uncertain"}
    for item in result.get("judgments", []):
        try:
            index = int(item["claim_index"])
        except (KeyError, TypeError, ValueError):
            continue
        verdict = str(item.get("verdict", "uncertain")).lower()
        verdicts[index] = {
            "verdict": verdict if verdict in allowed else "uncertain",
            "reason": str(item.get("reason", ""))[:500],
        }
    return verdicts


def answer_with_evidence(
    question: str,
    *,
    top_k: int = 6,
    paper_id: str | None = None,
    version_id: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if progress:
        progress("混合检索与重排")
    search = hybrid_search(question, top_k=top_k, paper_id=paper_id, version_id=version_id)
    evidence = search["results"]
    if not evidence:
        return {
            "question": question,
            "claims": [],
            "answer_status": "insufficient_evidence",
            "retrieval": search["retrieval"],
        }

    evidence_for_prompt = [
        {"source_id": item["source_id"], "title": item["title"], "page": item["page"], "text": item["text"]}
        for item in evidence
    ]
    if progress:
        progress(f"找到 {len(evidence)} 条候选证据，生成可核查陈述")
    generated = _llm_json(
        "You answer academic questions with evidence only. Break the answer into short atomic factual "
        "claims. Every claim must cite one or more supplied source_id values and include a verbatim "
        "contiguous quote copied from that source's text. Do not invent sources, pages, quotations, "
        "numbers, or units. Every number and unit in a claim must also appear in its cited quotes. "
        "State one independently checkable fact per claim. If evidence is insufficient, return no claim. "
        "Return only JSON shaped as "
        "{\"claims\":[{\"text\":\"...\",\"citations\":[{\"source_id\":\"...\",\"quote\":\"...\"}]}]}.",
        json.dumps({"question": question, "evidence": evidence_for_prompt}, ensure_ascii=False),
        max_tokens=3000,
    )
    raw_claims = generated.get("claims", [])
    if not isinstance(raw_claims, list):
        raise ValueError("Model response field 'claims' must be a list")
    evidence_by_id = {item["source_id"]: item for item in evidence}
    checked: list[dict[str, Any]] = []
    for raw in raw_claims:
        if not isinstance(raw, dict) or not str(raw.get("text", "")).strip():
            continue
        citations = raw.get("citations", []) if isinstance(raw.get("citations", []), list) else []
        checked_citations = []
        valid_quotes = []
        citation_exists = bool(citations)
        for citation in citations:
            if not isinstance(citation, dict):
                citation_exists = False
                continue
            source_id = str(citation.get("source_id", ""))
            quote = str(citation.get("quote", "")).strip()
            source = evidence_by_id.get(source_id)
            if source is None:
                citation_exists = False
                checked_citations.append({"source_id": source_id, "exists": False, "quote": quote})
                continue
            location = _locate_quote(quote, source["text"])
            checked_citations.append({
                "source_id": source_id,
                "exists": True,
                "paper_id": source.get("paper_id"),
                "version_id": source.get("version_id"),
                "title": source["title"],
                "source_filename": source["source_filename"],
                "page": source["page"],
                "locator": source.get("locator"),
                "quote": quote,
                "quote_location": location,
            })
            if location["found"]:
                valid_quotes.append({**checked_citations[-1], "text": source["text"]})

        number_tokens = _numeric_unit_tokens(str(raw["text"]))
        cited_tokens = [token for item in valid_quotes for token in _numeric_unit_tokens(item["quote"])]
        missing_numbers = list((Counter(number_tokens) - Counter(cited_tokens)).elements())
        numeric_check = not missing_numbers
        claim = {
            "text": str(raw["text"]).strip(),
            "citations": checked_citations,
            "checks": {
                "citation_exists": citation_exists,
                "source_location_available": bool(checked_citations) and all(
                    citation.get("paper_id") and citation.get("version_id") and isinstance(citation.get("page"), int)
                    and citation["page"] > 0
                    and (paper_id is None or citation["paper_id"] == paper_id)
                    and (version_id is None or citation["version_id"] == version_id)
                    for citation in checked_citations if citation.get("exists")
                ),
                "quote_located": bool(valid_quotes) and all(
                    citation.get("quote_location", {}).get("found", False)
                    for citation in checked_citations if citation.get("exists")
                ),
                "numeric_and_units_match": numeric_check,
                "numbers_checked": [{"number": number, "unit": unit} for number, unit in number_tokens],
                "numbers_missing_from_quotes": [
                    {"number": number, "unit": unit} for number, unit in missing_numbers
                ],
            },
            "_valid_quotes": valid_quotes,
        }
        checked.append(claim)

    if progress:
        progress("定位引文并核对数字、单位和语义支持")
    semantic_available = True
    try:
        semantic = _semantic_judgments(checked)
    except Exception:
        semantic = {}
        semantic_available = False

    for index, claim in enumerate(checked):
        claim.pop("_valid_quotes", None)
        judgment = semantic.get(index, {"verdict": "unavailable", "reason": "语义支持复核未能完成"})
        claim["checks"]["semantic_support"] = judgment
        claim["verified"] = bool(
            claim["checks"]["citation_exists"]
            and claim["checks"]["source_location_available"]
            and claim["checks"]["quote_located"]
            and claim["checks"]["numeric_and_units_match"]
            and judgment["verdict"] == "supported"
        )

    return {
        "question": question,
        "claims": checked,
        "answer_status": (
            "verified" if checked and all(item["verified"] for item in checked)
            else "needs_review" if checked else "insufficient_evidence"
        ),
        "semantic_verifier_available": semantic_available,
        "retrieval": search["retrieval"],
        "retrieval_method": ["qdrant_dense", "bm25", "rrf", "bge_reranker"],
    }
