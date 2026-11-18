from __future__ import annotations

import difflib
import hashlib
import re
import shutil
import uuid
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Callable

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from ..config import PROJECT_ROOT, get_settings
from ..models import Base, ExternalPaperRecord, Paper, PaperFingerprintBand, PaperVersion, ParagraphTranslation, ReadingPreference
from .connections import mysql_engine, qdrant_client
from .formula_ocr import extract_pdf_formulas
from .pdf_parsing import ParsedPdf, parse_pdf, _pymupdf_title
from .vector_ingestion import index_paper_content
from qdrant_client import models as qdrant_models


ARXIV_RE = re.compile(r"(?<!\d)(\d{4}\.\d{4,5})(?:v(\d+))?(?!\d)", re.IGNORECASE)
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.IGNORECASE)
WORD_RE = re.compile(r"[\u3400-\u9fff]+|[a-z0-9]+", re.IGNORECASE)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
MAX_SHINGLES = 25000
MINHASH_SIZE = 64
BANDS = 8
ROWS_PER_BAND = MINHASH_SIZE // BANDS
MERSENNE_PRIME = (1 << 61) - 1


@lru_cache
def ensure_paper_tables() -> None:
    Base.metadata.create_all(mysql_engine())


def _stable_identifiers(filename: str, text: str) -> tuple[str | None, str | None, str | None]:
    arxiv_match = ARXIV_RE.search(Path(filename).name) or ARXIV_RE.search(text[:10000])
    arxiv_id = arxiv_match.group(1).lower() if arxiv_match else None
    version_label = f"v{arxiv_match.group(2)}" if arxiv_match and arxiv_match.group(2) else None
    doi_match = DOI_RE.search(text[:50000])
    doi = doi_match.group(1).rstrip(".,;)]}").lower() if doi_match else None
    return arxiv_id, doi, version_label


def _version_number(label: str | None) -> int | None:
    match = re.fullmatch(r"v(\d+)", label or "", re.IGNORECASE)
    return int(match.group(1)) if match else None


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in WORD_RE.findall(text.casefold()):
        if re.fullmatch(r"[\u3400-\u9fff]+", match):
            if len(match) < 3:
                tokens.append(match)
            else:
                tokens.extend(match[i : i + 3] for i in range(len(match) - 2))
        else:
            tokens.append(match)
        if len(tokens) >= MAX_SHINGLES:
            break
    return tokens


def _shingles(text: str) -> set[str]:
    tokens = _tokenize(text[:250000])
    if len(tokens) < 3:
        return set(tokens)
    return {" ".join(tokens[i : i + 3]) for i in range(len(tokens) - 2)}


@lru_cache(maxsize=MINHASH_SIZE)
def _minhash_coefficients(seed: int) -> tuple[int, int]:
    material = f"research-copilot-minhash-{seed}".encode("ascii")
    digest = hashlib.blake2b(material, digest_size=16).digest()
    a = int.from_bytes(digest[:8], "big") % MERSENNE_PRIME
    b = int.from_bytes(digest[8:], "big") % MERSENNE_PRIME
    return (a or 1) | 1, b


def _minhash_bands(text: str) -> tuple[list[str], set[str]]:
    shingles = _shingles(text)
    if not shingles:
        return [], shingles
    hashed = [
        int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big")
        % MERSENNE_PRIME
        for shingle in shingles
    ]
    signature = []
    for seed in range(MINHASH_SIZE):
        a, b = _minhash_coefficients(seed)
        signature.append(min((a * value + b) % MERSENNE_PRIME for value in hashed))

    keys = []
    for band in range(BANDS):
        start = band * ROWS_PER_BAND
        payload = ",".join(str(value) for value in signature[start : start + ROWS_PER_BAND])
        digest = hashlib.sha256(payload.encode("ascii")).hexdigest()
        keys.append(f"{band}:{digest}")
    return keys, shingles


def _title_similarity(left: str, right: str) -> float:
    normalize = lambda value: " ".join(WORD_RE.findall(value.casefold()))
    a, b = normalize(left), normalize(right)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def _section_map(markdown: str) -> dict[str, str]:
    headings = list(HEADING_RE.finditer(markdown))
    if not headings:
        return {"全文": markdown}
    sections: dict[str, str] = {}
    for index, match in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
        sections[match.group(2).strip()] = markdown[match.end() : end].strip()
    return sections


def _compare_versions(previous: str, current: str, previous_label: str | None) -> tuple[str, list[dict]]:
    old_sections, new_sections = _section_map(previous), _section_map(current)
    changes = []
    added = removed = modified = 0
    for heading in sorted(set(old_sections) | set(new_sections)):
        if heading not in old_sections:
            added += 1
            changes.append({"section": heading, "change": "added"})
        elif heading not in new_sections:
            removed += 1
            changes.append({"section": heading, "change": "removed"})
        elif old_sections[heading] != new_sections[heading]:
            ratio = difflib.SequenceMatcher(
                None, old_sections[heading], new_sections[heading], autojunk=False
            ).ratio()
            modified += 1
            changes.append({"section": heading, "change": "modified", "similarity": round(ratio, 4)})
    label = previous_label or "上一版本"
    summary = f"相对{label}：新增 {added} 个章节，删除 {removed} 个章节，内容变化 {modified} 个章节。"
    return summary, changes


def _paper_json(paper: Paper, versions: list[PaperVersion]) -> dict:
    numbered = [item for item in versions if _version_number(item.version_label) is not None]
    recommended = max(
        numbered,
        key=lambda item: (_version_number(item.version_label) or 0, item.created_at),
        default=None,
    ) or max(versions, key=lambda item: item.created_at, default=None)
    return {
        "paper_id": paper.id,
        "arxiv_id": paper.arxiv_id,
        "doi": paper.doi,
        "canonical_title": paper.canonical_title,
        "recommended_version_id": recommended.id if recommended else None,
        "recommended_version_reason": (
            f"建议优先保留 {recommended.version_label}（版本号最高；旧版本仍保留用于对比）"
            if recommended and _version_number(recommended.version_label) is not None
            else "建议优先保留最近入库版本；其他版本仍保留用于对比" if recommended else None
        ),
        "versions": [
            {
                "version_id": version.id,
                "version_label": version.version_label,
                "source_filename": version.source_filename,
                "source_sha256": version.source_sha256,
                "parser": version.parser_name,
                "source_format": (version.structured_json or {}).get("source_format", "pdf"),
                "has_pdf": version.stored_pdf_path.lower().endswith(".pdf"),
                "needs_ocr": version.needs_ocr,
                "change_summary": version.change_summary,
                "created_at": version.created_at,
            }
            for version in versions
        ],
    }


def preview_pdf_duplicates(filename: str, content: bytes) -> dict:
    """Compare actual PDF text with stored versions before any ingestion job is queued."""
    if not content.startswith(b"%PDF"):
        raise ValueError("The uploaded file is not a valid PDF")
    if len(content) > get_settings().max_pdf_bytes:
        raise ValueError("PDF exceeds the upload limit")
    import pymupdf

    source_sha = hashlib.sha256(content).hexdigest()
    with pymupdf.open(stream=content, filetype="pdf") as pdf:
        text = "\n".join(page.get_text() for page in pdf)[:150000]
    title = _pymupdf_title(content) or ""
    incoming_shingles = _shingles(text)
    _, _, incoming_version = _stable_identifiers(filename, text)
    ensure_paper_tables()
    matches = []
    semantic_inputs: dict[str, str] = {}
    with Session(mysql_engine()) as session:
        exact = session.scalar(select(PaperVersion).where(PaperVersion.source_sha256 == source_sha))
        if exact:
            paper = session.get(Paper, exact.paper_id)
            matches.append({"paper_id": paper.id, "version_id": exact.id,
                            "title": paper.canonical_title, "version_label": exact.version_label,
                            "kind": "exact_file", "content_similarity": 1.0,
                            "containment": 1.0, "title_similarity": 1.0,
                            "semantic_similarity": 1.0})
        if incoming_shingles:
            versions = session.scalars(select(PaperVersion).order_by(PaperVersion.created_at.desc()).limit(500))
            for version in versions:
                if exact is not None and version.id == exact.id:
                    continue
                paper = session.get(Paper, version.paper_id)
                if not paper or not version.markdown:
                    continue
                stored_shingles = _shingles(version.markdown[:150000])
                if not stored_shingles:
                    continue
                shared = len(incoming_shingles & stored_shingles)
                union = len(incoming_shingles | stored_shingles)
                similarity = shared / union if union else 0.0
                containment = shared / min(len(incoming_shingles), len(stored_shingles))
                title_similarity = _title_similarity(title, version.title or paper.canonical_title)
                lexical_match = similarity >= 0.42 or containment >= 0.68
                semantic_candidate = title_similarity >= 0.78 and similarity >= 0.1
                if not lexical_match and not semantic_candidate:
                    continue
                semantic_inputs[version.id] = version.markdown[:5000]
                matches.append({"paper_id": paper.id, "version_id": version.id,
                                "title": paper.canonical_title, "version_label": version.version_label,
                                "kind": "content_overlap" if lexical_match else "semantic_candidate",
                                "content_similarity": round(similarity, 4),
                                "containment": round(containment, 4),
                                "title_similarity": round(title_similarity, 4),
                                "semantic_similarity": None})
    semantic_status = "not_needed" if not semantic_inputs else "checked"
    if semantic_inputs:
        try:
            from .model_adapters import get_embedder
            embedder = get_embedder()
            query_vector = embedder.get_text_embedding(text[:5000])
            query_norm = sum(value * value for value in query_vector) ** 0.5
            for match in sorted(matches, key=lambda row: row["content_similarity"], reverse=True)[:12]:
                candidate_vector = embedder.get_text_embedding(semantic_inputs[match["version_id"]])
                candidate_norm = sum(value * value for value in candidate_vector) ** 0.5
                score = (sum(a * b for a, b in zip(query_vector, candidate_vector))
                         / (query_norm * candidate_norm)) if query_norm and candidate_norm else 0.0
                match["semantic_similarity"] = round(score, 4)
        except Exception as exc:
            semantic_status = f"unavailable:{exc.__class__.__name__}"
    matches = [row for row in matches if row["kind"] != "semantic_candidate"
               or (row["semantic_similarity"] is not None and row["semantic_similarity"] >= 0.86)]
    for row in matches:
        if row["kind"] == "semantic_candidate":
            row["kind"] = "content_overlap"
    matches.sort(key=lambda row: (row["content_similarity"], row["containment"]), reverse=True)
    return {"filename": Path(filename).name, "incoming_version": incoming_version,
            "text_characters": len(text), "needs_ocr": not bool(text.strip()),
            "matches": matches[:8], "has_warning": bool(matches) or not bool(text.strip()),
            "method": "PDF 正文片段重合度 + BGE-M3 语义相似度；文件哈希仅用于判定完全相同文件",
            "semantic_status": semantic_status,
            "recommendation": (
                "文件与现有版本完全相同，建议保留现有版本。" if exact else
                "正文与已有论文高度重合；不同版本建议并存以保留演进记录，覆盖前请核对目标版本。"
                if matches else "扫描件无法在入库前提取正文；需要先运行 OCR，暂不能排除重复。"
                if not text.strip() else "未发现高重合的现有论文。"
            )}


def preview_pdf_batch(files: list[tuple[str, bytes]]) -> list[dict]:
    """Warn about both stored papers and overlapping files in the same upload batch."""
    if not files or len(files) > 20:
        raise ValueError("一次最多可检查 20 个 PDF")
    previews = [preview_pdf_duplicates(filename, content) for filename, content in files]
    import pymupdf
    shingles = []
    for _, content in files:
        with pymupdf.open(stream=content, filetype="pdf") as pdf:
            shingles.append(_shingles("\n".join(page.get_text() for page in pdf)[:150000]))
    for index, first in enumerate(shingles):
        batch_matches = []
        if first:
            for other_index, second in enumerate(shingles):
                if index == other_index or not second:
                    continue
                shared = len(first & second)
                union = len(first | second)
                similarity = shared / union if union else 0.0
                containment = shared / min(len(first), len(second))
                if similarity >= 0.42 or containment >= 0.68:
                    batch_matches.append({"filename": Path(files[other_index][0]).name,
                                          "content_similarity": round(similarity, 4),
                                          "containment": round(containment, 4),
                                          "version_label": previews[other_index]["incoming_version"]})
        previews[index]["batch_matches"] = batch_matches
        if batch_matches:
            previews[index]["has_warning"] = True
            if not previews[index]["matches"]:
                previews[index]["recommendation"] = (
                    "本批次中有正文高度重合的 PDF；建议保留不同版本用于对比，或跳过不需要的版本。"
                )
    return previews


def ingest_pdf(filename: str, content: bytes, *, progress: Callable[[str, int], None] | None = None,
               duplicate_action: str = "append", replace_version_id: str | None = None) -> dict:
    settings = get_settings()
    if duplicate_action not in {"append", "replace"}:
        raise ValueError("duplicate_action must be append or replace")
    if duplicate_action == "replace" and not replace_version_id:
        raise ValueError("replace_version_id is required when replacing a version")
    safe_name = Path(filename.replace("\\", "/")).name
    if not safe_name.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are accepted")
    if not content.startswith(b"%PDF"):
        raise ValueError("The uploaded file is not a valid PDF")
    if len(content) > settings.max_pdf_bytes:
        raise ValueError(f"PDF exceeds the {settings.max_pdf_bytes // (1024 * 1024)} MB upload limit")

    source_sha = hashlib.sha256(content).hexdigest()
    ensure_paper_tables()
    with Session(mysql_engine(), expire_on_commit=False) as session:
        exact = session.scalar(select(PaperVersion).where(PaperVersion.source_sha256 == source_sha))
        if exact:
            paper = session.get(Paper, exact.paper_id)
            versions = list(session.scalars(
                select(PaperVersion).where(PaperVersion.paper_id == exact.paper_id)
                .order_by(PaperVersion.created_at.desc())
            ))
            return {"duplicate": True, "duplicate_kind": "exact_file", "version_id": exact.id,
                    **_paper_json(paper, versions)}
    parsed, formula_extraction = _run_pdf_pipeline(safe_name, content, source_sha, progress=progress)
    if progress:
        progress("核对版本与语义重复", 63)
    arxiv_id, doi, parsed_version = _stable_identifiers(safe_name, parsed.full_text)
    title = parsed.title.strip() or Path(safe_name).stem
    abstract = parsed.abstract.strip()
    paper_text = "\n".join([title, abstract, parsed.full_text[:150000]])
    band_keys, shingles = _minhash_bands(paper_text)
    structured = {
        "metadata": {"title": title, "authors": parsed.authors or [], "abstract": abstract,
                     "arxiv_id": arxiv_id, "doi": doi, "version_label": parsed_version},
        "pages": parsed.pages,
        "sections": parsed.sections,
        "references": parsed.references or [],
        "parser": parsed.parser_name,
        "needs_ocr": parsed.needs_ocr,
        "warnings": parsed.warnings or [],
        "formula_extraction": formula_extraction,
        "figures": parsed.figures or [],
        "layout_pipeline": {
            "layout_and_text": parsed.parser_name,
            "figure_regions": "extracted" if parsed.figures else "not_available_from_selected_parser",
            "table_regions": (
                "layout_markdown"
                if parsed.parser_name in {"marker", "paddleocr"}
                else "not_available_from_selected_parser"
            ),
            "equation_regions": {
                "status": formula_extraction.get("status", "unknown"),
                "model": formula_extraction.get("model"),
                "count": formula_extraction.get("count", 0),
            },
            "inline_scripts": "pymupdf_geometry_when_available",
            "figure_analysis": (
                "completed" if any(
                    (figure.get("analysis") or {}).get("status") == "completed"
                    for figure in (parsed.figures or [])
                ) else "pending_multimodal_configuration"
            ),
        },
    }
    ensure_paper_tables()
    storage_dir = _paper_storage_dir()
    storage_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = storage_dir / f"{source_sha}.pdf"
    if not pdf_path.exists():
        pdf_path.write_bytes(content)

    if duplicate_action == "replace" and replace_version_id:
        return _replace_pdf_version(
            replace_version_id, safe_name, source_sha, pdf_path, parsed, structured,
            band_keys, shingles, arxiv_id, doi, parsed_version, progress,
        )

    with Session(mysql_engine(), expire_on_commit=False) as session:
        exact = session.scalar(select(PaperVersion).where(PaperVersion.source_sha256 == source_sha))
        if exact:
            paper = session.get(Paper, exact.paper_id)
            versions = list(session.scalars(
                select(PaperVersion).where(PaperVersion.paper_id == exact.paper_id)
                .order_by(PaperVersion.created_at.desc())
            ))
            return {"duplicate": True, "duplicate_kind": "exact_file", "version_id": exact.id,
                    **_paper_json(paper, versions)}

        paper = None
        stable_identifier = None
        if arxiv_id:
            paper = session.scalar(select(Paper).where(Paper.arxiv_id == arxiv_id))
            if paper is not None:
                stable_identifier = f"arxiv:{arxiv_id}"

        if paper is None and doi:
            paper = session.scalar(select(Paper).where(Paper.doi == doi))
            if paper is not None:
                stable_identifier = f"doi:{doi}"

        previous = None
        existing_versions: list[PaperVersion] = []
        successor_to_relink = None
        same_label_version = None
        duplicate_candidates = []
        if paper is not None:
            existing_versions = list(session.scalars(
                select(PaperVersion).where(PaperVersion.paper_id == paper.id)
            ))
            previous = max(existing_versions, key=lambda item: item.created_at) if existing_versions else None
            if parsed_version:
                same_label_version = session.scalar(
                    select(PaperVersion).where(
                        PaperVersion.paper_id == paper.id,
                        PaperVersion.version_label == parsed_version,
                    ).limit(1)
                )
                incoming_number = _version_number(parsed_version)
                if incoming_number is not None:
                    numbered_versions = [
                        item for item in existing_versions
                        if _version_number(item.version_label) is not None
                    ]
                    lower_versions = [
                        item for item in numbered_versions
                        if _version_number(item.version_label) < incoming_number
                    ]
                    higher_versions = [
                        item for item in numbered_versions
                        if _version_number(item.version_label) > incoming_number
                    ]
                    previous = max(
                        lower_versions,
                        key=lambda item: _version_number(item.version_label) or 0,
                        default=None,
                    )
                    successor_to_relink = min(
                        higher_versions,
                        key=lambda item: _version_number(item.version_label) or 0,
                        default=None,
                    )
        elif band_keys:
            matched = session.execute(
                select(PaperFingerprintBand.version_id)
                .where(PaperFingerprintBand.bucket_key.in_(band_keys))
                .group_by(PaperFingerprintBand.version_id)
                .having(func.count(PaperFingerprintBand.bucket_key) >= 2)
                .limit(50)
            ).all()
            candidate_ids = [row[0] for row in matched]
            candidates = list(session.scalars(
                select(PaperVersion).where(PaperVersion.id.in_(candidate_ids))
            )) if candidate_ids else []
            query_summary = "\n".join([title, abstract, parsed.full_text[:5000]])
            query_vector = None
            semantic_check_error = None
            if candidates:
                try:
                    from .model_adapters import get_embedder

                    query_vector = get_embedder().get_text_embedding(query_summary)
                except Exception as exc:
                    semantic_check_error = exc.__class__.__name__
            for candidate in candidates:
                candidate_paper = session.get(Paper, candidate.paper_id)
                candidate_shingles = _shingles(
                    "\n".join([candidate.title, candidate.abstract, candidate.markdown[:150000]])
                )
                union_size = len(shingles | candidate_shingles)
                jaccard = len(shingles & candidate_shingles) / union_size if union_size else 0.0
                title_score = _title_similarity(title, candidate.title)
                semantic_score = None
                if query_vector is not None:
                    try:
                        from .model_adapters import get_embedder

                        candidate_vector = get_embedder().get_text_embedding(
                            "\n".join([candidate.title, candidate.abstract, candidate.markdown[:5000]])
                        )
                        dot = sum(a * b for a, b in zip(query_vector, candidate_vector))
                        query_norm = sum(value * value for value in query_vector) ** 0.5
                        candidate_norm = sum(value * value for value in candidate_vector) ** 0.5
                        if query_norm and candidate_norm:
                            semantic_score = dot / (query_norm * candidate_norm)
                    except Exception as exc:
                        semantic_check_error = exc.__class__.__name__
                is_candidate = (
                    title_score >= 0.8
                    and (
                        (semantic_score is not None and semantic_score >= 0.82)
                        or (semantic_score is None and jaccard >= 0.75)
                    )
                )
                if is_candidate:
                    duplicate_candidates.append({
                        "paper_id": candidate_paper.id,
                        "version_id": candidate.id,
                        "title": candidate_paper.canonical_title,
                        "jaccard": round(jaccard, 4),
                        "title_similarity": round(title_score, 4),
                        "semantic_similarity": round(semantic_score, 4) if semantic_score is not None else None,
                        "verification": "bge_m3" if semantic_score is not None else "lexical_fallback",
                        "action": "review_before_linking",
                    })
            if semantic_check_error:
                structured["warnings"].append(
                    f"BGE semantic duplicate verification was unavailable ({semantic_check_error}); lexical LSH was used."
                )
            duplicate_candidates.sort(
                key=lambda item: (item["jaccard"] + item["title_similarity"]) / 2,
                reverse=True,
            )
            duplicate_candidates = duplicate_candidates[:5]

        version_id = str(uuid.uuid4())
        paper_id = paper.id if paper else str(uuid.uuid4())
        version_label = parsed_version
        change_summary = None
        version_changes = []
        version_parent_id = previous.id if previous else None
        version_warning = None
        if same_label_version is not None:
            same_shingles = _shingles(
                "\n".join([same_label_version.title, same_label_version.abstract,
                           same_label_version.markdown[:150000]])
            )
            union_size = len(shingles | same_shingles)
            same_similarity = len(shingles & same_shingles) / union_size if union_size else 0.0
            if same_similarity >= 0.85:
                existing_paper = session.get(Paper, same_label_version.paper_id)
                existing_versions = list(session.scalars(
                    select(PaperVersion).where(PaperVersion.paper_id == same_label_version.paper_id)
                    .order_by(PaperVersion.created_at.desc())
                ))
                return {
                    "duplicate": True,
                    "duplicate_kind": "same_identifier_version",
                    "version_id": same_label_version.id,
                    "similarity": round(same_similarity, 4),
                    **_paper_json(existing_paper, existing_versions),
                }
            version_label = f"{parsed_version}-alt-{source_sha[:8]}"
            version_parent_id = None
            successor_to_relink = None
            version_warning = (
                f"This file is labeled {parsed_version}, which already exists, but extracted text differs; "
                "stored as an alternate for manual review."
            )
        elif previous:
            if not version_label:
                version_label = f"revision-{session.scalar(select(func.count(PaperVersion.id)).where(PaperVersion.paper_id == paper.id)) + 1}"
            change_summary, version_changes = _compare_versions(
                previous.markdown, parsed.markdown, previous.version_label
            )
            structured["version_diff"] = version_changes
        elif not version_label:
            version_label = "v1"

        if paper is None:
            paper = Paper(
                id=paper_id,
                arxiv_id=arxiv_id,
                doi=doi,
                canonical_title=title,
            )
            session.add(paper)
            session.flush()
        elif not paper.canonical_title and title:
            paper.canonical_title = title

        if version_warning:
            structured["warnings"].append(version_warning)
        version = PaperVersion(
            id=version_id,
            paper_id=paper.id,
            parent_version_id=version_parent_id,
            version_label=version_label,
            source_filename=safe_name[:512],
            source_sha256=source_sha,
            stored_pdf_path=str(pdf_path),
            title=title,
            authors=parsed.authors or [],
            abstract=abstract,
            parser_name=parsed.parser_name,
            needs_ocr=parsed.needs_ocr,
            structured_json=structured,
            markdown=parsed.markdown,
            change_summary=change_summary,
        )
        session.add(version)
        session.flush()
        if successor_to_relink is not None:
            successor_to_relink.parent_version_id = version.id
            successor_to_relink.change_summary, _ = _compare_versions(
                parsed.markdown,
                successor_to_relink.markdown,
                version_label,
            )
        session.add_all([
            PaperFingerprintBand(version_id=version.id, bucket_key=key) for key in band_keys
        ])
        if progress:
            progress("保存文献与版本", 78)
        session.commit()
        try:
            if progress:
                progress("建立 Qdrant 检索索引", 88)
            vector_index = index_paper_content(
                paper_id=paper.id,
                version_id=version.id,
                title=title,
                markdown=parsed.markdown,
                source_filename=safe_name,
                parser_name=parsed.parser_name,
                pages=parsed.pages,
                replace_existing=False,
            )
        except Exception as exc:
            vector_index = {"status": "failed", "error_type": exc.__class__.__name__,
                            "error_detail": str(exc)[:240]}
            structured["warnings"].append(
                f"Qdrant indexing failed ({exc.__class__.__name__}); retry via the version index endpoint."
            )
        structured["vector_index"] = vector_index
        version.structured_json = dict(structured)
        session.commit()
        versions = list(session.scalars(
            select(PaperVersion).where(PaperVersion.paper_id == paper.id)
            .order_by(PaperVersion.created_at.desc())
        ))
        return {
            "duplicate": False,
            "linked_by": stable_identifier,
            "version_linked": bool(version_parent_id),
            "duplicate_candidates": duplicate_candidates,
            "parser": parsed.parser_name,
            "needs_ocr": parsed.needs_ocr,
            "warnings": structured["warnings"],
            "vector_index": vector_index,
            "structured": structured,
            "markdown": parsed.markdown,
            "change_summary": change_summary,
            **_paper_json(paper, versions),
        }


def _replace_pdf_version(
    version_id: str, filename: str, source_sha: str, pdf_path: Path, parsed: ParsedPdf,
    structured: dict, band_keys: list[str], incoming_shingles: set[str],
    arxiv_id: str | None, doi: str | None, parsed_version: str | None,
    progress: Callable[[str, int], None] | None,
) -> dict:
    old_path = None
    old_sha = None
    with Session(mysql_engine(), expire_on_commit=False) as session:
        version = session.get(PaperVersion, version_id)
        if version is None:
            raise ValueError("The version selected for replacement no longer exists")
        paper = session.get(Paper, version.paper_id)
        if arxiv_id and paper.arxiv_id and arxiv_id != paper.arxiv_id:
            raise ValueError("The incoming arXiv ID differs from the selected paper")
        if doi and paper.doi and doi != paper.doi:
            raise ValueError("The incoming DOI differs from the selected paper")
        old_shingles = _shingles(version.markdown[:150000])
        shared = len(incoming_shingles & old_shingles)
        union = len(incoming_shingles | old_shingles)
        overlap = shared / union if union else 0.0
        containment = shared / min(len(incoming_shingles), len(old_shingles)) if incoming_shingles and old_shingles else 0.0
        if overlap < 0.42 and containment < 0.68:
            raise ValueError("The selected version is not sufficiently similar in PDF text; replacement was cancelled")
        new_label = parsed_version or version.version_label
        conflict = session.scalar(select(PaperVersion.id).where(
            PaperVersion.paper_id == paper.id, PaperVersion.version_label == new_label,
            PaperVersion.id != version.id))
        if conflict:
            raise ValueError("Another stored version already uses this version label; choose that version or append")
        old_path = Path(version.stored_pdf_path) if version.stored_pdf_path else None
        old_sha = version.source_sha256
        parent = session.get(PaperVersion, version.parent_version_id) if version.parent_version_id else None
        change_summary, changes = _compare_versions(parent.markdown, parsed.markdown, parent.version_label) if parent else (None, [])
        structured["version_diff"] = changes
        version.source_filename = filename[:512]
        version.source_sha256 = source_sha
        version.stored_pdf_path = str(pdf_path)
        version.version_label = new_label
        version.title = parsed.title
        version.authors = parsed.authors or []
        version.abstract = parsed.abstract
        version.parser_name = parsed.parser_name
        version.needs_ocr = parsed.needs_ocr
        version.structured_json = structured
        version.markdown = parsed.markdown
        version.change_summary = change_summary
        if parsed.title:
            paper.canonical_title = parsed.title
        if not paper.arxiv_id:
            paper.arxiv_id = arxiv_id
        if not paper.doi:
            paper.doi = doi
        session.execute(delete(PaperFingerprintBand).where(PaperFingerprintBand.version_id == version.id))
        session.execute(delete(ParagraphTranslation).where(ParagraphTranslation.version_id == version.id))
        session.add_all([PaperFingerprintBand(version_id=version.id, bucket_key=key) for key in band_keys])
        for child in session.scalars(select(PaperVersion).where(PaperVersion.parent_version_id == version.id)):
            child.change_summary, _ = _compare_versions(parsed.markdown, child.markdown, new_label)
        session.commit()
        try:
            if progress:
                progress("更新 Qdrant 检索索引", 88)
            vector_index = index_paper_content(
                paper_id=paper.id, version_id=version.id, title=parsed.title,
                markdown=parsed.markdown, source_filename=filename,
                parser_name=parsed.parser_name, pages=parsed.pages,
            )
        except Exception as exc:
            vector_index = {"status": "failed", "error_type": exc.__class__.__name__,
                            "error_detail": str(exc)[:240]}
            structured["warnings"].append(f"Qdrant indexing failed ({exc.__class__.__name__}); retry via the version index endpoint.")
        structured["vector_index"] = vector_index
        version.structured_json = dict(structured)
        session.commit()
        versions = list(session.scalars(select(PaperVersion).where(PaperVersion.paper_id == paper.id)
                                        .order_by(PaperVersion.created_at.desc())))
        result = {"duplicate": False, "duplicate_action": "replace", "replaced_version_id": version.id,
                  "version_id": version.id, "linked_by": "content_overlap", "version_linked": bool(version.parent_version_id),
                  "duplicate_candidates": [], "parser": parsed.parser_name, "needs_ocr": parsed.needs_ocr,
                  "warnings": structured["warnings"], "vector_index": vector_index,
                  "structured": structured, "markdown": parsed.markdown,
                  "change_summary": change_summary, **_paper_json(paper, versions)}
    storage_root = Path(get_settings().paper_storage_dir)
    if not storage_root.is_absolute():
        storage_root = PROJECT_ROOT / storage_root
    storage_root = storage_root.resolve()
    if old_path and old_path != pdf_path:
        safe_old = old_path.resolve()
        if safe_old.is_relative_to(storage_root):
            try:
                safe_old.unlink(missing_ok=True)
            except OSError:
                pass
    if old_sha and old_sha != source_sha:
        old_assets = (storage_root / "assets" / old_sha).resolve()
        if old_assets.is_relative_to(storage_root) and old_assets.is_dir():
            try:
                shutil.rmtree(old_assets)
            except OSError:
                pass
    return result


def _paper_storage_dir() -> Path:
    storage_dir = Path(get_settings().paper_storage_dir)
    return storage_dir if storage_dir.is_absolute() else PROJECT_ROOT / storage_dir


def _run_pdf_pipeline(filename: str, content: bytes, source_sha: str, *,
                      progress: Callable[[str, int], None] | None = None) -> tuple[ParsedPdf, dict]:
    settings = get_settings()
    if settings.formula_ocr_enabled or settings.pdf_parser.strip().lower() in {"auto", "paddleocr"}:
        # Windows native DLLs are sensitive to import order. Preload these
        # before rendering pages or loading other models in a long PDF job.
        try:
            import torch  # noqa: F401
            from paddleocr import FormulaRecognitionPipeline, PPStructureV3  # noqa: F401
        except Exception:
            pass  # The parser/formula stage records the actionable warning.
    assets_dir = _paper_storage_dir() / "assets" / source_sha
    asset_url_prefix = (
        f"{settings.public_api_base_url.rstrip('/')}{settings.api_prefix}"
        f"/papers/assets/{source_sha}"
    )
    if progress:
        progress("解析版面、正文与图表", 15)
    parsed = parse_pdf(
        filename,
        content,
        assets_dir=assets_dir,
        asset_url_prefix=asset_url_prefix,
        progress=progress,
    )
    if progress:
        progress("识别公式", 47)
    if parsed.parser_name == "paddleocr":
        # PPStructureV3 already runs its formula recognition model and writes
        # LaTeX into page Markdown. Avoid loading a second large PDF pipeline.
        formulas = []
        for page in parsed.pages:
            for index, match in enumerate(re.finditer(r"\$\$\s*(.*?)\s*\$\$", page.get("text", ""), re.S), start=1):
                latex = match.group(1).strip()
                if latex:
                    formulas.append({"page": page.get("page"), "formula_index": index,
                                     "latex": latex, "region_id": None, "bbox": None})
        formula_extraction = {"status": "layout_integrated", "model": "PPStructureV3",
                              "count": len(formulas), "formulas": formulas}
    else:
        formula_extraction = extract_pdf_formulas(filename, content)
    formulas = formula_extraction.get("formulas", [])
    if formulas:
        by_page: dict[int, list[dict]] = {}
        for formula in formulas:
            by_page.setdefault(int(formula["page"]), []).append(formula)
        if parsed.parser_name != "paddleocr":
            formula_markdown = ["## Recognized equations (local formula OCR)"]
            for page_number, page_formulas in sorted(by_page.items()):
                for formula in page_formulas:
                    formula_markdown.extend([
                        f"### Page {page_number} · Formula {formula['formula_index']}",
                        f"$${formula['latex']}$$",
                    ])
            parsed.markdown = "\n\n".join([parsed.markdown, *formula_markdown])
            parsed.full_text = "\n\n".join([
                parsed.full_text,
                *[formula["latex"] for formula in formulas],
            ])
        for page in parsed.pages:
            page_number = page.get("page")
            page_formulas = by_page.get(page_number, [])
            if page_formulas:
                page["formulas"] = page_formulas
                if parsed.parser_name != "paddleocr":
                    page["text"] = "\n\n".join([
                        page.get("text", ""),
                        *[f"$$\n{item['latex']}\n$$" for item in page_formulas],
                    ]).strip()
    elif formula_extraction.get("warning"):
        parsed.warnings = (parsed.warnings or []) + [formula_extraction["warning"]]
    return parsed, formula_extraction


def reparse_paper_version(paper_id: str, version_id: str, *,
                          progress: Callable[[str, int], None] | None = None) -> dict | None:
    """Re-run the active PDF pipeline for an already stored version."""
    ensure_paper_tables()
    with Session(mysql_engine(), expire_on_commit=False) as session:
        version = session.get(PaperVersion, version_id)
        if version is None or version.paper_id != paper_id:
            return None
        stored_pdf_path = Path(version.stored_pdf_path)
        filename = version.source_filename
        source_sha = version.source_sha256

    if not stored_pdf_path.is_file():
        raise ValueError("The original PDF for this version is no longer available")
    content = stored_pdf_path.read_bytes()
    parsed, formula_extraction = _run_pdf_pipeline(filename, content, source_sha, progress=progress)
    if progress:
        progress("更新文献与版本解析", 74)
    arxiv_id, doi, version_label = _stable_identifiers(filename, parsed.full_text)
    title = parsed.title.strip() or Path(filename).stem
    structured = {
        "metadata": {
            "title": title,
            "authors": parsed.authors or [],
            "abstract": parsed.abstract or "",
            "arxiv_id": arxiv_id,
            "doi": doi,
            "version_label": version_label,
        },
        "pages": parsed.pages,
        "sections": parsed.sections,
        "references": parsed.references or [],
        "parser": parsed.parser_name,
        "needs_ocr": parsed.needs_ocr,
        "warnings": parsed.warnings or [],
        "formula_extraction": formula_extraction,
        "figures": parsed.figures or [],
        "layout_pipeline": {
            "layout_and_text": parsed.parser_name,
            "figure_regions": "extracted" if parsed.figures else "not_available_from_selected_parser",
            "table_regions": (
                "layout_markdown"
                if parsed.parser_name in {"marker", "paddleocr"}
                else "not_available_from_selected_parser"
            ),
            "equation_regions": {
                "status": formula_extraction.get("status", "unknown"),
                "model": formula_extraction.get("model"),
                "count": formula_extraction.get("count", 0),
            },
            "inline_scripts": "pymupdf_geometry_when_available",
            "figure_analysis": (
                "completed" if any(
                    (figure.get("analysis") or {}).get("status") == "completed"
                    for figure in (parsed.figures or [])
                ) else "pending_multimodal_configuration"
            ),
        },
    }

    with Session(mysql_engine(), expire_on_commit=False) as session:
        paper = session.get(Paper, paper_id)
        version = session.get(PaperVersion, version_id)
        if paper is None or version is None or version.paper_id != paper.id:
            return None
        version.title = title
        version.authors = parsed.authors or []
        version.abstract = parsed.abstract or ""
        version.parser_name = parsed.parser_name
        version.needs_ocr = parsed.needs_ocr
        version.markdown = parsed.markdown
        version.structured_json = structured
        if title and (not paper.canonical_title or version.version_label == "v1"):
            paper.canonical_title = title
        session.commit()
        try:
            if progress:
                progress("重建 Qdrant 检索索引", 87)
            vector_index = index_paper_content(
                paper_id=paper.id,
                version_id=version.id,
                title=title,
                markdown=parsed.markdown,
                source_filename=filename,
                parser_name=parsed.parser_name,
                pages=parsed.pages,
            )
        except Exception as exc:
            vector_index = {"status": "failed", "error_type": exc.__class__.__name__,
                            "error_detail": str(exc)[:240]}
            structured["warnings"].append(
                f"Qdrant reindexing failed ({exc.__class__.__name__}); retry via the version index endpoint."
            )
        structured["vector_index"] = vector_index
        version.structured_json = dict(structured)
        session.commit()
        return {
            "paper_id": paper.id,
            "version_id": version.id,
            "parser": parsed.parser_name,
            "needs_ocr": parsed.needs_ocr,
            "warnings": structured["warnings"],
            "vector_index": vector_index,
            "structured": structured,
            "markdown": parsed.markdown,
        }

def list_papers(limit: int = 50) -> list[dict]:
    ensure_paper_tables()
    with Session(mysql_engine(), expire_on_commit=False) as session:
        papers = list(session.scalars(select(Paper).order_by(Paper.created_at.desc()).limit(limit)))
        output = []
        for paper in papers:
            versions = list(session.scalars(
                select(PaperVersion).where(PaperVersion.paper_id == paper.id)
                .order_by(PaperVersion.created_at.desc())
            ))
            output.append(_paper_json(paper, versions))
        return output


def delete_paper(paper_id: str) -> bool:
    """Remove a canonical paper, all its versions, and their local search data."""
    ensure_paper_tables()
    settings = get_settings()
    storage_root = Path(settings.paper_storage_dir)
    if not storage_root.is_absolute():
        storage_root = PROJECT_ROOT / storage_root
    storage_root = storage_root.resolve()
    with Session(mysql_engine()) as session:
        paper = session.get(Paper, paper_id)
        if paper is None:
            return False
        versions = list(session.scalars(select(PaperVersion).where(PaperVersion.paper_id == paper_id)))
        version_ids = [version.id for version in versions]
        source_hashes = [version.source_sha256 for version in versions]
        pdf_paths = [Path(version.stored_pdf_path) for version in versions if version.stored_pdf_path]

        session.execute(update(ExternalPaperRecord).where(ExternalPaperRecord.paper_id == paper_id).values(
            paper_id=None, version_id=None, download_status="not_attempted",
            status_detail="本地入库论文已删除，可重新获取全文。"))
        session.execute(delete(ReadingPreference).where(ReadingPreference.paper_id == paper_id))
        if version_ids:
            session.execute(delete(ParagraphTranslation).where(ParagraphTranslation.version_id.in_(version_ids)))
            session.execute(delete(PaperFingerprintBand).where(PaperFingerprintBand.version_id.in_(version_ids)))
            session.execute(update(PaperVersion).where(PaperVersion.id.in_(version_ids)).values(parent_version_id=None))
            session.execute(delete(PaperVersion).where(PaperVersion.id.in_(version_ids)))
        session.delete(paper)
        session.commit()

    try:
        client = qdrant_client()
        collection = settings.qdrant_papers_collection
        if client.collection_exists(collection):
            client.delete(
                collection_name=collection,
                points_selector=qdrant_models.FilterSelector(filter=qdrant_models.Filter(must=[
                    qdrant_models.FieldCondition(key="paper_id", match=qdrant_models.MatchValue(value=paper_id))
                ])),
                wait=True,
            )
    except Exception:
        import logging
        logging.getLogger(__name__).exception("Paper %s removed from database; vector cleanup pending", paper_id)

    for path in pdf_paths:
        resolved = path.resolve()
        if resolved.is_relative_to(storage_root) and resolved.is_file():
            try:
                resolved.unlink()
            except OSError:
                pass
    for source_hash in source_hashes:
        assets = (storage_root / "assets" / source_hash).resolve()
        if assets.is_relative_to(storage_root) and assets.is_dir():
            try:
                shutil.rmtree(assets)
            except OSError:
                pass
    return True


def ingest_external_jats(
    *, title: str, doi: str | None, authors: list[str], source_name: str,
    source_url: str, license_name: str | None, content: bytes,
) -> dict:
    """Store a legally available JATS full text and index its text without fabricating a PDF."""
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise ValueError("The open full-text XML is not valid JATS/XML") from exc

    def local_name(element: ET.Element) -> str:
        return element.tag.rsplit("}", 1)[-1].lower()

    def node_text(element: ET.Element) -> str:
        return " ".join(" ".join(element.itertext()).split())

    lines = [f"# {title}"]
    abstract_node = next((node for node in root.iter() if local_name(node) == "abstract"), None)
    if abstract_node is not None:
        abstract_text = node_text(abstract_node)
        if abstract_text:
            lines.extend(["", "## Abstract", "", abstract_text])
    body = next((node for node in root.iter() if local_name(node) == "body"), None)

    def add_section(section: ET.Element, depth: int = 2) -> None:
        section_title = next((child for child in section if local_name(child) in {"title", "label"}), None)
        if section_title is not None:
            heading = node_text(section_title)
            if heading:
                lines.extend(["", f"{'#' * min(depth, 6)} {heading}", ""])
        for child in section:
            kind = local_name(child)
            if kind == "p":
                text = node_text(child)
                if text:
                    lines.extend([text, ""])
            elif kind == "sec":
                add_section(child, depth + 1)
            elif kind in {"fig", "table-wrap"}:
                label = next((item for item in child if local_name(item) == "label"), None)
                caption = next((item for item in child if local_name(item) == "caption"), None)
                description = " ".join(filter(None, [node_text(label) if label is not None else "",
                                                        node_text(caption) if caption is not None else ""]))
                if description:
                    lines.extend([description, ""])

    if body is not None:
        for section in body:
            if local_name(section) == "sec":
                add_section(section)
            elif local_name(section) == "p":
                text = node_text(section)
                if text:
                    lines.extend(["", text])
    back = next((node for node in root.iter() if local_name(node) == "back"), None)
    if back is not None:
        refs = []
        for ref in back.iter():
            if local_name(ref) == "ref":
                text = node_text(ref)
                if text:
                    refs.append(text)
        if refs:
            lines.extend(["", "## References", ""])
            lines.extend(f"[{index}] {value}" for index, value in enumerate(refs, start=1))
    markdown = "\n".join(lines).strip()
    if len(markdown) < 300:
        raise ValueError("The open full-text XML contains too little readable article text")

    sha = hashlib.sha256(content).hexdigest()
    storage_dir = _paper_storage_dir()
    storage_dir.mkdir(parents=True, exist_ok=True)
    file_path = storage_dir / f"{sha}.xml"
    if not file_path.exists():
        file_path.write_bytes(content)
    ensure_paper_tables()
    with Session(mysql_engine(), expire_on_commit=False) as session:
        exact = session.scalar(select(PaperVersion).where(PaperVersion.source_sha256 == sha))
        if exact:
            paper = session.get(Paper, exact.paper_id)
            versions = list(session.scalars(select(PaperVersion).where(PaperVersion.paper_id == exact.paper_id)
                                             .order_by(PaperVersion.created_at.desc())))
            return {"duplicate": True, "duplicate_kind": "exact_file", "paper_id": exact.paper_id,
                    "version_id": exact.id, "source_format": "xml", **_paper_json(paper, versions)}
        paper = session.scalar(select(Paper).where(Paper.doi == doi)) if doi else None
        if paper is None:
            paper = Paper(id=str(uuid.uuid4()), doi=doi, canonical_title=title)
            session.add(paper)
            session.flush()
        previous = session.scalar(select(PaperVersion).where(PaperVersion.paper_id == paper.id)
                                  .order_by(PaperVersion.created_at.desc()).limit(1))
        version_id = str(uuid.uuid4())
        pages = [{"page": None, "text": markdown}]
        structured = {"metadata": {"title": title, "authors": authors, "abstract": node_text(abstract_node) if abstract_node is not None else "",
                                   "doi": doi, "arxiv_id": None, "version_label": None},
                      "pages": pages, "sections": [], "references": [], "parser": "europepmc-jats",
                      "needs_ocr": False, "warnings": [], "source_format": "xml", "source_name": source_name,
                      "source_url": source_url, "license": license_name,
                      "text_status": "open_access_full_text_xml"}
        version = PaperVersion(id=version_id, paper_id=paper.id,
                               parent_version_id=previous.id if previous else None,
                               version_label="open-full-text", source_filename=f"{Path(source_name).name or 'article'}.xml"[:512],
                               source_sha256=sha, stored_pdf_path=str(file_path), title=title, authors=authors,
                               abstract=structured["metadata"]["abstract"], parser_name="europepmc-jats",
                               needs_ocr=False, structured_json=structured, markdown=markdown,
                               change_summary=None)
        session.add(version)
        session.commit()
        paper_id = paper.id
        try:
            vector_index = index_paper_content(paper_id=paper_id, version_id=version_id, title=title,
                markdown=markdown, source_filename=version.source_filename, parser_name="europepmc-jats", pages=pages)
        except Exception as exc:
            vector_index = {"status": "failed", "error_type": exc.__class__.__name__}
        structured["vector_index"] = vector_index
        version.structured_json = dict(structured)
        session.commit()
        versions = list(session.scalars(select(PaperVersion).where(PaperVersion.paper_id == paper_id)
                                         .order_by(PaperVersion.created_at.desc())))
        return {"duplicate": False, "paper_id": paper_id, "version_id": version_id,
                "source_format": "xml", "parser": "europepmc-jats", "vector_index": vector_index,
                **_paper_json(paper, versions)}


def get_paper_version(paper_id: str, version_id: str) -> dict | None:
    ensure_paper_tables()
    with Session(mysql_engine(), expire_on_commit=False) as session:
        paper = session.get(Paper, paper_id)
        version = session.get(PaperVersion, version_id)
        if not paper or not version or version.paper_id != paper.id:
            return None
        return {
            "paper_id": paper.id,
            "version_id": version.id,
            "version_label": version.version_label,
            "parent_version_id": version.parent_version_id,
            "title": version.title,
            "authors": version.authors,
            "abstract": version.abstract,
            "parser": version.parser_name,
            "needs_ocr": version.needs_ocr,
            "structured": version.structured_json,
            "markdown": version.markdown,
            "change_summary": version.change_summary,
        }


def reindex_paper_version(paper_id: str, version_id: str) -> dict | None:
    ensure_paper_tables()
    with Session(mysql_engine(), expire_on_commit=False) as session:
        paper = session.get(Paper, paper_id)
        version = session.get(PaperVersion, version_id)
        if not paper or not version or version.paper_id != paper.id:
            return None
        result = index_paper_content(
            paper_id=paper.id,
            version_id=version.id,
            title=version.title,
            markdown=version.markdown,
            source_filename=version.source_filename,
            parser_name=version.parser_name,
            pages=(version.structured_json or {}).get("pages", []),
        )
        updated = dict(version.structured_json or {})
        updated["vector_index"] = result
        version.structured_json = updated
        session.commit()
        return result
