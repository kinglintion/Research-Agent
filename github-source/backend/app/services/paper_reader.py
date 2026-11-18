"""Page-linked reading data and paragraph-aligned translation for stored PDFs."""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from bisect import bisect_right
from difflib import SequenceMatcher
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import ParagraphTranslation, PaperVersion
from .connections import mysql_engine
from .model_adapters import get_llm_client
from .paper_ingestion import ensure_paper_tables


HEADING_RE = re.compile(
    r"^(?:abstract|references|bibliography|appendix(?:\s+[A-Z])?|"
    r"llm usage disclosure|[A-Z]\s+[A-Z][^\n]{2,100}|"
    r"\d+(?:\.\d+){0,3}\s+[A-Z][^\n]{2,100})$",
    re.IGNORECASE,
)
REF_HEADING_RE = re.compile(r"(?im)^\s*(?:references|bibliography)\s*$")
REF_END_RE = re.compile(
    r"(?m)^\s*(?:[A-Z]\s+[A-Z][^\n]{2,90}|"
    r"(?i:appendix|supplementary|acknowledg(?:e)?ments|llm usage disclosure)\b[^\n]{0,90})\s*$"
)
NUMBERED_REF_RE = re.compile(r"(?m)^\s*(?:\[(\d{1,4})\]|(\d{1,4})[.)])\s+")
INLINE_CITATION_RE = re.compile(r"\[\s*(\d{1,4}(?:\s*[,;–-]\s*\d{1,4})*)\s*\]")
AUTHOR_YEAR_GROUP_RE = re.compile(r"\(([^)]*\b(?:19|20)\d{2}[a-z]?[^)]*)\)", re.IGNORECASE)
AUTHOR_YEAR_PART_RE = re.compile(
    r"\b([A-Z][A-Za-zÀ-ÿ-]+)(?:\s+et\s+al\.?|\s+and\s+[A-Z][A-Za-zÀ-ÿ-]+)?"
    r",?\s+((?:19|20)\d{2}[a-z]?)\b"
)
PROTECTED_RE = re.compile(
    r"```[\s\S]*?```|`[^`\n]+`|\$\$[\s\S]*?\$\$|(?<!\$)\$[^$\n]+\$(?!\$)|"
    r"\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]|"
    r"<sup>[^<]*</sup>|<sub>[^<]*</sub>|\\cite\{[^}]+\}|"
    r"\[\s*\d{1,4}(?:\s*[,;–-]\s*\d{1,4})*\s*\]|"
    r"\([^)]*\b(?:19|20)\d{2}[a-z]?[^)]*\)",
    re.IGNORECASE,
)


def _version(paper_id: str, version_id: str) -> PaperVersion | None:
    ensure_paper_tables()
    with Session(mysql_engine(), expire_on_commit=False) as session:
        version = session.get(PaperVersion, version_id)
        if version is None or version.paper_id != paper_id:
            return None
        return version


def pdf_file(paper_id: str, version_id: str) -> tuple[Path, str] | None:
    version = _version(paper_id, version_id)
    if version is None:
        return None
    if not version.stored_pdf_path.lower().endswith(".pdf"):
        return None
    path = Path(version.stored_pdf_path).resolve()
    storage = Path(get_settings().paper_storage_dir)
    if not storage.is_absolute():
        storage = Path(__file__).resolve().parents[3] / storage
    if not path.is_relative_to(storage.resolve()) or not path.is_file():
        return None
    return path, version.source_filename


def _split_page(text: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n", text or "") if block.strip()]


def _citation_numbers(text: str) -> list[int]:
    numbers: set[int] = set()
    for match in INLINE_CITATION_RE.finditer(text):
        value = match.group(1)
        for part in re.split(r"\s*[,;]\s*", value):
            range_match = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
            if range_match:
                start, end = int(range_match.group(1)), int(range_match.group(2))
                if 0 < end - start <= 20:
                    numbers.update(range(start, end + 1))
            elif part.isdigit():
                numbers.add(int(part))
    return sorted(numbers)


def _numbered_references(pages: list[tuple[int, str]]) -> list[dict]:
    refs: list[dict] = []
    offsets: list[int] = []
    parts: list[str] = []
    position = 0
    for _, text in pages:
        offsets.append(position)
        parts.append(text)
        position += len(text) + 1
    joined = "\n".join(parts)
    matches = list(NUMBERED_REF_RE.finditer(joined))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(joined)
        raw = " ".join(joined[match.end():end].split()).strip()
        number = int(match.group(1) or match.group(2))
        page = pages[max(0, bisect_right(offsets, match.start()) - 1)][0]
        if raw and len(raw) <= 2000 and not any(item["number"] == number for item in refs):
            refs.append({"id": f"ref-{number}", "number": number, "text": raw, "page": page})
    return sorted(refs, key=lambda item: item["number"])


def _author_year_keys(text: str) -> list[str]:
    keys: set[str] = set()
    for group in AUTHOR_YEAR_GROUP_RE.finditer(text):
        for part in group.group(1).split(";"):
            match = AUTHOR_YEAR_PART_RE.search(part)
            if match:
                keys.add(f"{match.group(1).casefold()}:{match.group(2).casefold()}")
    return sorted(keys)


def _author_year_references(pages: list[tuple[int, str]]) -> list[dict]:
    """Read clearly separated bibliography entries; ambiguous matches remain unlinked."""
    entries: list[dict] = []
    pending: list[str] = []
    started_on = 0
    stop = False

    def flush() -> None:
        if not pending:
            return
        raw = " ".join(pending).strip()
        pending.clear()
        years = re.findall(r"\b(?:19|20)\d{2}[a-z]?\b", raw, re.IGNORECASE)
        if not years or not 30 <= len(raw) <= 2000:
            return
        first_author = raw.split(",", 1)[0].split(".", 1)[0].strip()
        surname = first_author.split()[-1] if first_author else ""
        if not re.fullmatch(r"[A-Za-zÀ-ÿ-]{2,60}", surname):
            return
        year = years[0].casefold()
        key = f"{surname.casefold()}:{year}"
        label = f"{surname}, {year}"
        entries.append({"id": f"ref-ay-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]}",
                        "number": None, "label": label, "author_year_key": key,
                        "text": raw, "page": started_on})

    for page, text in pages:
        for block in _split_page(text):
            plain = " ".join(block.split()).strip()
            if not plain or plain.isdigit() or plain.casefold() == "preprint. under review.":
                continue
            if not pending and len(plain) < 85 and "." not in plain and "," not in plain and not re.search(r"\d{4}", plain):
                stop = True
                break
            if pending and re.search(r"\b(?:19|20)\d{2}[a-z]?\b", " ".join(pending), re.IGNORECASE):
                if re.match(r"^[A-Z][A-Za-zÀ-ÿ-]+(?:\s+[A-Z][A-Za-zÀ-ÿ-]+|\.)", plain):
                    flush()
            if not pending:
                started_on = page
            pending.append(plain)
            joined = " ".join(pending)
            if re.search(r"\b(?:19|20)\d{2}[a-z]?\b", joined, re.IGNORECASE) and plain.endswith("."):
                flush()
        if stop:
            break
    flush()
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["author_year_key"]] = counts.get(entry["author_year_key"], 0) + 1
    return [entry for entry in entries if counts[entry["author_year_key"]] == 1]


def _pdf_outline(path: Path) -> list[dict]:
    try:
        import pymupdf

        with pymupdf.open(path) as document:
            return [
                {"id": f"toc-{index}", "level": max(1, min(level, 6)), "title": title, "page": page}
                for index, (level, title, page, *_) in enumerate(document.get_toc())
                if title and page > 0
            ]
    except (ImportError, OSError, ValueError, RuntimeError):
        return []


def _embedded_pages(path: Path) -> list[dict]:
    """Supply page anchors when a Markdown-first parser omitted page records."""
    try:
        import pymupdf

        with pymupdf.open(path) as document:
            return [
                {"page": index, "text": page.get_text("text").strip()}
                for index, page in enumerate(document, start=1)
            ]
    except (ImportError, OSError, ValueError, RuntimeError):
        return []


def reader_document(paper_id: str, version_id: str) -> dict | None:
    version = _version(paper_id, version_id)
    located = pdf_file(paper_id, version_id)
    if version is None or located is None:
        return None
    structured = version.structured_json or {}
    pages = structured.get("pages") or _embedded_pages(located[0])
    paragraphs: list[dict] = []
    ref_pages: list[tuple[int, str]] = []
    outline = _pdf_outline(located[0])
    has_pdf_outline = bool(outline)
    in_references = False
    references_complete = False
    for entry in pages:
        page = int(entry.get("page") or 0)
        if page <= 0:
            continue
        text = str(entry.get("text") or "")
        if not in_references and not references_complete:
            match = REF_HEADING_RE.search(text)
            if match:
                in_references = True
                body, reference_text = text[:match.start()], text[match.end():]
                ref_pages.append((page, reference_text))
            else:
                body = text
        elif in_references:
            end_match = REF_END_RE.search(text)
            if end_match:
                in_references = False
                references_complete = True
                body = text[end_match.start():]
                if end_match.start():
                    ref_pages.append((page, text[:end_match.start()]))
            else:
                body = ""
                ref_pages.append((page, text))
        else:
            body = text

        for sequence, block in enumerate(_split_page(body), start=1):
            digest = hashlib.sha256(block.encode("utf-8")).hexdigest()[:10]
            paragraph_id = f"p{page:04d}-{sequence:03d}-{digest}"
            first_line = block.splitlines()[0].strip().lstrip("# ")
            is_heading = len(block) < 160 and bool(HEADING_RE.fullmatch(first_line))
            if is_heading and not has_pdf_outline:
                outline.append({
                    "id": f"heading-{paragraph_id}",
                    "level": min(first_line.count(".") + 1, 4),
                    "title": first_line,
                    "page": page,
                    "paragraph_id": paragraph_id,
                })
            paragraphs.append({
                "id": paragraph_id,
                "page": page,
                "text": block,
                "kind": "heading" if is_heading else "paragraph",
                "citation_numbers": _citation_numbers(block),
            })

    numbered = _numbered_references(ref_pages)
    author_year = _author_year_references(ref_pages)
    cited_numbers = {number for paragraph in paragraphs for number in paragraph["citation_numbers"]}
    numbered_has_body_links = any(item["number"] in cited_numbers for item in numbered)
    references = numbered if numbered_has_body_links or not author_year else author_year
    references_by_number = {item["number"]: item for item in references if item["number"] is not None}
    references_by_author_year = {
        item["author_year_key"]: item for item in references if item.get("author_year_key")
    }
    for paragraph in paragraphs:
        paragraph["author_year_keys"] = _author_year_keys(paragraph["text"])
        paragraph["linked_reference_ids"] = [
            references_by_number[number]["id"]
            for number in paragraph["citation_numbers"] if number in references_by_number
        ] + [
            references_by_author_year[key]["id"]
            for key in paragraph["author_year_keys"] if key in references_by_author_year
        ]
    for reference in references:
        reference["backlink_paragraph_ids"] = [
            paragraph["id"] for paragraph in paragraphs
            if reference["id"] in paragraph["linked_reference_ids"]
        ]

    try:
        import pymupdf

        with pymupdf.open(located[0]) as document:
            page_count = len(document)
    except (ImportError, OSError, ValueError, RuntimeError):
        page_count = max((int(entry.get("page") or 0) for entry in pages), default=0)

    return {
        "paper_id": paper_id,
        "version_id": version_id,
        "title": version.title,
        "authors": version.authors or [],
        "source_filename": version.source_filename,
        "page_count": page_count,
        "outline": outline,
        "paragraphs": paragraphs,
        "references": references,
        "figures": structured.get("figures") or [],
        "reference_status": (
            "numbered_links_available" if references is numbered and numbered else
            "author_year_links_available" if author_year else "references_not_reliably_detected"
        ),
        "text_status": "parsed" if structured.get("pages") else (
            "embedded_text_fallback" if paragraphs else "no_page_text"
        ),
    }


def _layout_key(text: str) -> str:
    plain = html.unescape(re.sub(r"<[^>]+>", "", text))
    return "".join(char.casefold() for char in unicodedata.normalize("NFKC", plain) if char.isalnum())


def match_page_blocks(paragraphs: list[dict], blocks: list[tuple]) -> list[dict]:
    """Map parsed paragraphs onto native PDF text blocks; uncertain matches stay on the source page."""
    text_blocks = [block for block in blocks if len(block) >= 7 and block[6] == 0 and str(block[4]).strip()]
    keys = [_layout_key(str(block[4])) for block in text_blocks]
    used: set[int] = set()
    regions: list[dict] = []
    for paragraph in paragraphs:
        source = _layout_key(str(paragraph.get("text") or ""))
        if len(source) < 4:
            continue
        best_index, best_score = -1, 0.0
        for index, target in enumerate(keys):
            if index in used or len(target) < 4:
                continue
            if source == target:
                score = 1.0
            elif source in target or target in source:
                score = min(len(source), len(target)) / max(len(source), len(target))
            else:
                score = SequenceMatcher(None, source[:160], target[:160], autojunk=False).ratio()
            if score > best_score:
                best_index, best_score = index, score
        if best_index < 0 or best_score < 0.72:
            continue
        used.add(best_index)
        block = text_blocks[best_index]
        x0, y0, x1, y1 = (float(value) for value in block[:4])
        if x1 <= x0 or y1 <= y0:
            continue
        if y1 - y0 > (x1 - x0) * 3 and len(source) < 100:
            continue  # Preserve vertical arXiv stamps and rotated marginal text.
        lines = max(1, len(str(block[4]).splitlines()))
        font_size = min(19.0, max(6.5, (y1 - y0) / lines * 0.82))
        regions.append({"paragraph_id": paragraph["id"], "bbox": [x0, y0, x1, y1],
                        "font_size": round(font_size, 2), "match_score": round(best_score, 3)})
    return regions


def reader_page_layout(paper_id: str, version_id: str, page_number: int) -> dict | None:
    located = pdf_file(paper_id, version_id)
    if located is None:
        return None
    document = reader_document(paper_id, version_id)
    if document is None:
        return None
    try:
        import pymupdf

        with pymupdf.open(located[0]) as pdf:
            if page_number < 1 or page_number > len(pdf):
                return None
            page = pdf[page_number - 1]
            regions = match_page_blocks(
                [item for item in document["paragraphs"] if item["page"] == page_number],
                page.get_text("blocks"),
            )
            return {"page": page_number, "width": page.rect.width, "height": page.rect.height,
                    "rotation": page.rotation, "regions": regions,
                    "matched": len(regions),
                    "total": sum(item["page"] == page_number for item in document["paragraphs"])}
    except (ImportError, OSError, ValueError, RuntimeError):
        return None


def _protect(text: str) -> tuple[str, dict[str, str]]:
    kept: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        key = f"KEEP_TOKEN_{len(kept):04d}"
        kept[key] = match.group(0)
        return key

    return PROTECTED_RE.sub(replace, text), kept


def translate_paragraph(
    paper_id: str, version_id: str, paragraph_id: str, target_language: str = "zh-CN"
) -> dict | None:
    if target_language not in {"zh-CN", "en"}:
        raise ValueError("target_language must be zh-CN or en")
    document = reader_document(paper_id, version_id)
    if document is None:
        return None
    paragraph = next((item for item in document["paragraphs"] if item["id"] == paragraph_id), None)
    if paragraph is None:
        return None
    source = paragraph["text"]
    source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    ensure_paper_tables()
    with Session(mysql_engine(), expire_on_commit=False) as session:
        saved = session.scalar(select(ParagraphTranslation).where(
            ParagraphTranslation.version_id == version_id,
            ParagraphTranslation.paragraph_id == paragraph_id,
            ParagraphTranslation.target_language == target_language,
        ))
        if saved and saved.source_sha256 == source_sha:
            return {"paragraph_id": paragraph_id, "page": paragraph["page"],
                    "target_language": target_language, "translation": saved.translated_text, "cached": True}

    protected, kept = _protect(source)
    settings = get_settings()
    translated = ""
    for _ in range(2):
        request = {
            "model": settings.llm_model,
            "messages": [
                {"role": "system", "content": (
                    f"Translate this single academic paragraph into {target_language}. "
                    "Return only the translation. Preserve each KEEP_TOKEN_#### exactly once, "
                    "in its original position relative to the sentence. Do not translate, "
                    "edit, remove, or add these tokens. Preserve paragraph breaks."
                )},
                {"role": "user", "content": protected},
            ],
            "temperature": 0,
            "max_tokens": min(4096, max(512, len(source) * 2)),
        }
        if "deepseek" in settings.llm_api_base.lower():
            request["extra_body"] = {"thinking": {"type": "disabled"}}
        response = get_llm_client().chat.completions.create(**request)
        translated = (response.choices[0].message.content or "").strip()
        if translated and all(translated.count(key) == 1 for key in kept):
            break
    else:
        raise ValueError("The translation changed a formula, code span, or citation marker")
    for key, original in kept.items():
        translated = translated.replace(key, original)

    with Session(mysql_engine(), expire_on_commit=False) as session:
        saved = session.scalar(select(ParagraphTranslation).where(
            ParagraphTranslation.version_id == version_id,
            ParagraphTranslation.paragraph_id == paragraph_id,
            ParagraphTranslation.target_language == target_language,
        ))
        if saved is None:
            saved = ParagraphTranslation(version_id=version_id, paragraph_id=paragraph_id,
                                         target_language=target_language, source_sha256=source_sha,
                                         translated_text=translated)
            session.add(saved)
        else:
            saved.source_sha256 = source_sha
            saved.translated_text = translated
        session.commit()
    return {"paragraph_id": paragraph_id, "page": paragraph["page"],
            "target_language": target_language, "translation": translated, "cached": False}


def translate_selection(
    paper_id: str, version_id: str, paragraph_id: str, selected_text: str, target_language: str = "zh-CN"
) -> dict | None:
    """Translate a selected phrase in context while preserving math/code/citation tokens."""
    if target_language not in {"zh-CN", "en"}:
        raise ValueError("target_language must be zh-CN or en")
    selected = " ".join(selected_text.split())
    if not selected:
        raise ValueError("Select text to translate")
    document = reader_document(paper_id, version_id)
    paragraph = next((item for item in document["paragraphs"] if item["id"] == paragraph_id), None) if document else None
    if paragraph is None:
        return None
    source = paragraph["text"]
    if re.sub(r"\s+", "", selected).casefold() not in re.sub(r"\s+", "", source).casefold():
        return None
    protected, kept = _protect(selected)
    settings = get_settings()
    request = {"model": settings.llm_model, "messages": [
        {"role": "system", "content": f"Translate the selected academic text into {target_language}. Use the paragraph context only to resolve meaning. Return only the translation. Preserve every KEEP_TOKEN_#### exactly once."},
        {"role": "user", "content": f"Paragraph context:\n{source[:3000]}\n\nSelected text:\n{protected}"}],
        "temperature": 0, "max_tokens": min(1200, max(128, len(selected) * 2))}
    if "deepseek" in settings.llm_api_base.lower():
        request["extra_body"] = {"thinking": {"type": "disabled"}}
    response = get_llm_client().chat.completions.create(**request)
    translated = (response.choices[0].message.content or "").strip()
    if not translated or any(translated.count(key) != 1 for key in kept):
        raise ValueError("Translation changed a formula, code span, or citation marker")
    for key, original in kept.items():
        translated = translated.replace(key, original)
    return {"paragraph_id": paragraph_id, "page": paragraph["page"], "source": selected,
            "target_language": target_language, "translation": translated}


def _clean_streamed_translation(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text).strip()
    return re.sub(r"\s+([,.;:!?，。；：！？、])", r"\1", text)


def stream_selection_translation(
    paper_id: str, version_id: str, page: int, selected_text: str,
    context_before: str = "", context_after: str = "", target_language: str = "zh-CN",
):
    """Yield model deltas and a validated final translation for a PDF.js text-layer selection."""
    if target_language not in {"zh-CN", "en"}:
        raise ValueError("target_language must be zh-CN or en")
    selected = " ".join(selected_text.split())
    if not selected or len(selected) > 2000:
        raise ValueError("Select between 2 and 2000 characters")
    document = reader_document(paper_id, version_id)
    if document is None:
        raise ValueError("Paper version or PDF not found")
    page_paragraphs = [item["text"] for item in document["paragraphs"] if item["page"] == page]
    if not page_paragraphs:
        raise ValueError("No parsed context is available for the selected PDF page")
    page_context = "\n\n".join(page_paragraphs)
    before = " ".join(context_before.split())[-700:]
    after = " ".join(context_after.split())[:700]
    # The parsed page provides a server-side context check; the PDF.js snippets preserve local layout context.
    context = page_context[:5000]
    if before or after:
        context = f"PDF text layer before selection:\n{before}\n\nPDF text layer after selection:\n{after}\n\nParsed page text:\n{context}"
    protected, kept = _protect(selected)
    settings = get_settings()
    request = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": (
                f"Translate only the selected academic passage into {target_language}. Use surrounding context "
                "to resolve terminology and pronouns. Return only the translation. Preserve punctuation and "
                "every KEEP_TOKEN_#### exactly once; do not translate or alter those tokens. Do not add commentary."
            )},
            {"role": "user", "content": f"Surrounding context (do not translate):\n{context}\n\nSelected text to translate:\n{protected}"},
        ],
        "temperature": 0,
        "max_tokens": min(1600, max(192, len(selected) * 2)),
        "stream": True,
    }
    if "deepseek" in settings.llm_api_base.lower():
        request["extra_body"] = {"thinking": {"type": "disabled"}}
    stream = get_llm_client().chat.completions.create(**request)
    parts: list[str] = []
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content or ""
        if delta:
            parts.append(delta)
            yield "token", delta
    translated = "".join(parts).strip()
    if not translated or any(translated.count(key) != 1 for key in kept):
        raise ValueError("The streamed translation changed a formula, code span, or citation marker")
    for key, original in kept.items():
        translated = translated.replace(key, original)
    yield "done", _clean_streamed_translation(translated)
