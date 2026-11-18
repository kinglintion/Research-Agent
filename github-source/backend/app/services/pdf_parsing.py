from __future__ import annotations

import html
import importlib.util
import re
import statistics
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import quote

import httpx
from pypdf import PdfReader

from ..config import get_settings


@dataclass
class ParsedPdf:
    parser_name: str
    markdown: str
    full_text: str
    pages: list[dict]
    sections: list[dict]
    title: str = ""
    authors: list[str] | None = None
    abstract: str = ""
    references: list[dict] | None = None
    warnings: list[str] | None = None
    needs_ocr: bool = False
    figures: list[dict] | None = None


def _tei_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def _parse_grobid(filename: str, content: bytes, base_url: str) -> ParsedPdf:
    url = f"{base_url.rstrip('/')}/api/processFulltextDocument"
    response = httpx.post(
        url,
        files={"input": (filename, content, "application/pdf")},
        data={"consolidateHeader": "1", "consolidateCitations": "0"},
        timeout=180,
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}

    title = _tei_text(root.find(".//tei:teiHeader/tei:fileDesc/tei:titleStmt/tei:title", ns))
    abstract = "\n\n".join(
        text for text in (_tei_text(node) for node in root.findall(".//tei:profileDesc/tei:abstract/tei:p", ns)) if text
    )
    authors = []
    for author in root.findall(".//tei:sourceDesc/tei:biblStruct/tei:analytic/tei:author", ns):
        name = _tei_text(author.find("tei:persName", ns))
        if name:
            authors.append(name)

    references = []
    for index, entry in enumerate(root.findall(".//tei:listBibl/tei:biblStruct", ns), start=1):
        ref_title = _tei_text(entry.find(".//tei:analytic/tei:title", ns)) or _tei_text(
            entry.find(".//tei:monogr/tei:title", ns)
        )
        ref_authors = [
            name for name in (
                _tei_text(author.find("tei:persName", ns))
                for author in entry.findall(".//tei:author", ns)
            ) if name
        ]
        date_node = entry.find(".//tei:date", ns)
        doi_node = entry.find(".//tei:idno[@type='DOI']", ns)
        references.append({
            "index": index,
            "title": ref_title,
            "authors": ref_authors,
            "year": date_node.get("when", "")[:4] if date_node is not None else "",
            "doi": _tei_text(doi_node),
            "raw": _tei_text(entry),
        })

    sections: list[dict] = []
    markdown_parts = [f"# {title}" if title else f"# {filename}"]
    if authors:
        markdown_parts.append("**Authors:** " + "; ".join(authors))
    if abstract:
        markdown_parts.extend(["## Abstract", abstract])
    body = root.find(".//tei:text/tei:body", ns)

    def walk_div(div: ET.Element, depth: int = 2) -> None:
        heading = _tei_text(div.find("tei:head", ns)) or ""
        paragraphs = [_tei_text(p) for p in div.findall("tei:p", ns)]
        paragraphs = [paragraph for paragraph in paragraphs if paragraph]
        if heading:
            markdown_parts.append(f"{'#' * min(depth, 6)} {heading}")
        markdown_parts.extend(paragraphs)
        if heading or paragraphs:
            sections.append({"heading": heading, "paragraphs": paragraphs})
        for child in div.findall("tei:div", ns):
            walk_div(child, depth + 1)

    if body is not None:
        for div in body.findall("tei:div", ns):
            walk_div(div)
        if not sections:
            body_text = _tei_text(body)
            if body_text:
                markdown_parts.append(body_text)
    if references:
        markdown_parts.append("## References")
        markdown_parts.extend(
            f"{item['index']}. {item['title'] or item['raw']}"
            for item in references
        )

    markdown = "\n\n".join(part for part in markdown_parts if part)
    full_text = "\n\n".join([title, abstract, *[p for sec in sections for p in sec["paragraphs"]]])
    return ParsedPdf(
        parser_name="grobid",
        markdown=markdown,
        full_text=full_text,
        pages=[],
        sections=sections,
        title=title,
        authors=authors,
        abstract=abstract,
        references=references,
        warnings=[],
        needs_ocr=not bool(full_text.strip()),
    )


def _parse_pypdf(
    filename: str, content: bytes, warning: str | None = None, *,
    assets_dir: Path | None = None, asset_url_prefix: str | None = None,
    progress: Callable[[str, int], None] | None = None,
) -> ParsedPdf:
    import io

    reader = PdfReader(io.BytesIO(content), strict=False)
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise ValueError("PDF is encrypted and cannot be parsed without a password") from exc

    layout_pages = None
    try:
        layout_pages = _pymupdf_script_pages(content)
    except (ImportError, ModuleNotFoundError):
        warning = (warning + " " if warning else "") + (
            "PyMuPDF is not installed; inline superscript/subscript positioning "
            "may be lost in the text fallback."
        )
    except Exception as exc:
        warning = (warning + " " if warning else "") + (
            f"Position-aware text formatting failed ({exc.__class__.__name__}); "
            "plain PDF text extraction was used."
        )

    pages = []
    page_parts = []
    for index, page in enumerate(reader.pages, start=1):
        if layout_pages is not None and index <= len(layout_pages):
            text = layout_pages[index - 1]
        else:
            try:
                text = page.extract_text(extraction_mode="layout") or ""
            except TypeError:
                text = page.extract_text() or ""
        text = text.strip()
        pages.append({"page": index, "text": text})
        page_parts.append(f"## Page {index}\n\n{text}" if text else f"## Page {index}\n\n[No embedded text detected]")

    full_text = "\n\n".join(page["text"] for page in pages if page["text"])
    first_lines = [line.strip() for line in full_text.splitlines() if line.strip()]
    title = (getattr(reader.metadata, "title", None) or "").strip() if reader.metadata else ""
    if not title:
        try:
            title = _pymupdf_title(content)
        except (ImportError, ModuleNotFoundError, RuntimeError, ValueError):
            pass
    for line in first_lines[:16]:
        if len(line) < 12 or len(line) > 300:
            continue
        if re.search(r"(?i)^(arxiv|doi|abstract|preprint|proceedings|©|copyright)\b", line):
            continue
        if not title:
            title = line
        break

    abstract = ""
    abstract_match = re.search(
        r"(?is)\babstract\b\s*[:.]?\s*(.{40,5000}?)(?=\n\s*(?:\d+(?:\.\d+)*\s+)?(?:introduction|keywords|index terms)\b)",
        full_text,
    )
    if abstract_match:
        abstract = " ".join(abstract_match.group(1).split())

    references = _references_from_text(full_text)

    figures: list[dict] = []
    if assets_dir is not None:
        try:
            figures = _extract_pymupdf_figures(
                content, assets_dir=assets_dir, asset_url_prefix=asset_url_prefix,
                progress=progress,
            )
            for figure in figures:
                page_index = figure["page"] - 1
                caption = figure["caption"]
                page_parts[page_index] += (
                    f"\n\n![{caption}]({figure['url']})\n\n**{caption}**"
                )
                pages[page_index].setdefault("figures", []).append(figure["id"])
                description = (figure.get("analysis") or {}).get("description")
                if description:
                    page_parts[page_index] += f"\n\n**Figure description:** {description}"
                    pages[page_index]["text"] += f"\n\nFigure description: {description}"
        except Exception as exc:
            warning = (warning + " " if warning else "") + (
                f"PyMuPDF figure extraction failed ({exc.__class__.__name__})."
            )
    markdown = "\n\n".join([f"# {title or filename}", *page_parts])
    warnings = [warning] if warning else []
    chars_per_page = len(full_text) / max(len(pages), 1)
    needs_ocr = chars_per_page < 80
    if needs_ocr:
        warnings.append("Few embedded text characters were found; this PDF may require OCR.")
    return ParsedPdf(
        parser_name="pypdf",
        markdown=markdown,
        full_text=full_text,
        pages=pages,
        sections=[],
        title=title,
        authors=[],
        abstract=abstract,
        references=references,
        warnings=warnings,
        needs_ocr=needs_ocr,
        figures=figures,
    )


def _pymupdf_title(content: bytes) -> str:
    """Select the prominent first-page title, ignoring arXiv sidebars."""
    import pymupdf

    with pymupdf.open(stream=content, filetype="pdf") as document:
        if not document:
            return ""
        first = document[0]
        candidates: list[tuple[float, float, str]] = []
        for block in first.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = " ".join(span.get("text", "").strip() for span in spans).strip()
                if not 12 <= len(text) <= 250 or line["bbox"][1] > first.rect.height * 0.45:
                    continue
                if re.search(r"(?i)\b(?:arxiv:|doi:|copyright|provided proper attribution)\b", text):
                    continue
                size = max((float(span.get("size", 0)) for span in spans), default=0)
                if size >= 13:
                    candidates.append((size, -line["bbox"][1], text))
        return max(candidates)[2] if candidates else ""


def _extract_pymupdf_figures(
    content: bytes, *, assets_dir: Path, asset_url_prefix: str | None,
    progress: Callable[[str, int], None] | None = None,
) -> list[dict]:
    """Crop embedded and vector figures immediately above their PDF captions."""
    import pymupdf

    from .vision_analysis import analyze_figure
    from concurrent.futures import ThreadPoolExecutor, as_completed

    figures: list[dict] = []
    assets_dir.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(stream=content, filetype="pdf") as document:
        for page_index, page in enumerate(document):
            blocks = sorted(page.get_text("blocks"), key=lambda block: block[1])
            captions = [
                block for block in blocks
                if re.match(r"(?i)^\s*(?:figure|fig\.)\s*\d+[.:]", block[4].strip())
            ]
            if not captions:
                continue
            image_rects = [rect for image in page.get_images(full=True)
                           for rect in page.get_image_rects(image[0])]
            drawing_rects = [rect for rect in page.cluster_drawings()
                             if rect.width >= 45 and rect.height >= 24]
            regions = image_rects + drawing_rects
            previous_caption_bottom = 30.0
            for caption_block in captions:
                if len(figures) >= 20:
                    break
                caption_rect = pymupdf.Rect(caption_block[:4])
                nearby = [rect for rect in regions
                          if rect.y0 >= previous_caption_bottom - 3
                          and rect.y0 < caption_rect.y0 - 25
                          and rect.y1 <= caption_rect.y1 + 30
                          and -45 <= caption_rect.y0 - rect.y1 <= 150]
                previous_caption_bottom = caption_rect.y1
                if not nearby:
                    continue
                closest_bottom = max(rect.y1 for rect in nearby)
                selected = [rect for rect in nearby if closest_bottom - rect.y1 <= 85]
                crop = pymupdf.Rect(selected[0])
                for rect in selected[1:]:
                    crop |= rect
                crop += (-8, -8, 8, 8)
                crop &= page.rect
                crop.y1 = min(crop.y1, caption_rect.y0 - 4)
                if crop.width < 50 or crop.height < 30:
                    continue
                label_match = re.match(r"(?i)^\s*((?:figure|fig\.)\s*\d+)", caption_block[4])
                label = label_match.group(1) if label_match else f"Figure {len(figures) + 1}"
                caption = " ".join(caption_block[4].split())[:500]
                asset_name = f"figure-p{page_index + 1:03d}-{len(figures) + 1:03d}.png"
                asset_path = assets_dir / asset_name
                page.get_pixmap(matrix=pymupdf.Matrix(1.8, 1.8), clip=crop, alpha=False).save(asset_path)
                url = (f"{asset_url_prefix.rstrip('/')}/{quote(asset_name)}"
                       if asset_url_prefix else asset_name)
                figures.append({
                    "id": f"figure-{len(figures) + 1:03d}", "kind": "figure",
                    "label": label, "caption": caption, "page": page_index + 1,
                    "bbox": [round(value, 1) for value in crop], "asset_name": asset_name,
                    "url": url, "analysis": {"status": "pending"},
                })
                if progress:
                    progress(f"提取第 {page_index + 1} 页 {label}", min(30, 18 + len(figures)))
    if figures:
        with ThreadPoolExecutor(max_workers=min(4, len(figures))) as executor:
            tasks = {executor.submit(analyze_figure, assets_dir / figure["asset_name"],
                                     context=figure["caption"]): figure for figure in figures}
            for completed, future in enumerate(as_completed(tasks), start=1):
                try:
                    tasks[future]["analysis"] = future.result()
                except Exception as exc:
                    tasks[future]["analysis"] = {"status": "failed", "error_type": exc.__class__.__name__}
                if progress:
                    progress(f"理解图表 {completed}/{len(figures)}", 30 + int(13 * completed / len(figures)))
    return figures


def _pymupdf_script_pages(content: bytes) -> list[str]:
    """Extract text in positioned spans and encode detected scripts as HTML sup/sub.

    Marker remains the preferred layout parser. This improves the pypdf fallback
    without changing the source order of ordinary text spans.
    """
    import pymupdf

    document = pymupdf.open(stream=content, filetype="pdf")
    output = []
    for page in document:
        page_data = page.get_text("dict", sort=True)
        blocks = []
        for block in page_data.get("blocks", []):
            if "lines" not in block:
                continue
            lines = []
            for line in block["lines"]:
                spans = sorted(line.get("spans", []), key=lambda span: span.get("bbox", [0])[0])
                if not spans:
                    continue
                regular = [
                    span for span in spans
                    if not (int(span.get("flags", 0)) & 1)
                ]
                regular_sizes = [float(span.get("size", 0)) for span in regular]
                base_size = statistics.median(regular_sizes) if regular_sizes else max(
                    float(span.get("size", 0)) for span in spans
                )
                baseline_spans = [
                    span for span in regular
                    if float(span.get("size", 0)) >= base_size * 0.88
                ] or regular or spans
                baseline_values = [
                    float(span.get("origin", [0, span.get("bbox", [0, 0, 0, 0])[3]])[1])
                    for span in baseline_spans
                ]
                baseline = statistics.median(baseline_values)
                rendered_spans = []
                for span in spans:
                    value = str(span.get("text", ""))
                    if not value:
                        continue
                    size = float(span.get("size", base_size))
                    origin = span.get("origin", [0, span.get("bbox", [0, 0, 0, 0])[3]])
                    vertical_delta = float(origin[1]) - baseline
                    explicit_sup = bool(int(span.get("flags", 0)) & 1)
                    small = base_size > 0 and size <= base_size * 0.84
                    if explicit_sup or (small and vertical_delta < -base_size * 0.12):
                        value = f"<sup>{html.escape(value)}</sup>"
                    elif small and vertical_delta > base_size * 0.12:
                        value = f"<sub>{html.escape(value)}</sub>"
                    rendered_spans.append(value)
                lines.append("".join(rendered_spans))
            if lines:
                blocks.append("\n".join(lines))
        output.append("\n\n".join(blocks))
    document.close()
    return output


def _markdown_result(parser_name: str, markdown: str, warning: str | None = None) -> ParsedPdf:
    headings = list(re.finditer(r"^(#{1,6})\s+(.+)$", markdown, re.MULTILINE))
    sections = []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
        sections.append({"heading": heading.group(2).strip(), "text": markdown[heading.end():end].strip()})
    title = next((item["heading"] for item in sections if item["heading"]), "")
    abstract_match = re.search(
        r"(?is)^#{1,6}\s*abstract\s*\n(.*?)(?=^#{1,6}\s|\Z)", markdown, re.MULTILINE
    )
    abstract = abstract_match.group(1).strip() if abstract_match else ""
    full_text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", markdown)
    full_text = re.sub(r"[`*_>#]", " ", full_text)
    warnings = [warning] if warning else []
    return ParsedPdf(
        parser_name=parser_name,
        markdown=markdown,
        full_text=full_text,
        pages=[],
        sections=sections,
        title=title,
        authors=[],
        abstract=abstract,
        references=_references_from_text(full_text),
        warnings=warnings,
        needs_ocr=len(full_text.strip()) < 200,
    )


def _references_from_text(text: str) -> list[dict]:
    heading = re.search(r"(?im)^\s*(?:#{1,6}\s*)?(?:references|bibliography)\s*$", text)
    if not heading:
        return []
    raw = text[heading.end():]
    raw = re.split(r"(?im)^\s*(?:#{1,6}\s*)?(?:supplementary material|appendix|acknowledgements)\b[^\n]*$", raw)[0]
    if re.search(r"(?m)^\s*(?:\[\d+\]|\d+[.)]\s+)", raw):
        entries = re.split(r"\n\s*(?=\[\d+\]|\d+[.)]\s+)", raw)
    else:
        # Unnumbered bibliographies often wrap lines within an entry. The
        # publication year followed by a period is a safer boundary than a
        # blank line for PyMuPDF's geometry-based text output.
        entries = []
        start = 0
        for end in re.finditer(r"\b(?:19|20)\d{2}[a-z]?\.(?=\s|$)", raw, re.I):
            piece = raw[start:end.end()].strip()
            if len(piece) >= 30:
                entries.append(piece)
                start = end.end()
        if not entries and raw.strip():
            entries = [raw]
    return [{"index": index, "raw": " ".join(entry.split())}
            for index, entry in enumerate(entries, start=1) if entry.strip()]


def _figure_context(markdown: str, image_url: str) -> str:
    link_end = markdown.find(f"]({image_url})")
    if link_end < 0:
        return ""
    start = max(0, link_end - 1200)
    end = min(len(markdown), link_end + len(image_url) + 1200)
    context = markdown[start:end]
    context = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", context)
    context = re.sub(r"<[^>]+>|[`*_>#]", " ", context)
    return " ".join(context.split())[-2000:]


def _parse_marker(
    filename: str,
    content: bytes,
    *,
    assets_dir: Path | None = None,
    asset_url_prefix: str | None = None,
) -> ParsedPdf:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    with tempfile.TemporaryDirectory(prefix="research-copilot-marker-") as temp_dir:
        pdf_path = Path(temp_dir) / Path(filename).name
        pdf_path.write_bytes(content)
        converter = PdfConverter(artifact_dict=create_model_dict())
        rendered = converter(str(pdf_path))
        output = text_from_rendered(rendered)
        markdown = output[0] if isinstance(output, tuple) else str(output)
        images = output[2] if isinstance(output, tuple) and len(output) > 2 else getattr(rendered, "images", {})

    figures = []
    if images and assets_dir is not None:
        from .vision_analysis import analyze_figure

        assets_dir.mkdir(parents=True, exist_ok=True)
        for index, (source_name, image) in enumerate(images.items(), start=1):
            source_basename = Path(str(source_name).replace("\\", "/")).name
            suffix = Path(source_basename).suffix or ".png"
            asset_name = f"figure-{index:03d}{suffix.lower()}"
            asset_path = assets_dir / asset_name
            image.save(asset_path)
            asset_url = (
                f"{asset_url_prefix.rstrip('/')}/{quote(asset_name)}"
                if asset_url_prefix else asset_name
            )
            source_paths = {str(source_name).replace("\\", "/"), source_basename}
            for source_path in source_paths:
                markdown = markdown.replace(f"]({source_path})", f"]({asset_url})")
                markdown = markdown.replace(f"](./{source_path})", f"]({asset_url})")
            markdown = re.sub(
                rf"\]\((?:[^)]*/)?{re.escape(source_basename)}\)",
                f"]({asset_url})",
                markdown,
            )
            analysis = analyze_figure(
                asset_path,
                context=_figure_context(markdown, asset_url),
            )
            page_match = re.search(r"(?i)(?:page|p)[_-]?(\d+)", source_basename)
            figures.append({
                "id": f"figure-{index:03d}",
                "kind": "figure",
                "asset_name": asset_name,
                "url": asset_url,
                "page": int(page_match.group(1)) if page_match else None,
                "analysis": analysis,
            })

        # Some renderer versions return image assets without inserting a Markdown
        # reference. Append them explicitly so no extracted figure is silently lost.
        for figure in figures:
            if figure["url"] not in markdown:
                markdown += (
                    f"\n\n![Extracted figure {figure['id']}]({figure['url']})\n"
                )
            description = figure["analysis"].get("description")
            if description:
                markdown += f"\n\n**Figure description:** {description}\n"
            elif figure["analysis"].get("status") == "pending_configuration":
                markdown += (
                    "\n\n> Figure semantics are pending multimodal API configuration. "
                    "The extracted image is retained above.\n"
                )

    parsed = _markdown_result("marker", markdown)
    parsed.figures = figures
    return parsed


def _parse_paddleocr(
    filename: str,
    content: bytes,
    *,
    assets_dir: Path | None = None,
    asset_url_prefix: str | None = None,
    progress: Callable[[str, int], None] | None = None,
) -> ParsedPdf:
    # Preserve the Windows DLL import order verified for this environment.
    import torch  # noqa: F401
    from paddleocr import PPStructureV3

    figures = []
    with tempfile.TemporaryDirectory(prefix="research-copilot-paddle-") as temp_dir:
        root = Path(temp_dir)
        pdf_path = root / Path(filename).name
        pdf_path.write_bytes(content)
        output_dir = root / "output"
        output_dir.mkdir()
        pipeline = PPStructureV3()
        try:
            import pymupdf

            page_inputs = []
            with pymupdf.open(pdf_path) as document:
                total_pages = len(document)
                for page_number, page in enumerate(document, start=1):
                    image_path = root / f"page-{page_number:04d}.png"
                    page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False).save(image_path)
                    page_inputs.append(image_path)
        except (ImportError, RuntimeError, ValueError):
            page_inputs = [pdf_path]
            total_pages = len(PdfReader(str(pdf_path), strict=False).pages)
        markdown_parts = []
        ocr_pages = []
        for input_index, page_input in enumerate(page_inputs, start=1):
            if progress:
                progress(f"扫描版 OCR：第 {input_index}/{total_pages} 页", min(43, 15 + int((input_index - 1) / max(total_pages, 1) * 28)))
            for result_index, result in enumerate(pipeline.predict(str(page_input)), start=1):
                index = input_index if len(page_inputs) > 1 else result_index
                saver = getattr(result, "save_to_markdown", None)
                if callable(saver):
                    saver(save_path=str(output_dir))
                result_markdown = getattr(result, "markdown", None)
                if isinstance(result_markdown, str) and result_markdown.strip():
                    markdown_parts.append(result_markdown)
                    ocr_pages.append({"page": index, "text": result_markdown.strip()})
                if assets_dir is not None and page_input.suffix.lower() == ".png" and len(figures) < 20:
                    from PIL import Image

                    payload = getattr(result, "json", {})
                    payload = payload() if callable(payload) else payload
                    layout = payload.get("res", payload) if isinstance(payload, dict) else {}
                    blocks = layout.get("parsing_res_list") or []
                    captions = [block for block in blocks if block.get("block_label") == "figure_title"]
                    with Image.open(page_input) as page_image:
                        sx = page_image.width / max(float(layout.get("width") or page_image.width), 1)
                        sy = page_image.height / max(float(layout.get("height") or page_image.height), 1)
                        for block in blocks:
                            if block.get("block_label") not in {"image", "chart", "figure"} or len(figures) >= 20:
                                continue
                            bbox = block.get("block_bbox") or []
                            if len(bbox) != 4:
                                continue
                            try:
                                x0, y0, x1, y1 = [float(value) for value in bbox]
                            except (TypeError, ValueError):
                                continue
                            crop_box = (max(0, int(x0 * sx)), max(0, int(y0 * sy)),
                                        min(page_image.width, int(x1 * sx)), min(page_image.height, int(y1 * sy)))
                            if crop_box[2] - crop_box[0] < 100 or crop_box[3] - crop_box[1] < 80:
                                continue
                            following = [caption for caption in captions
                                         if (caption.get("block_bbox") or [0, 0, 0, 0])[1] >= y1
                                         and (caption.get("block_bbox") or [0, 0, 0, 0])[1] - y1 < 150]
                            caption_block = min(following, key=lambda item: item["block_bbox"][1]) if following else None
                            caption = str(caption_block.get("block_content") or "").strip() if caption_block else ""
                            # Keep the full caption inside the image asset. The layout detector may
                            # classify the diagram and its caption as separate blocks.
                            if caption_block:
                                caption_box = caption_block.get("block_bbox") or []
                                if len(caption_box) == 4:
                                    crop_box = (
                                        max(0, min(crop_box[0], int(float(caption_box[0]) * sx) - 8)),
                                        crop_box[1],
                                        min(page_image.width, max(crop_box[2], int(float(caption_box[2]) * sx) + 8)),
                                        min(page_image.height, max(crop_box[3], int(float(caption_box[3]) * sy) + 8)),
                                    )
                            asset_name = f"figure-p{index:03d}-{len(figures) + 1:03d}.png"
                            assets_dir.mkdir(parents=True, exist_ok=True)
                            page_image.crop(crop_box).save(assets_dir / asset_name)
                            asset_url = (f"{asset_url_prefix.rstrip('/')}/{quote(asset_name)}"
                                         if asset_url_prefix else asset_name)
                            figures.append({"id": f"figure-{len(figures) + 1:03d}",
                                            "kind": block.get("block_label"), "page": index,
                                            "bbox": [round(value / 2, 1) for value in crop_box],
                                            "asset_name": asset_name, "url": asset_url,
                                            "caption": caption, "label": f"Figure {len(figures) + 1}"})
            if progress:
                progress(f"扫描版 OCR：完成 {input_index}/{total_pages} 页", min(44, 15 + int(input_index / max(total_pages, 1) * 29)))
        if not markdown_parts:
            markdown_files = sorted(output_dir.rglob("*.md")) + sorted(output_dir.rglob("*.markdown"))
            markdown_parts = [path.read_text(encoding="utf-8") for path in markdown_files]
            ocr_pages = [
                {"page": index, "text": part.strip()}
                for index, part in enumerate(markdown_parts, start=1) if part.strip()
            ]
        if not markdown_parts:
            raise RuntimeError("PaddleOCR returned no Markdown output")
        markdown = "\n\n".join(markdown_parts)
        if assets_dir is not None:
            from .vision_analysis import analyze_figure

            assets_dir.mkdir(parents=True, exist_ok=True)
            for figure in figures:
                analysis = analyze_figure(assets_dir / figure["asset_name"], context=figure.get("caption") or "")
                figure["analysis"] = analysis
                markdown += f"\n\n![{figure.get('caption') or figure['label']}]({figure['url']})\n"
                if analysis.get("description"):
                    markdown += f"\n\n**Figure description (page {figure['page']}):** {analysis['description']}\n"
                    for page in ocr_pages:
                        if page["page"] == figure["page"]:
                            page["text"] += f"\n\nFigure description: {analysis['description']}"
            for image_path in sorted(output_dir.rglob("*")):
                if not image_path.is_file() or image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                    continue
                index = len(figures) + 1
                asset_name = f"figure-{index:03d}{image_path.suffix.lower()}"
                asset_path = assets_dir / asset_name
                asset_path.write_bytes(image_path.read_bytes())
                asset_url = (
                    f"{asset_url_prefix.rstrip('/')}/{quote(asset_name)}"
                    if asset_url_prefix else asset_name
                )
                source_paths = {
                    image_path.relative_to(output_dir).as_posix(),
                    image_path.name,
                }
                for source_path in source_paths:
                    markdown = markdown.replace(f"]({source_path})", f"]({asset_url})")
                    markdown = markdown.replace(f"](./{source_path})", f"]({asset_url})")
                markdown = re.sub(
                    rf"\]\((?:[^)]*/)?{re.escape(image_path.name)}\)",
                    f"]({asset_url})",
                    markdown,
                )
                analysis = analyze_figure(
                    asset_path,
                    context=_figure_context(markdown, asset_url),
                )
                figures.append({
                    "id": f"figure-{index:03d}",
                    "kind": "figure",
                    "asset_name": asset_name,
                    "url": asset_url,
                    "page": None,
                    "analysis": analysis,
                })
                if asset_url not in markdown:
                    markdown += f"\n\n![Extracted image {index}]({asset_url})\n"
                if analysis.get("description"):
                    markdown += f"\n\n**Figure description:** {analysis['description']}\n"
                elif analysis.get("status") == "pending_configuration":
                    markdown += (
                        "\n\n> Figure semantics are pending multimodal API configuration. "
                        "The extracted image is retained above.\n"
                    )
    parsed = _markdown_result("paddleocr", markdown)
    try:
        import io

        pdf_page_count = len(PdfReader(io.BytesIO(content), strict=False).pages)
    except Exception:
        pdf_page_count = 0
    if pdf_page_count and len(ocr_pages) == pdf_page_count:
        parsed.pages = ocr_pages
    else:
        parsed.warnings.append(
            "OCR Markdown page count does not match the PDF; paragraph-to-page links were not inferred."
        )
    parsed.figures = figures
    return parsed


def parse_pdf(
    filename: str,
    content: bytes,
    *,
    assets_dir: Path | None = None,
    asset_url_prefix: str | None = None,
    progress: Callable[[str, int], None] | None = None,
) -> ParsedPdf:
    settings = get_settings()
    warning = None
    parser = settings.pdf_parser.strip().lower()

    if parser in {"marker", "paddleocr"}:
        try:
            if parser == "marker":
                return _parse_marker(
                    filename, content, assets_dir=assets_dir, asset_url_prefix=asset_url_prefix
                )
            return _parse_paddleocr(
                filename, content, assets_dir=assets_dir, asset_url_prefix=asset_url_prefix,
                progress=progress,
            )
        except Exception as exc:
            warning = f"{parser} parser unavailable or failed ({exc.__class__.__name__}); pypdf fallback was used."
            return _parse_pypdf(filename, content, warning,
                                assets_dir=assets_dir, asset_url_prefix=asset_url_prefix,
                                progress=progress)

    if parser not in {"auto", "grobid", "pypdf"}:
        warning = f"Unknown PDF_PARSER '{settings.pdf_parser}'; pypdf fallback was used."

    if parser == "auto":
        if importlib.util.find_spec("marker") is not None:
            try:
                return _parse_marker(
                    filename, content, assets_dir=assets_dir, asset_url_prefix=asset_url_prefix
                )
            except Exception as exc:
                warning = (
                    f"Marker layout parsing failed ({exc.__class__.__name__}); "
                    "falling back to text extraction, so figure assets may be missing."
                )
        else:
            warning = (
                "Marker is not installed; using text extraction fallback. "
                "PyMuPDF caption-based figure extraction will be attempted."
            )

    if parser in {"auto", "grobid"} and settings.grobid_url:
        try:
            parsed = _parse_grobid(filename, content, settings.grobid_url)
            if len(parsed.full_text.strip()) >= 200:
                if warning:
                    parsed.warnings = (parsed.warnings or []) + [warning]
                return parsed
            warning = "GROBID returned little text; pypdf fallback was used."
        except Exception as exc:
            warning = f"GROBID parsing failed ({exc.__class__.__name__}); pypdf fallback was used."
    parsed = _parse_pypdf(filename, content, warning,
                          assets_dir=assets_dir, asset_url_prefix=asset_url_prefix,
                          progress=progress)
    if parser == "auto" and parsed.needs_ocr:
        # Whole-document OCR remains separate from the formula-only OCR stage.
        try:
            return _parse_paddleocr(
                filename, content, assets_dir=assets_dir, asset_url_prefix=asset_url_prefix,
                progress=progress,
            )
        except Exception as exc:
            parsed.warnings.append(f"PaddleOCR OCR unavailable or failed ({exc.__class__.__name__}: {str(exc)[:240]}).")
    return parsed
