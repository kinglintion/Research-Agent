"""Redis-backed status for PDF jobs running in a bounded local worker pool."""

from __future__ import annotations

import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from ..config import PROJECT_ROOT, get_settings
from .connections import redis_client
from .paper_ingestion import ingest_pdf, reparse_paper_version


JOB_TTL_SECONDS = 86400
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="research-pdf")
logger = logging.getLogger(__name__)


def _key(job_id: str) -> str:
    return f"research:pdf-job:{job_id}"


def _save(job_id: str, state: dict) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    redis_client().set(_key(job_id), json.dumps(state, ensure_ascii=False), ex=JOB_TTL_SECONDS)


def get_job(job_id: str) -> dict | None:
    try:
        uuid.UUID(job_id)
    except ValueError:
        return None
    raw = redis_client().get(_key(job_id))
    return json.loads(raw) if raw else None


def submit_pdf_job(filename: str, content: bytes, *, duplicate_action: str = "append",
                   replace_version_id: str | None = None) -> dict:
    settings = get_settings()
    if not content.startswith(b"%PDF"):
        raise ValueError("The uploaded file is not a valid PDF")
    if len(content) > settings.max_pdf_bytes:
        raise ValueError(f"PDF exceeds the {settings.max_pdf_bytes // (1024 * 1024)} MB upload limit")
    if duplicate_action not in {"append", "replace"} or (duplicate_action == "replace" and not replace_version_id):
        raise ValueError("A valid duplicate action and replacement version are required")
    job_id = str(uuid.uuid4())
    directory = PROJECT_ROOT / ".data" / "jobs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{job_id}.pdf"
    path.write_bytes(content)
    state = {"job_id": job_id, "filename": Path(filename).name[:200], "status": "queued",
             "stage": "等待处理", "progress": 0, "duplicate_action": duplicate_action,
             "replace_version_id": replace_version_id,
             "created_at": datetime.now(timezone.utc).isoformat()}
    try:
        _save(job_id, state)
        _executor.submit(_run, job_id, path, state)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return state


def submit_reparse_job(paper_id: str, version_id: str) -> dict:
    uuid.UUID(paper_id)
    uuid.UUID(version_id)
    job_id = str(uuid.uuid4())
    state = {"job_id": job_id, "filename": "已入库 PDF 重新解析", "status": "queued",
             "stage": "等待处理", "progress": 0, "paper_id": paper_id, "version_id": version_id,
             "created_at": datetime.now(timezone.utc).isoformat()}
    _save(job_id, state)
    _executor.submit(_run_reparse, job_id, state)
    return state


def _finish(state: dict, result: dict) -> None:
    versions = result.get("versions") or []
    newest = versions[0] if versions else {}
    structured = result.get("structured") or {}
    vector_index = result.get("vector_index") or structured.get("vector_index") or {}
    formula_status = (structured.get("formula_extraction") or {}).get("status")
    figure_failed = any((figure.get("analysis") or {}).get("status") == "failed"
                        for figure in structured.get("figures") or [])
    partial = not result.get("duplicate") and (
        result.get("needs_ocr") or vector_index.get("status") == "failed"
        or formula_status in {"failed", "unavailable"} or figure_failed
    )
    state.update(status="partial" if partial else "completed",
                 stage="部分完成，需处理警告" if partial else "完成", progress=100, result={
        "paper_id": result.get("paper_id"), "version_id": result.get("version_id") or newest.get("version_id"),
        "duplicate": result.get("duplicate"), "duplicate_kind": result.get("duplicate_kind"),
        "duplicate_action": result.get("duplicate_action"), "replaced_version_id": result.get("replaced_version_id"),
        "version_linked": result.get("version_linked"), "parser": result.get("parser") or newest.get("parser"),
        "figures": len(structured.get("figures") or []),
        "formulas": (structured.get("formula_extraction") or {}).get("count", 0),
        "formula_status": formula_status, "needs_ocr": result.get("needs_ocr"),
        "vector_index": vector_index, "warnings": result.get("warnings") or [],
    })


def _run_reparse(job_id: str, state: dict) -> None:
    def progress(stage: str, percent: int) -> None:
        state.update(status="running", stage=stage, progress=percent)
        _save(job_id, state)

    try:
        progress("读取已入库原始 PDF", 5)
        result = reparse_paper_version(state["paper_id"], state["version_id"], progress=progress)
        if result is None:
            raise ValueError("The stored paper version was not found")
        _finish(state, result)
        _save(job_id, state)
    except Exception as exc:
        logger.exception("PDF reparse job %s failed at stage %s", job_id, state.get("stage"))
        state.update(status="failed", stage="重新解析失败", error_type=exc.__class__.__name__,
                     error_message=str(exc)[:300] if isinstance(exc, ValueError) else "处理失败，请根据任务 ID 查看后端日志")
        _save(job_id, state)


def _run(job_id: str, path: Path, state: dict) -> None:
    def progress(stage: str, percent: int) -> None:
        state.update(status="running", stage=stage, progress=percent)
        _save(job_id, state)

    try:
        progress("检查文件与已有版本", 5)
        result = ingest_pdf(state["filename"], path.read_bytes(), progress=progress,
                            duplicate_action=state.get("duplicate_action", "append"),
                            replace_version_id=state.get("replace_version_id"))
        _finish(state, result)
        _save(job_id, state)
    except Exception as exc:
        logger.exception("PDF job %s failed at stage %s", job_id, state.get("stage"))
        state.update(status="failed", stage="处理失败", error_type=exc.__class__.__name__,
                     error_message=str(exc)[:300] if isinstance(exc, ValueError) else "处理失败，请根据任务 ID 查看后端日志")
        _save(job_id, state)
    finally:
        path.unlink(missing_ok=True)
