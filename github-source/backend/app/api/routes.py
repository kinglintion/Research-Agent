from __future__ import annotations

import importlib.util
import json
import re
import time
from queue import Empty, Queue
from pathlib import Path
from threading import Thread

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from ..config import PROJECT_ROOT, get_settings
from ..schemas import (
    HealthComponent,
    HealthResponse,
    EvidenceQuestionRequest,
    HybridSearchRequest,
    ModelComponentResult,
    ModelDiagnosticRequest,
    ModelDiagnosticResponse,
    ParagraphTranslationRequest,
    SelectionTranslationRequest,
    ScholarlySearchRequest,
    UserProfileRequest,
    ArxivFollowRequest,
    RecommendationFeedbackRequest,
    ReadingPreferenceRequest,
    CitationGraphRequest,
    GraphAnalysisRequest,
    DraftAlignmentRequest,
    ResearchWritingRequest,
    CsvAnalysisRequest,
    CsvPlotRequest,
    MermaidRequest,
    ResearchAgentRequest,
)
from ..services.connections import check_mysql, check_qdrant, check_redis
from ..services.hybrid_retrieval import ModelJSONOutputError, answer_with_evidence, hybrid_search
from ..services.model_adapters import check_embedding, check_llm, check_reranker
from ..services.paper_ingestion import (
    get_paper_version,
    ingest_pdf,
    list_papers,
    delete_paper,
    preview_pdf_duplicates,
    preview_pdf_batch,
    reparse_paper_version,
    reindex_paper_version,
)
from ..services.paper_reader import pdf_file, reader_document, reader_page_layout, translate_paragraph, translate_selection, stream_selection_translation
from ..services.scholarly_discovery import acquire_external_paper, get_external_paper, search_external_papers
from ..services.research_workbench import analyze_csv, normalize_tabular_upload, build_citation_graph, generate_research_draft, mermaid_architecture, generate_graph_review, assess_future_work_coverage
from ..services.concept_graph import build_concept_graph
from ..services.draft_alignment import align_draft_to_library
from ..services.vision_analysis import vision_configuration
from ..services.job_queue import get_job, submit_pdf_job, submit_reparse_job
from ..services.research_agent import get_agent_task, submit_agent_task, resume_agent_task
from ..services.personalization import get_arxiv_follows, get_recommendations, save_profile, save_recommendation_feedback, save_arxiv_follow, remove_arxiv_follow, record_reading_preference


router = APIRouter()


@router.post("/agent/tasks", status_code=202)
def create_research_agent_task(request: ResearchAgentRequest) -> dict:
    try:
        return submit_agent_task(request.goal, request.paper_ids, task_type=request.task_type,
                                 version_id=request.version_id,
                                 csv_filename=request.csv_filename, csv_text=request.csv_text)
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"message": "Agent task could not be queued", "error_type": type(exc).__name__}) from exc


@router.get("/agent/tasks/{task_id}")
def research_agent_task(task_id: str) -> dict:
    state = get_agent_task(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Agent task not found or expired")
    return state


@router.post("/agent/tasks/{task_id}/resume", status_code=202)
def resume_research_agent_task(task_id: str) -> dict:
    try:
        state = resume_agent_task(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if state is None:
        raise HTTPException(status_code=404, detail="Agent task not found or expired")
    return state


@router.get("/agent/tasks/{task_id}/events")
def research_agent_events(task_id: str, after: int = Query(default=0, ge=0)) -> StreamingResponse:
    if get_agent_task(task_id) is None:
        raise HTTPException(status_code=404, detail="Agent task not found or expired")

    def events():
        previous = after
        while True:
            state = get_agent_task(task_id)
            if state is None:
                return
            for item in state["events"][previous:]:
                yield f"event: update\ndata: {json.dumps(item, ensure_ascii=False)}\n\n"
            previous = len(state["events"])
            if state["status"] in {"completed", "failed", "needs_review"}:
                return
            yield ": heartbeat\n\n"
            time.sleep(2)

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/tasks/papers", status_code=202)
def queue_pdf(file: UploadFile = File(...), duplicate_action: str = Form("append"),
              replace_version_id: str | None = Form(None)) -> dict:
    content = file.file.read(get_settings().max_pdf_bytes + 1)
    try:
        return submit_pdf_job(file.filename or "upload.pdf", content,
                              duplicate_action=duplicate_action, replace_version_id=replace_version_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"message": "PDF task could not be queued", "error_type": exc.__class__.__name__}) from exc


@router.post("/tasks/papers/batch", status_code=202)
def queue_pdf_batch(files: list[UploadFile] = File(...)) -> dict:
    if not files or len(files) > 20:
        raise HTTPException(status_code=422, detail="一次最多可提交 20 个 PDF")
    jobs, rejected = [], []
    for item in files:
        filename = item.filename or "upload.pdf"
        try:
            content = item.file.read(get_settings().max_pdf_bytes + 1)
            jobs.append(submit_pdf_job(filename, content))
        except ValueError as exc:
            rejected.append({"filename": filename, "error": str(exc)})
    return {"jobs": jobs, "rejected": rejected}


@router.post("/tasks/papers/{paper_id}/versions/{version_id}/reparse", status_code=202)
def queue_reparse(paper_id: str, version_id: str) -> dict:
    try:
        return submit_reparse_job(paper_id, version_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"message": "PDF reparse task could not be queued",
                                                 "error_type": exc.__class__.__name__}) from exc


@router.get("/tasks/{job_id}")
def pdf_task(job_id: str) -> dict:
    state = get_job(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="PDF task not found or expired")
    return state


@router.get("/tasks/{job_id}/events")
def pdf_task_events(job_id: str) -> StreamingResponse:
    if get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="PDF task not found or expired")

    def events():
        last = ""
        deadline = time.monotonic() + 3600
        while time.monotonic() < deadline:
            state = get_job(job_id)
            if state is None:
                yield 'event: error\ndata: {"message":"PDF task expired"}\n\n'
                return
            encoded = json.dumps(state, ensure_ascii=False)
            if encoded != last:
                yield f"event: progress\ndata: {encoded}\n\n"
                last = encoded
            if state["status"] in {"completed", "partial", "failed"}:
                return
            yield ": heartbeat\n\n"
            time.sleep(2)

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/qa/stream")
def ask_with_evidence_stream(request: EvidenceQuestionRequest) -> StreamingResponse:
    if request.version_id and not request.paper_id:
        raise HTTPException(status_code=422, detail="paper_id is required when version_id is provided")

    def events():
        updates: Queue[tuple[str, dict]] = Queue()

        def run():
            try:
                result = answer_with_evidence(request.question, top_k=request.top_k,
                    paper_id=request.paper_id, version_id=request.version_id,
                    progress=lambda stage: updates.put(("progress", {"stage": stage})))
                updates.put(("result", result))
            except Exception as exc:
                updates.put(("error", {"message": "证据问答失败", "error_type": exc.__class__.__name__}))

        Thread(target=run, daemon=True).start()
        while True:
            try:
                event, payload = updates.get(timeout=10)
            except Empty:
                yield ": heartbeat\n\n"
                continue
            yield f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            if event in {"result", "error"}:
                return

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/research/graph")
def research_graph(request: CitationGraphRequest) -> dict:
    try:
        return build_citation_graph(request.paper_ids, request.limit, request.citation_strength)
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"message": "Citation graph generation failed", "error_type": exc.__class__.__name__}) from exc


@router.post("/research/concept-graph")
def research_concept_graph(request: GraphAnalysisRequest) -> dict:
    try:
        return build_concept_graph(request.paper_ids, request.limit)
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"message": "Concept graph generation failed", "error_type": exc.__class__.__name__}) from exc


@router.post("/research/draft-alignment")
def research_draft_alignment(request: DraftAlignmentRequest) -> dict:
    try:
        return align_draft_to_library(request.draft_text, request.paper_ids, request.limit)
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"message": "Draft alignment failed", "error_type": exc.__class__.__name__}) from exc


@router.post("/research/graph-review")
def research_graph_review(request: GraphAnalysisRequest) -> dict:
    try:
        return generate_graph_review(request.paper_ids, request.limit)
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"message": "Graph literature review failed", "error_type": exc.__class__.__name__}) from exc


@router.post("/research/future-work")
def research_future_work(request: GraphAnalysisRequest) -> dict:
    try:
        return assess_future_work_coverage(request.paper_ids, request.limit)
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"message": "Future work evidence assessment failed", "error_type": exc.__class__.__name__}) from exc


@router.post("/research/write")
def research_write(request: ResearchWritingRequest) -> dict:
    try:
        return generate_research_draft(request.idea, request.max_sources)
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"message": "Research draft generation failed", "error_type": exc.__class__.__name__}) from exc


@router.post("/research/csv/normalize")
async def research_csv_normalize(file: UploadFile = File(...)) -> dict:
    try:
        return normalize_tabular_upload(file.filename or "table.csv", await file.read())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/research/csv")
def research_csv(request: CsvAnalysisRequest) -> dict:
    try:
        return analyze_csv(request.filename, request.csv_text, request.x_column, request.y_column,
                           request.chart_type, request.aggregation, request.radar_columns)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/research/csv/generate")
def research_csv_generate(request: CsvPlotRequest) -> dict:
    from ..services.csv_plot_agent import generate_csv_plot
    try:
        return generate_csv_plot(request.filename, request.csv_text, request.goal)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"message": "CSV analysis failed", "error_type": exc.__class__.__name__}) from exc


@router.post("/research/mermaid")
def research_mermaid(request: MermaidRequest) -> dict:
    return mermaid_architecture(request.idea)


@router.post("/discovery/search")
def discover_papers(request: ScholarlySearchRequest) -> dict:
    try:
        return search_external_papers(request.query, request.limit)
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"message": "Academic search failed", "error_type": exc.__class__.__name__}) from exc


@router.get("/discovery/{record_id}")
def external_paper(record_id: str) -> dict:
    try:
        result = get_external_paper(record_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"message": "External paper record unavailable", "error_type": exc.__class__.__name__}) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="External paper record not found")
    return result


@router.post("/discovery/{record_id}/acquire")
def acquire_paper(record_id: str) -> dict:
    try:
        result = acquire_external_paper(record_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"message": "Full-text acquisition failed", "error_type": exc.__class__.__name__}) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="External paper record not found")
    return result


@router.post("/search")
def search_papers(request: HybridSearchRequest) -> dict:
    if request.version_id and not request.paper_id:
        raise HTTPException(status_code=422, detail="paper_id is required when version_id is provided")
    try:
        return hybrid_search(
            request.query,
            top_k=request.top_k,
            paper_id=request.paper_id,
            version_id=request.version_id,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={"message": "Hybrid paper search failed", "error_type": exc.__class__.__name__},
        ) from exc


@router.post("/qa")
def ask_with_evidence(request: EvidenceQuestionRequest) -> dict:
    if request.version_id and not request.paper_id:
        raise HTTPException(status_code=422, detail="paper_id is required when version_id is provided")
    try:
        return answer_with_evidence(
            request.question,
            top_k=request.top_k,
            paper_id=request.paper_id,
            version_id=request.version_id,
        )
    except ModelJSONOutputError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "The language model did not return a complete JSON answer",
                "reason": exc.reason,
                "finish_reason": exc.finish_reason,
            },
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={"message": "Evidence-grounded answering failed", "error_type": exc.__class__.__name__},
        ) from exc


@router.get("/papers/assets/{source_sha256}/{asset_name}")
def paper_asset(source_sha256: str, asset_name: str) -> FileResponse:
    if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise HTTPException(status_code=404, detail="Paper asset not found")
    safe_name = Path(asset_name).name
    if not safe_name or safe_name != asset_name:
        raise HTTPException(status_code=404, detail="Paper asset not found")

    storage_dir = Path(get_settings().paper_storage_dir)
    if not storage_dir.is_absolute():
        storage_dir = PROJECT_ROOT / storage_dir
    asset_path = storage_dir / "assets" / source_sha256 / safe_name
    if not asset_path.is_file():
        raise HTTPException(status_code=404, detail="Paper asset not found")
    return FileResponse(asset_path)


@router.post("/papers", status_code=201)
def upload_paper(file: UploadFile = File(...)) -> dict:
    settings = get_settings()
    content = file.file.read(settings.max_pdf_bytes + 1)
    try:
        return ingest_pdf(file.filename or "upload.pdf", content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={"message": "Paper ingestion failed", "error_type": exc.__class__.__name__},
        ) from exc


@router.get("/papers")
def papers(limit: int = Query(default=50, ge=1, le=200)) -> dict:
    try:
        return {"papers": list_papers(limit)}
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={"message": "Paper list is unavailable", "error_type": exc.__class__.__name__},
        ) from exc


@router.post("/papers/duplicate-check")
def check_paper_duplicates(file: UploadFile = File(...)) -> dict:
    content = file.file.read(get_settings().max_pdf_bytes + 1)
    try:
        return preview_pdf_duplicates(file.filename or "upload.pdf", content)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail={
            "message": "PDF duplicate check failed", "error_type": exc.__class__.__name__,
        }) from exc


@router.post("/papers/duplicate-check/batch")
def check_paper_batch_duplicates(files: list[UploadFile] = File(...)) -> dict:
    if not files or len(files) > 20:
        raise HTTPException(status_code=422, detail="一次最多可检查 20 个 PDF")
    items = [(item.filename or "upload.pdf", item.file.read(get_settings().max_pdf_bytes + 1)) for item in files]
    try:
        return {"previews": preview_pdf_batch(items)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail={
            "message": "PDF batch duplicate check failed", "error_type": exc.__class__.__name__,
        }) from exc


@router.delete("/papers/{paper_id}")
def remove_paper(paper_id: str) -> dict:
    try:
        removed = delete_paper(paper_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail={
            "message": "Paper deletion failed", "error_type": exc.__class__.__name__,
        }) from exc
    if not removed:
        raise HTTPException(status_code=404, detail="Paper not found")
    return {"removed": True, "paper_id": paper_id}


@router.get("/papers/{paper_id}/versions/{version_id}")
def paper_version(paper_id: str, version_id: str) -> dict:
    try:
        result = get_paper_version(paper_id, version_id)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={"message": "Paper version is unavailable", "error_type": exc.__class__.__name__},
        ) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Paper version not found")
    return result


@router.get("/papers/{paper_id}/versions/{version_id}/pdf")
def paper_version_pdf(paper_id: str, version_id: str) -> FileResponse:
    located = pdf_file(paper_id, version_id)
    if located is None:
        raise HTTPException(status_code=404, detail="Original PDF not found")
    path, filename = located
    return FileResponse(path, media_type="application/pdf", filename=filename,
                        content_disposition_type="inline")


@router.get("/papers/{paper_id}/versions/{version_id}/reader")
def paper_reader(paper_id: str, version_id: str) -> dict:
    try:
        result = reader_document(paper_id, version_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"message": "Paper reader unavailable",
                                                      "error_type": exc.__class__.__name__}) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Paper version or PDF not found")
    return result


@router.get("/papers/{paper_id}/versions/{version_id}/reader/pages/{page_number}/layout")
def paper_reader_page_layout(paper_id: str, version_id: str, page_number: int) -> dict:
    try:
        result = reader_page_layout(paper_id, version_id, page_number)
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"message": "PDF page layout unavailable",
                                                      "error_type": exc.__class__.__name__}) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="PDF page layout not found")
    return result


@router.post("/papers/{paper_id}/versions/{version_id}/paragraphs/{paragraph_id}/translate")
def translate_paper_paragraph(
    paper_id: str, version_id: str, paragraph_id: str, request: ParagraphTranslationRequest
) -> dict:
    try:
        result = translate_paragraph(paper_id, version_id, paragraph_id, request.target_language)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"message": "Paragraph translation failed",
                                                      "error_type": exc.__class__.__name__}) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Paragraph not found")
    return result


@router.post("/papers/{paper_id}/versions/{version_id}/paragraphs/{paragraph_id}/translate-selection")
def translate_selected_paper_text(
    paper_id: str, version_id: str, paragraph_id: str, request: SelectionTranslationRequest
) -> dict:
    try:
        result = translate_selection(paper_id, version_id, paragraph_id, request.text, request.target_language)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"message": "Selected-text translation failed",
                                                      "error_type": exc.__class__.__name__}) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Paragraph or selected text not found")
    return result


@router.post("/papers/{paper_id}/versions/{version_id}/translate-selection/stream")
def stream_selected_pdf_text(
    paper_id: str, version_id: str, request: SelectionTranslationRequest
) -> StreamingResponse:
    try:
        events = stream_selection_translation(paper_id, version_id, request.page, request.text,
            request.context_before, request.context_after, request.target_language)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    def frames():
        try:
            for event, payload in events:
                yield f"event: {event}\ndata: {json.dumps({'translation': payload} if event == 'done' else {'text': payload}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'message': 'Selection translation failed', 'error_type': exc.__class__.__name__, 'detail': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(frames(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/recommendations/profile")
def recommendations_profile(request: UserProfileRequest) -> dict:
    return save_profile(request.user_id, request.interests)


@router.get("/recommendations")
def recommendations(user_id: str = Query(min_length=1, max_length=80), limit: int = Query(default=20, ge=1, le=50)) -> dict:
    return get_recommendations(user_id, limit)


@router.post("/recommendations/{record_id}/feedback")
def recommendation_feedback(record_id: str, request: RecommendationFeedbackRequest) -> dict:
    return save_recommendation_feedback(request.user_id, record_id, request.vote)


@router.post("/recommendations/reading")
def recommendation_reading(request: ReadingPreferenceRequest) -> dict:
    return record_reading_preference(request.user_id, request.paper_id, request.version_id)


@router.post("/arxiv/follows")
def follow_arxiv(request: ArxivFollowRequest) -> dict:
    return save_arxiv_follow(request.user_id, request.query)


@router.get("/arxiv/follows")
def arxiv_follows(user_id: str = Query(min_length=1, max_length=80)) -> dict:
    return {"follows": get_arxiv_follows(user_id)}


@router.delete("/arxiv/follows/{follow_id}")
def unfollow_arxiv(follow_id: int, user_id: str = Query(min_length=1, max_length=80)) -> dict:
    return {"removed": remove_arxiv_follow(user_id, follow_id)}


@router.post("/papers/{paper_id}/versions/{version_id}/reparse")
def reparse_version(paper_id: str, version_id: str) -> dict:
    try:
        result = reparse_paper_version(paper_id, version_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={"message": "Paper reparse failed", "error_type": exc.__class__.__name__},
        ) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Paper version not found")
    return result


@router.post("/papers/{paper_id}/versions/{version_id}/index")
def index_version(paper_id: str, version_id: str) -> dict:
    try:
        result = reindex_paper_version(paper_id, version_id)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={"message": "Qdrant indexing failed", "error_type": exc.__class__.__name__},
        ) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Paper version not found")
    return result


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    checks = {
        "mysql": check_mysql,
        "redis": check_redis,
        "qdrant": check_qdrant,
    }
    components: dict[str, HealthComponent] = {}
    for name, check in checks.items():
        status, detail = check()
        components[name] = HealthComponent(status=status, detail=detail)

    overall = "ok" if all(item.status == "ok" for item in components.values()) else "degraded"
    return HealthResponse(
        status=overall,
        service=settings.app_name,
        components=components,
    )


@router.get("/capabilities")
def capabilities() -> dict:
    settings = get_settings()
    return {
        "llm": {"provider": "openai-compatible", "model": settings.llm_model},
        "embedding": {
            "model": settings.embed_model,
            "dimension": settings.expected_embedding_dimension,
        },
        "reranker": {"model": settings.reranker_model},
        "vision": {
            "provider": "openai-compatible",
            "configured": bool(vision_configuration()),
            "model": vision_configuration().get("model"),
            "source": vision_configuration().get("source"),
        },
        "pdf_layout": {
            "preferred_parser": "marker when installed; otherwise configured fallback",
            "marker_installed": importlib.util.find_spec("marker") is not None,
            "position_aware_text": importlib.util.find_spec("pymupdf") is not None,
        },
        "storage": {"metadata": "mysql", "cache_and_jobs": "redis", "vectors": "qdrant"},
        "pdf_parsers": {
            "selected": settings.pdf_parser,
            "grobid": {"configured": bool(settings.grobid_url)},
            "marker": {"installed": importlib.util.find_spec("marker") is not None},
            "paddleocr": {"installed": importlib.util.find_spec("paddleocr") is not None},
            "formula_ocr": {
                "enabled": settings.formula_ocr_enabled,
                "model": settings.formula_ocr_model,
                "paddleocr_installed": importlib.util.find_spec("paddleocr") is not None,
                "kind": "local_formula_to_latex",
            },
        },
        "vector_index": {
            "provider": "qdrant",
            "collection": settings.qdrant_papers_collection,
            "embedding_dimension": settings.expected_embedding_dimension,
        },
    }


@router.post("/diagnostics/models", response_model=ModelDiagnosticResponse)
def diagnose_models(request: ModelDiagnosticRequest) -> ModelDiagnosticResponse:
    settings = get_settings()
    if not settings.enable_model_diagnostics:
        raise HTTPException(status_code=404, detail="Model diagnostics are disabled")

    checks = {
        "llm": lambda: check_llm(request.prompt),
        "embedding": lambda: check_embedding(request.query),
        "reranker": lambda: check_reranker(request.query, request.passages),
    }
    results: dict[str, ModelComponentResult] = {}
    for name, check in checks.items():
        try:
            results[name] = ModelComponentResult(**check())
        except Exception as exc:
            results[name] = ModelComponentResult(
                status="unavailable",
                model={
                    "llm": settings.llm_model,
                    "embedding": settings.embed_model,
                    "reranker": settings.reranker_model,
                }[name],
                details={"error_type": exc.__class__.__name__},
            )

    status = "ok" if all(item.status == "ok" for item in results.values()) else "degraded"
    return ModelDiagnosticResponse(status=status, components=results)
