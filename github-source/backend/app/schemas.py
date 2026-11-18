from __future__ import annotations

from pydantic import BaseModel, Field


class HealthComponent(BaseModel):
    status: str
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str
    service: str
    components: dict[str, HealthComponent]


class ModelDiagnosticRequest(BaseModel):
    prompt: str = "Reply with exactly this JSON object: {\"ok\":true}"
    query: str = "What evidence supports the stated research result?"
    passages: list[str] = Field(
        default_factory=lambda: [
            "The experiment reports the result in Table 2 and describes the setup in Section 4."
        ],
        max_length=8,
    )


class ModelComponentResult(BaseModel):
    status: str
    model: str | None = None
    latency_ms: float | None = None
    details: dict = Field(default_factory=dict)


class ModelDiagnosticResponse(BaseModel):
    status: str
    components: dict[str, ModelComponentResult]


class HybridSearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=4000)
    top_k: int = Field(default=6, ge=1, le=20)
    paper_id: str | None = None
    version_id: str | None = None


class EvidenceQuestionRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    top_k: int = Field(default=6, ge=1, le=12)
    paper_id: str | None = None
    version_id: str | None = None


class ParagraphTranslationRequest(BaseModel):
    target_language: str = Field(default="zh-CN", pattern="^(zh-CN|en)$")


class SelectionTranslationRequest(ParagraphTranslationRequest):
    text: str = Field(min_length=2, max_length=2000)
    page: int = Field(default=1, ge=1)
    context_before: str = Field(default="", max_length=1000)
    context_after: str = Field(default="", max_length=1000)


class UserProfileRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=80)
    interests: str = Field(default="", max_length=4000)


class ArxivFollowRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=80)
    query: str = Field(min_length=2, max_length=500)


class RecommendationFeedbackRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=80)
    vote: int = Field(ge=-1, le=1)


class ReadingPreferenceRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=80)
    paper_id: str = Field(min_length=1, max_length=36)
    version_id: str = Field(min_length=1, max_length=36)


class ScholarlySearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    limit: int = Field(default=8, ge=1, le=15)


class CitationGraphRequest(BaseModel):
    paper_ids: list[str] = Field(default_factory=list, max_length=20)
    limit: int = Field(default=20, ge=1, le=20)
    citation_strength: str = Field(default="all", pattern="^(all|medium|strong)$")


class GraphAnalysisRequest(BaseModel):
    paper_ids: list[str] = Field(default_factory=list, max_length=20)
    limit: int = Field(default=20, ge=1, le=20)


class DraftAlignmentRequest(BaseModel):
    draft_text: str = Field(min_length=10, max_length=12000)
    paper_ids: list[str] = Field(default_factory=list, max_length=20)
    limit: int = Field(default=12, ge=1, le=20)


class ResearchWritingRequest(BaseModel):
    idea: str = Field(min_length=10, max_length=5000)
    max_sources: int = Field(default=8, ge=1, le=12)


class CsvAnalysisRequest(BaseModel):
    filename: str = Field(default="data.csv", max_length=200)
    csv_text: str = Field(min_length=1, max_length=2_000_000)
    x_column: str | None = None
    y_column: str | None = None
    chart_type: str = Field(default="bar", pattern="^(bar|line|scatter|table|radar3d|stacked_bar)$")
    aggregation: str = Field(default="mean", pattern="^(mean|sum|median|count)$")
    radar_columns: list[str] = Field(default_factory=list, max_length=6)


class CsvPlotRequest(BaseModel):
    filename: str = Field(default="data.csv", max_length=200)
    csv_text: str = Field(min_length=1, max_length=2_000_000)
    goal: str = Field(min_length=2, max_length=12000)


class MermaidRequest(BaseModel):
    idea: str = Field(min_length=5, max_length=3000)


class ResearchAgentRequest(BaseModel):
    goal: str = Field(min_length=2, max_length=12000)
    paper_ids: list[str] = Field(default_factory=list, max_length=20)
    version_id: str | None = None
    task_type: str = Field(default="research", pattern="^(research|writing)$")
    csv_filename: str | None = Field(default=None, max_length=200)
    csv_text: str | None = Field(default=None, max_length=2_000_000)
