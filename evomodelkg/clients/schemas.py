from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class CandidateRelation(BaseModel):
    relation_type: str
    source_id: str
    target_id: str
    confidence: float = 0.5
    properties: dict = Field(default_factory=dict)
    evidence: str = ""
    source: str = "readme"
    evidence_span: str = ""
    normalization_note: str = ""

class CandidateAttribute(BaseModel):
    entity_type: str
    entity_id: str
    attribute: str
    value: str
    confidence: float = 0.5
    evidence: str = ""
    source: str = "readme"
    evidence_span: str = ""
    normalization_note: str = ""


class ReadmeExtractionResult(BaseModel):
    relations: list[CandidateRelation] = Field(default_factory=list)
    attributes: list[CandidateAttribute] = Field(default_factory=list)
    summary: str = ""


# --- v2 agent outputs ---


class TemplateGateResult(BaseModel):
    is_template_readme: bool = False
    template_score: float = 0.0
    recommended_path: Literal["skip", "light", "full"] = "full"
    reason: str = ""


class ModeRouterResult(BaseModel):
    active_schema_id: Literal["model_schema", "dataset_schema"]
    relation_whitelist: list[str] = Field(default_factory=list)
    attribute_whitelist: list[str] = Field(default_factory=list)


class MergeCandidateRelation(CandidateRelation):
    """Candidate relation with stable id for conflict resolution."""

    temp_id: str = ""
    origin: Literal["readme", "baseline"] = "readme"


class MergeCandidateAttribute(CandidateAttribute):
    temp_id: str = ""
    origin: Literal["readme", "baseline"] = "readme"


class DetectedConflict(BaseModel):
    conflict_id: str
    conflict_type: str
    description: str
    relation_temp_ids: list[str] = Field(default_factory=list)
    attribute_temp_ids: list[str] = Field(default_factory=list)
    participants: list[dict[str, Any]] = Field(default_factory=list)


class InitialMergeResult(BaseModel):
    normalized_relations: list[MergeCandidateRelation] = Field(default_factory=list)
    normalized_attributes: list[MergeCandidateAttribute] = Field(default_factory=list)
    conflicts: list[DetectedConflict] = Field(default_factory=list)
    unmapped: list[dict[str, Any]] = Field(default_factory=list)


class ConflictResolution(BaseModel):
    conflict_id: str
    action: Literal[
        "keep_relation",
        "drop_relation",
        "keep_attribute",
        "drop_attribute",
        "prefer_baseline_relation",
        "prefer_baseline_attribute",
        "merge_relation_properties",
    ]
    keep_temp_id: str = ""
    drop_temp_ids: list[str] = Field(default_factory=list)
    rationale: str = ""


class ConflictFixerOutput(BaseModel):
    resolutions: list[ConflictResolution] = Field(default_factory=list)
    summary: str = ""


class FinalMergeResult(BaseModel):
    final_relations: list[CandidateRelation] = Field(default_factory=list)
    final_attributes: list[CandidateAttribute] = Field(default_factory=list)
    merge_audit: str = ""


class OverlapScorecard(BaseModel):
    accuracy_overlap: Optional[float] = None
    correct_count: int = 0
    predicted_count: int = 0
    notes: str = ""


class CompletenessReport(BaseModel):
    baseline_relation_count: int = 0
    baseline_attribute_count: int = 0
    candidate_relation_count: int = 0
    candidate_attribute_count: int = 0
    total_relation_count: int = 0
    total_attribute_count: int = 0
    baseline_coverage: dict[str, float] = Field(default_factory=dict)
    merged_coverage: dict[str, float] = Field(default_factory=dict)
    completeness_score: float = 0.0
    missing_dimensions: list[str] = Field(default_factory=list)


class EnrichmentOutput(BaseModel):
    resource_id: str
    resource_type: str
    template_gate: Optional[TemplateGateResult] = None
    mode_router: Optional[ModeRouterResult] = None
    initial_merge: Optional[InitialMergeResult] = None
    conflict_fix: Optional[ConflictFixerOutput] = None
    final_merge: Optional[FinalMergeResult] = None
    overlap_scorecard: Optional[OverlapScorecard] = None
    readme_result: ReadmeExtractionResult
    completeness_report: Optional[CompletenessReport] = None
    extra_attributes: list[CandidateAttribute] = Field(default_factory=list)
