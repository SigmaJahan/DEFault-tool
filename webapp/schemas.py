"""
Pydantic v2 request/response schemas for the DEFault professional tool API.

These schemas are the single source of truth for TypeScript type generation
on the frontend and for FastAPI's automatic OpenAPI documentation.
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Request bodies ────────────────────────────────────────────────────────────

class AnalyzeCodeRequest(BaseModel):
    """User pastes raw Keras/TF model code for instant static analysis."""
    code: str = Field(..., description="Raw Python code containing a Keras model definition")
    model_name: str = Field("pasted_model", description="Optional name for the model")


class TrainingHistoryRequest(BaseModel):
    """
    User provides epoch-by-epoch training metrics for full 3-stage diagnosis.
    Optionally also provides model code to include Stage 3 static analysis.
    """
    loss: list[float] = Field(..., description="Training loss per epoch")
    val_loss: list[float] = Field(..., description="Validation loss per epoch")
    train_acc: list[float] = Field(..., description="Training accuracy per epoch (0-1 scale)")
    val_acc: list[float] = Field(..., description="Validation accuracy per epoch (0-1 scale)")
    code: Optional[str] = Field(None, description="Optional model code for Stage 3 static analysis")
    model_name: str = Field("pasted_model")


# ── Sub-objects ───────────────────────────────────────────────────────────────

class ShapFeature(BaseModel):
    """A single feature in the SHAP explanation waterfall."""
    feature: str
    shap_value: float = Field(..., description="Signed SHAP value: + pushes toward buggy, - toward correct")
    abs_shap: float
    feature_value: float = Field(..., description="The actual extracted numeric value of the feature")
    hint: str = Field(..., description="Human-readable guidance for fixing this issue")


class StaticAnalysisResult(BaseModel):
    """Stage 3 root cause analysis output based on model architecture features."""
    probability_buggy: float
    threshold: float
    predicted_buggy: bool
    top_features: list[ShapFeature]
    all_features: dict[str, float] = Field(default_factory=dict, description="All 31 extracted static features")
    base_value: float = Field(0.5, description="SHAP expected value E[f(x)] — waterfall chart baseline")


class StageOneResult(BaseModel):
    """Stage 1: Binary fault detection result."""
    probability: float
    threshold: float
    predicted_positive: bool


class CategoryResult(BaseModel):
    """A single fault category result from Stage 2."""
    name: str
    probability: float
    threshold: float
    predicted_positive: bool


class StageTwoResult(BaseModel):
    """Stage 2: All 7 category classifiers — not just flagged ones."""
    categories: list[CategoryResult]
    flagged: list[str] = Field(..., description="Names of categories that exceeded threshold")


class TrainingSummary(BaseModel):
    """Derived statistics from the user's training history."""
    epochs: int
    final_train_acc: float
    final_val_acc: float
    final_loss: Optional[float] = None
    final_val_loss: Optional[float] = None
    loss_oscillation: float = Field(..., description="Std dev of epoch-to-epoch loss changes")
    acc_gap: float = Field(..., description="val_acc - train_acc (negative = overfitting)")
    decrease_acc_count: int = Field(..., description="Epochs where train_acc dropped")
    increase_loss_count: int = Field(..., description="Epochs where train loss increased")
    available_dynamic_features: list[str] = Field(default_factory=list)
    missing_dynamic_features: list[str] = Field(default_factory=list)


# ── Response bodies ───────────────────────────────────────────────────────────

class AnalyzeCodeResponse(BaseModel):
    """Response for POST /api/analyze-code — static analysis only."""
    analysis_mode: str = Field("static_only")
    model_name: str
    stage3_static: StaticAnalysisResult
    all_features: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class FullAnalysisResponse(BaseModel):
    """
    Response for POST /api/analyze-history — full or partial 3-stage diagnosis.
    analysis_mode: "partial_dynamic" when code not provided (no stage3),
                   "full" when code also provided (all 3 stages).
    """
    analysis_mode: str = Field(..., description="'partial_dynamic' or 'full'")
    model_name: str
    stage1_detection: StageOneResult
    stage2_categories: StageTwoResult
    stage3_static: Optional[StaticAnalysisResult] = None
    training_summary: TrainingSummary
    warnings: list[str] = Field(default_factory=list)


# ── Fault taxonomy ────────────────────────────────────────────────────────────

class FaultTaxonomyNode(BaseModel):
    """A node in the DNN fault hierarchy tree (paper Fig. 3)."""
    id: str
    label: str
    description: str
    fault_category: Optional[str] = Field(None, description="Maps to a Stage 2 category name if applicable")
    children: list[FaultTaxonomyNode] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


FaultTaxonomyNode.model_rebuild()


class FaultTaxonomyResponse(BaseModel):
    root: FaultTaxonomyNode
