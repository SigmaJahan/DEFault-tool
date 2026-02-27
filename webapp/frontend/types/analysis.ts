// TypeScript types mirroring webapp/schemas.py
// Keep in sync with backend Pydantic models.

export interface ShapFeature {
  feature: string;
  shap_value: number;   // SIGNED: + = pushes toward buggy, - = toward correct
  abs_shap: number;
  feature_value: number;
  hint: string;
}

export interface StaticAnalysisResult {
  probability_buggy: number;
  threshold: number;
  predicted_buggy: boolean;
  top_features: ShapFeature[];
  all_features: Record<string, number>;
  base_value: number;   // SHAP E[f(x)] — waterfall baseline
}

export interface StageOneResult {
  probability: number;
  threshold: number;
  predicted_positive: boolean;
}

export interface CategoryResult {
  name: string;
  probability: number;
  threshold: number;
  predicted_positive: boolean;
}

export interface StageTwoResult {
  categories: CategoryResult[];
  flagged: string[];
}

export interface TrainingSummary {
  epochs: number;
  final_train_acc: number;
  final_val_acc: number;
  final_loss: number | null;
  final_val_loss: number | null;
  loss_oscillation: number;
  acc_gap: number;
  decrease_acc_count: number;
  increase_loss_count: number;
  available_dynamic_features: string[];
  missing_dynamic_features: string[];
}

export interface AnalyzeCodeResponse {
  analysis_mode: "static_only";
  model_name: string;
  stage3_static: StaticAnalysisResult;
  all_features: Record<string, number>;
  warnings: string[];
}

export interface FullAnalysisResponse {
  analysis_mode: "partial_dynamic" | "full" | "full_training";
  model_name: string;
  stage1_detection: StageOneResult;
  stage2_categories: StageTwoResult;
  stage3_static: StaticAnalysisResult | null;
  training_summary: TrainingSummary;
  warnings: string[];
}

export interface FaultTaxonomyNode {
  id: string;
  label: string;
  description: string;
  fault_category?: string;
  children: FaultTaxonomyNode[];
}

export interface FaultTaxonomyResponse {
  root: FaultTaxonomyNode;
}

// Frontend-only types
export type AnalysisMode =
  | "idle"
  | "analyzing_static"
  | "analyzing_full"
  | "training"
  | "done_static"
  | "done_full"
  | "done_training"
  | "error";

export type DataMode = "dummy" | "uploaded";

export type PipelineStage = 0 | 1 | 2 | 3;  // 0=idle, 1=detection, 2=categorization, 3=rca

export interface TrainingHistory {
  loss: number[];
  val_loss: number[];
  train_acc: number[];
  val_acc: number[];
}

export interface EpochMetric {
  epoch: number;
  total: number;
  loss: number;
  val_loss: number;
  acc: number;
  val_acc: number;
  elapsed_ms?: number;
}

export type TrainEvent =
  | ({ type: "epoch" } & EpochMetric)
  | { type: "training_done"; epochs_completed: number }
  | { type: "analyzing"; stage: number }
  | { type: "complete"; result: FullAnalysisResponse; data_mode: DataMode }
  | { type: "error"; message: string; traceback?: string };

export interface DiagnosisState {
  code: string;
  modelName: string;
  mode: AnalysisMode;
  currentStage: PipelineStage;
  codeResult: AnalyzeCodeResponse | null;
  fullResult: FullAnalysisResponse | null;
  error: string | null;
  warnings: string[];
  taxonomy: FaultTaxonomyResponse | null;
  // Training state
  dataMode: DataMode;
  uploadedFile: File | null;
  trainingEpochs: number;
  numSamples: number;
  liveMetrics: EpochMetric[];
  trainingProgress: { current: number; total: number } | null;
  isDummyData: boolean;
  abortController: AbortController | null;
}

// Category display metadata
export const CATEGORY_META: Record<string, { label: string; color: string; icon: string; description: string }> = {
  activation:     { label: "Activation",     color: "#f0883e", icon: "⚡", description: "Wrong activation function for layer or task" },
  layer:          { label: "Layer",          color: "#79c0ff", icon: "🧱", description: "Incorrect layer type, config, or missing layers" },
  hyperparameter: { label: "Hyperparameter", color: "#d2a8ff", icon: "🎛", description: "Incorrect learning rate, batch size, or epochs" },
  loss:           { label: "Loss Function",  color: "#ffa657", icon: "📉", description: "Inappropriate loss function for the task" },
  optimization:   { label: "Optimization",   color: "#56d364", icon: "🔧", description: "Wrong optimizer or misconfigured optimization" },
  regularizer:    { label: "Regularization", color: "#f778ba", icon: "🛡", description: "Incorrect regularization parameters" },
  weights:        { label: "Weights",        color: "#a5d6ff", icon: "⚖️", description: "Poor or incorrect weight initialization" },
};
