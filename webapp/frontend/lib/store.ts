import { create } from "zustand";
import type {
  AnalyzeCodeResponse,
  AnalysisMode,
  DataMode,
  DiagnosisState,
  EpochMetric,
  FaultTaxonomyResponse,
  FullAnalysisResponse,
  PipelineStage,
} from "@/types/analysis";

interface DiagnosisActions {
  loadSession: (s: import("@/lib/sessions").SavedSession) => void;
  setCode: (code: string) => void;
  setModelName: (name: string) => void;
  setMode: (mode: AnalysisMode) => void;
  setCurrentStage: (stage: PipelineStage) => void;
  setCodeResult: (r: AnalyzeCodeResponse | null) => void;
  setFullResult: (r: FullAnalysisResponse | null) => void;
  setError: (e: string | null) => void;
  setWarnings: (w: string[]) => void;
  setTaxonomy: (t: FaultTaxonomyResponse | null) => void;
  // Training actions
  setDataMode: (mode: DataMode) => void;
  setUploadedFile: (file: File | null) => void;
  setTrainingConfig: (config: Partial<{ epochs: number; numSamples: number; dataMode: DataMode }>) => void;
  appendEpochMetric: (metric: EpochMetric) => void;
  setTrainingProgress: (p: { current: number; total: number } | null) => void;
  setIsDummyData: (v: boolean) => void;
  setAbortController: (ctrl: AbortController | null) => void;
  resetTraining: () => void;
  reset: () => void;
}

const PLACEHOLDER_CODE = `# Paste your Keras model code here
# Example (the motivating example from the DEFault paper):

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, Activation, MaxPooling2D,
    Flatten, Dense, Dropout
)

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), input_shape=(28, 28, 1)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64),
        Activation('relu'),
        Dropout(0.5),
        Dense(1),
        Activation('sigmoid'),   # ← DEFault will flag this
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )
    return model
`;

const initialState: DiagnosisState = {
  code: PLACEHOLDER_CODE,
  modelName: "my_model",
  mode: "idle",
  currentStage: 0,
  codeResult: null,
  fullResult: null,
  error: null,
  warnings: [],
  taxonomy: null,
  // Training state
  dataMode: "dummy",
  uploadedFile: null,
  trainingEpochs: 5,
  numSamples: 200,
  liveMetrics: [],
  trainingProgress: null,
  isDummyData: false,
  abortController: null,
};

export const useDiagnosisStore = create<DiagnosisState & DiagnosisActions>((set) => ({
  ...initialState,

  loadSession: (s) =>
    set({
      code:           s.code,
      modelName:      s.modelName,
      dataMode:       s.dataMode,
      trainingEpochs: s.trainingEpochs,
      numSamples:     s.numSamples,
      fullResult:     s.fullResult,
      codeResult:     s.codeResult,
      liveMetrics:    s.liveMetrics,
      isDummyData:    s.isDummyData,
      mode:           s.fullResult ? "done_training"
                    : s.codeResult ? "done_static"
                    : "idle",
      error: null,
      warnings: s.fullResult?.warnings ?? s.codeResult?.warnings ?? [],
      currentStage: 0,
    }),

  setCode: (code) => set({ code }),
  setModelName: (modelName) => set({ modelName }),
  setMode: (mode) => set({ mode }),
  setCurrentStage: (currentStage) => set({ currentStage }),
  setCodeResult: (codeResult) => set({ codeResult }),
  setFullResult: (fullResult) => set({ fullResult }),
  setError: (error) => set({ error }),
  setWarnings: (warnings) => set({ warnings }),
  setTaxonomy: (taxonomy) => set({ taxonomy }),

  setDataMode: (dataMode) => set({ dataMode }),
  setUploadedFile: (uploadedFile) => set({ uploadedFile }),
  setTrainingConfig: (config) =>
    set((s) => ({
      trainingEpochs: config.epochs      ?? s.trainingEpochs,
      numSamples:     config.numSamples  ?? s.numSamples,
      dataMode:       config.dataMode    ?? s.dataMode,
    })),
  appendEpochMetric: (metric) =>
    set((s) => ({
      liveMetrics:      [...s.liveMetrics, metric],
      trainingProgress: { current: metric.epoch, total: metric.total },
    })),
  setTrainingProgress: (trainingProgress) => set({ trainingProgress }),
  setIsDummyData: (isDummyData) => set({ isDummyData }),
  setAbortController: (abortController) => set({ abortController }),
  resetTraining: () =>
    set({
      liveMetrics:      [],
      trainingProgress: null,
      isDummyData:      false,
      abortController:  null,
    }),

  // Full reset preserves code but clears everything else
  reset: () =>
    set((s) => ({
      ...initialState,
      code:      s.code,
      modelName: s.modelName,
      taxonomy:  s.taxonomy,
      dataMode:  s.dataMode,
      trainingEpochs: s.trainingEpochs,
      numSamples:     s.numSamples,
    })),
}));
