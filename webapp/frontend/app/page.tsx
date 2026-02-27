"use client";

import { useEffect, useCallback, useState } from "react";
import {
  AlertCircle, AlertTriangle, Zap, RefreshCw, Play,
  Code2, GitBranch, BarChart2,
} from "lucide-react";

import { useDiagnosisStore } from "@/lib/store";
import { analyzeCode, getFaultTaxonomy, streamTraining } from "@/lib/api";
import { saveDraft } from "@/lib/sessions";
import { SessionManager } from "@/components/SessionManager";
import type { SavedSession } from "@/lib/sessions";

import { CodeEditor } from "@/components/CodeEditor";
import { DatasetUploader } from "@/components/DatasetUploader";
import { LiveTrainingChart } from "@/components/LiveTrainingChart";
import { PipelineVisualizer } from "@/components/PipelineVisualizer";
import { DetectionGauge } from "@/components/DetectionGauge";
import { FaultCategoryChart } from "@/components/FaultCategoryChart";
import { SHAPWaterfallChart } from "@/components/SHAPWaterfallChart";
import { FaultTaxonomyTree } from "@/components/FaultTaxonomyTree";
import { InsightsPanel } from "@/components/InsightsPanel";

// Mobile panel tab IDs
type MobileTab = "editor" | "pipeline" | "results";

export default function Home() {
  const store = useDiagnosisStore();
  const {
    code, modelName, mode, currentStage,
    codeResult, fullResult, error, warnings, taxonomy,
    dataMode, uploadedFile, trainingEpochs, numSamples,
    liveMetrics, isDummyData, abortController,
    setCode, setModelName, setMode,
    setCurrentStage, setCodeResult, setFullResult,
    setError, setWarnings, setTaxonomy,
    setDataMode, setUploadedFile, setTrainingConfig,
    appendEpochMetric, setIsDummyData, setAbortController,
    resetTraining, loadSession,
  } = store;

  const [mobileTab, setMobileTab] = useState<MobileTab>("editor");

  const isRunning = mode === "analyzing_static" || mode === "analyzing_full" || mode === "training";
  const hasDone = mode === "done_static" || mode === "done_full" || mode === "done_training";

  const staticResult = fullResult?.stage3_static ?? codeResult?.stage3_static ?? null;
  const stage1 = fullResult?.stage1_detection ?? null;
  const stage2 = fullResult?.stage2_categories ?? null;

  // Auto-save draft whenever results arrive
  useEffect(() => {
    if (hasDone) {
      saveDraft({ code, modelName, dataMode, trainingEpochs, numSamples, fullResult, codeResult, liveMetrics, isDummyData });
    }
  }, [hasDone]); // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-switch to results on mobile once analysis completes
  useEffect(() => {
    if (hasDone) setMobileTab("results");
  }, [hasDone]);

  // Auto-switch to pipeline on mobile while training
  useEffect(() => {
    if (mode === "training") setMobileTab("pipeline");
  }, [mode]);

  // Load fault taxonomy once on mount
  useEffect(() => {
    getFaultTaxonomy()
      .then(setTaxonomy)
      .catch(() => {/* non-critical */});
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Abort SSE stream on unmount
  useEffect(() => {
    return () => { abortController?.abort(); };
  }, [abortController]);

  // ── Static-only analysis ─────────────────────────────────────────────────
  const runStaticAnalysis = useCallback(async () => {
    if (!code.trim() || isRunning) return;
    setMode("analyzing_static");
    setCurrentStage(3);
    setError(null);
    setCodeResult(null);
    setFullResult(null);
    setWarnings([]);
    try {
      const result = await analyzeCode(code.trim(), modelName);
      setCodeResult(result);
      setWarnings(result.warnings ?? []);
      setMode("done_static");
    } catch (e: unknown) {
      setMode("error");
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setCurrentStage(0);
    }
  }, [code, modelName, isRunning, setMode, setCurrentStage, setError, setCodeResult, setFullResult, setWarnings]);

  // ── Train & Diagnose ─────────────────────────────────────────────────────
  const runTrainingAnalysis = useCallback(async () => {
    if (!code.trim() || isRunning) return;
    if (dataMode === "uploaded" && !uploadedFile) {
      setError("Please upload a dataset file first.");
      return;
    }
    const ctrl = new AbortController();
    setAbortController(ctrl);
    resetTraining();
    setMode("training");
    setCurrentStage(1);
    setError(null);
    setCodeResult(null);
    setFullResult(null);
    setWarnings([]);
    setIsDummyData(dataMode === "dummy");
    try {
      for await (const event of streamTraining(
        { code: code.trim(), modelName, dataMode, epochs: trainingEpochs, numSamples, datasetFile: uploadedFile },
        ctrl.signal,
      )) {
        if (event.type === "epoch") {
          appendEpochMetric(event);
        } else if (event.type === "analyzing") {
          setCurrentStage(event.stage as 1 | 2 | 3);
        } else if (event.type === "complete") {
          setFullResult(event.result);
          setIsDummyData(event.data_mode === "dummy");
          setWarnings(event.result.warnings ?? []);
          setMode("done_training");
        } else if (event.type === "error") {
          setError(event.message);
          setMode("error");
        }
      }
    } catch (e: unknown) {
      if ((e as Error).name !== "AbortError") {
        setMode("error");
        setError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setCurrentStage(0);
      setAbortController(null);
    }
  }, [
    code, modelName, isRunning, dataMode, uploadedFile, trainingEpochs, numSamples,
    setAbortController, resetTraining, setMode, setCurrentStage, setError,
    setCodeResult, setFullResult, setWarnings, setIsDummyData, appendEpochMetric,
  ]);

  // ── Cmd/Ctrl+Enter → Check Model ────────────────────────────────────────
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        runStaticAnalysis();
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [runStaticAnalysis]);

  return (
    /* Root: full height on desktop, scrollable on mobile */
    <div
      className="min-h-screen md:h-screen flex flex-col md:overflow-hidden"
      style={{ background: "var(--bg-base)", color: "var(--text-primary)" }}
    >
      {/* ── Top bar ───────────────────────────────────────────────────────── */}
      <header
        role="banner"
        className="flex items-center gap-3 px-4 md:px-5 py-2.5 flex-shrink-0"
        style={{
          background: "var(--bg-card)",
          borderBottom: "1px solid var(--border)",
        }}
      >
        <Zap size={15} aria-hidden="true" style={{ color: "var(--accent)" }} />

        <div className="flex flex-col leading-tight">
          <span className="font-semibold text-sm tracking-tight">
            DEFault
          </span>
          <span
            className="text-[9px] uppercase tracking-widest hidden sm:block"
            style={{ color: "var(--text-muted)" }}
          >
            Detection &amp; Explain Fault
          </span>
        </div>

        <div
          className="ml-auto flex items-center gap-2 md:gap-3 text-[10px]"
          style={{ color: "var(--text-muted)" }}
        >
          <span className="hidden lg:flex items-center gap-1" aria-label="Keyboard shortcut: Command Enter to check model">
            <kbd
              className="px-1 py-0.5 rounded font-mono"
              style={{ background: "var(--bg-overlay)", border: "1px solid var(--border)" }}
            >⌘</kbd>
            <span aria-hidden="true">+</span>
            <kbd
              className="px-1 py-0.5 rounded font-mono"
              style={{ background: "var(--bg-overlay)", border: "1px solid var(--border)" }}
            >↵</kbd>
            <span>check model</span>
          </span>

          {/* Session manager */}
          <SessionManager
            currentState={{ code, modelName, dataMode, trainingEpochs, numSamples, fullResult, codeResult, liveMetrics, isDummyData }}
            onLoad={(s: SavedSession) => { loadSession(s); setMobileTab("results"); }}
            isRunning={isRunning}
            hasResults={hasDone}
          />

          <label className="sr-only" htmlFor="model-name-input">Model name</label>
          <input
            id="model-name-input"
            className="font-mono text-[10px] px-2 py-1 rounded"
            style={{
              background: "var(--bg-input)",
              border: "1px solid var(--border)",
              color: "var(--text-secondary)",
              width: 110,
              outline: "none",
            }}
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            placeholder="model name"
            aria-label="Model name"
          />
        </div>
      </header>

      {/* ── Mobile tab bar ──────────────────────────────────────────────── */}
      <nav
        role="tablist"
        aria-label="Panel navigation"
        className="flex md:hidden border-b flex-shrink-0"
        style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}
      >
        {(
          [
            { id: "editor", label: "Editor", icon: <Code2 size={13} aria-hidden="true" /> },
            { id: "pipeline", label: "Pipeline", icon: <GitBranch size={13} aria-hidden="true" /> },
            { id: "results", label: "Results", icon: <BarChart2 size={13} aria-hidden="true" /> },
          ] as { id: MobileTab; label: string; icon: React.ReactNode }[]
        ).map(({ id, label, icon }) => (
          <button
            key={id}
            role="tab"
            aria-selected={mobileTab === id}
            aria-controls={`panel-${id}`}
            onClick={() => setMobileTab(id)}
            className="flex-1 flex items-center justify-center gap-1.5 py-2 text-xs font-medium transition-colors"
            style={{
              color: mobileTab === id ? "var(--accent)" : "var(--text-muted)",
              borderBottom: mobileTab === id ? "2px solid var(--accent)" : "2px solid transparent",
              background: "transparent",
            }}
          >
            {icon}
            {label}
            {id === "results" && (hasDone || mode === "error") && (
              <span
                className="w-1.5 h-1.5 rounded-full"
                style={{ background: "var(--accent)" }}
                aria-label="Results available"
              />
            )}
          </button>
        ))}
      </nav>

      {/* ── 3-column layout (stacked on mobile) ──────────────────────────── */}
      <div className="flex flex-col md:flex-row flex-1 overflow-y-auto md:overflow-hidden">

        {/* ── LEFT: Editor ────────────────────────────────────────────── */}
        <aside
          id="panel-editor"
          role="region"
          aria-label="Model code editor and training configuration"
          className={`
            flex flex-col flex-shrink-0 overflow-y-auto
            w-full md:w-[29%] md:min-w-[270px]
            border-b md:border-b-0 md:border-r
            ${mobileTab !== "editor" ? "hidden md:flex" : "flex"}
          `}
          style={{
            borderColor: "var(--border)",
            background: "var(--bg-card)",
          }}
        >
          <div className="flex flex-col gap-3 p-4 h-full">
            {/* Code editor */}
            <div className="flex flex-col" style={{ flex: "1 1 280px", minHeight: 0 }}>
              <div
                className="text-[9px] uppercase tracking-widest mb-1.5"
                style={{ color: "var(--text-muted)" }}
                id="code-editor-label"
              >
                Model Code: Keras / TensorFlow
              </div>
              <div
                className="rounded-lg overflow-hidden"
                style={{ border: "1px solid var(--border)", flex: 1, minHeight: 200 }}
                role="group"
                aria-labelledby="code-editor-label"
              >
                <CodeEditor
                  value={code}
                  onChange={setCode}
                  disabled={isRunning}
                />
              </div>
            </div>

            {/* Training configuration */}
            <div className="flex-shrink-0">
              <DatasetUploader
                dataMode={dataMode}
                uploadedFile={uploadedFile}
                epochs={trainingEpochs}
                numSamples={numSamples}
                disabled={isRunning}
                onDataModeChange={setDataMode}
                onFileChange={setUploadedFile}
                onEpochsChange={(n) => setTrainingConfig({ epochs: n })}
                onNumSamplesChange={(n) => setTrainingConfig({ numSamples: n })}
              />
            </div>

            {/* Error display */}
            {error && (
              <div
                role="alert"
                aria-live="assertive"
                className="flex items-start gap-2 text-[11px] px-3 py-2 rounded-md flex-shrink-0"
                style={{
                  background: "var(--fault-red-bg)",
                  border: "1px solid var(--fault-red)",
                  color: "var(--fault-red)",
                }}
              >
                <AlertCircle size={12} aria-hidden="true" className="flex-shrink-0 mt-0.5" />
                <span className="break-words">{error}</span>
              </div>
            )}

            {/* Action buttons */}
            <div className="flex gap-2 flex-shrink-0 pt-1" role="group" aria-label="Analysis actions">
              {/* Check Model */}
              <button
                className="flex-1 py-2.5 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-all"
                style={{
                  background: "var(--bg-overlay)",
                  color: isRunning ? "var(--text-muted)" : "var(--text-secondary)",
                  cursor: isRunning ? "not-allowed" : "pointer",
                  border: "1px solid var(--border)",
                  opacity: isRunning ? 0.6 : 1,
                }}
                onClick={runStaticAnalysis}
                disabled={isRunning}
                aria-disabled={isRunning}
                aria-label="Check model: instant static analysis, no training required"
              >
                <Zap size={12} aria-hidden="true" />
                Check Model
              </button>

              {/* Train & Diagnose */}
              <button
                className="flex-1 py-2.5 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-all"
                style={{
                  background: isRunning ? "var(--bg-overlay)" : "var(--accent)",
                  color: isRunning ? "var(--text-muted)" : "#fff",
                  cursor: isRunning ? "not-allowed" : "pointer",
                  opacity: isRunning && mode !== "training" ? 0.6 : 1,
                }}
                onClick={runTrainingAnalysis}
                disabled={isRunning}
                aria-disabled={isRunning}
                aria-busy={mode === "training"}
                aria-label={
                  mode === "training"
                    ? "Training in progress…"
                    : "Train and diagnose: real model training with full 3-stage fault diagnosis"
                }
              >
                {isRunning && mode === "training" ? (
                  <><RefreshCw size={12} className="animate-spin" aria-hidden="true" />Training…</>
                ) : isRunning ? (
                  <><RefreshCw size={12} className="animate-spin" aria-hidden="true" />Analyzing…</>
                ) : (
                  <><Play size={12} aria-hidden="true" />Train &amp; Diagnose</>
                )}
              </button>

              {hasDone && (
                <button
                  className="py-2.5 px-3 rounded-lg text-xs transition-all"
                  style={{
                    background: "var(--bg-overlay)",
                    border: "1px solid var(--border)",
                    color: "var(--text-muted)",
                    cursor: "pointer",
                  }}
                  onClick={() => { store.reset(); setMobileTab("editor"); }}
                  aria-label="Reset: clear all results and start over"
                >
                  Reset
                </button>
              )}
            </div>
          </div>
        </aside>

        {/* ── CENTER: Pipeline + live chart + taxonomy ──────────────────── */}
        <main
          id="panel-pipeline"
          role="region"
          aria-label="Analysis pipeline and training charts"
          aria-live="polite"
          aria-busy={isRunning}
          className={`
            flex flex-col overflow-y-auto flex-shrink-0
            w-full md:w-[35%] md:min-w-[270px]
            ${mobileTab !== "pipeline" ? "hidden md:flex" : "flex"}
          `}
        >
          <div className="p-4 space-y-3">
            <Panel>
              <PipelineVisualizer
                currentStage={currentStage}
                mode={mode}
                detectionProb={stage1?.probability}
                detectionPositive={stage1?.predicted_positive}
                flaggedCategories={stage2?.flagged}
                topRootCause={staticResult?.top_features[0]?.feature}
              />
            </Panel>

            {/* Live training chart */}
            {(mode === "training" || mode === "done_training") && liveMetrics.length > 0 && (
              <Panel>
                <LiveTrainingChart
                  epochs={liveMetrics}
                  totalEpochs={trainingEpochs}
                  isRunning={mode === "training"}
                  isDummyData={isDummyData}
                />
              </Panel>
            )}

            {/* Detection gauge */}
            {stage1 && (
              <Panel>
                <div
                  className="text-[9px] uppercase tracking-widest mb-3"
                  style={{ color: "var(--text-muted)" }}
                  id="detection-heading"
                >
                  Stage 1: Fault Detection
                </div>
                <DetectionGauge
                  probability={stage1.probability}
                  threshold={stage1.threshold}
                  isFault={stage1.predicted_positive}
                />
              </Panel>
            )}

            {/* Category chart */}
            {stage2 && (
              <Panel>
                <div
                  className="text-[9px] uppercase tracking-widest mb-3"
                  style={{ color: "var(--text-muted)" }}
                >
                  Stage 2: Fault Categories
                </div>
                <FaultCategoryChart categories={stage2.categories} />
              </Panel>
            )}

            {/* Fault taxonomy tree */}
            {taxonomy && (
              <Panel>
                <FaultTaxonomyTree
                  root={taxonomy.root}
                  flaggedCategories={stage2?.flagged ?? []}
                  topShapFeatures={staticResult?.top_features.map((f) => f.feature) ?? []}
                />
              </Panel>
            )}
          </div>
        </main>

        {/* ── RIGHT: SHAP + Insights ────────────────────────────────────── */}
        <aside
          id="panel-results"
          role="region"
          aria-label="SHAP explanations and fault insights"
          aria-live="polite"
          className={`
            flex flex-col overflow-y-auto
            w-full md:flex-1 md:min-w-[230px]
            ${mobileTab !== "results" ? "hidden md:flex" : "flex"}
          `}
        >
          <div className="p-4 space-y-3">
            {/* Dummy-data warning */}
            {isDummyData && mode === "done_training" && (
              <div
                role="note"
                className="flex items-start gap-2 text-[11px] px-3 py-2.5 rounded-md"
                style={{
                  background: "var(--warn-yellow-bg)",
                  border: "1px solid var(--warn-yellow)",
                  color: "var(--warn-yellow)",
                }}
              >
                <AlertTriangle size={12} aria-hidden="true" className="flex-shrink-0 mt-0.5" />
                <span>
                  Results based on <strong>synthetic dummy data</strong>. Upload your dataset for
                  diagnosis with real training dynamics.
                </span>
              </div>
            )}

            {/* SHAP waterfall */}
            {staticResult && staticResult.top_features.length > 0 && (
              <Panel>
                <SHAPWaterfallChart
                  features={staticResult.top_features}
                  baseValue={staticResult.base_value}
                  predictedProbability={staticResult.probability_buggy}
                />
              </Panel>
            )}

            {/* Insights */}
            {(hasDone || mode === "error") && (
              <Panel>
                <InsightsPanel
                  stage3={staticResult}
                  stage2={stage2}
                  warnings={warnings}
                />
              </Panel>
            )}

            {/* Loading skeletons */}
            {isRunning && <AnalyzingSkeleton />}

            {/* Empty state */}
            {!hasDone && !isRunning && mode !== "error" && <EmptyState />}
          </div>
        </aside>
      </div>
    </div>
  );
}

// ── Panel card ────────────────────────────────────────────────────────────

function Panel({ children }: { children: React.ReactNode }) {
  return (
    <div
      className="rounded-lg p-4 animate-fade-in"
      style={{ background: "var(--bg-card)", border: "1px solid var(--border)" }}
    >
      {children}
    </div>
  );
}

// ── Empty state ───────────────────────────────────────────────────────────

function EmptyState() {
  return (
    <div
      className="flex flex-col items-center justify-center py-16 px-4 text-center"
      role="status"
      aria-label="Ready to diagnose: no results yet"
    >
      <div
        className="w-12 h-12 rounded-full flex items-center justify-center mb-4"
        style={{ background: "var(--bg-card)", border: "1px solid var(--border)" }}
        aria-hidden="true"
      >
        <Zap size={20} style={{ color: "var(--text-muted)" }} />
      </div>
      <p className="text-sm font-medium mb-1.5" style={{ color: "var(--text-secondary)" }}>
        Ready to diagnose
      </p>
      <p className="text-[11px] leading-relaxed" style={{ color: "var(--text-muted)", maxWidth: 220 }}>
        Paste your Keras model code, then either:
        <br /><br />
        <strong style={{ color: "var(--text-secondary)" }}>Check Model</strong>: instant static analysis (~3-5 s)
        <br /><br />
        <strong style={{ color: "var(--text-secondary)" }}>Train &amp; Diagnose</strong>: real training + full 3-stage diagnosis with dynamic features
      </p>
    </div>
  );
}

// ── Loading skeleton ──────────────────────────────────────────────────────

function AnalyzingSkeleton() {
  return (
    <div className="space-y-3" role="status" aria-label="Analyzing…" aria-busy="true">
      <span className="sr-only">Analysis in progress…</span>
      {[72, 110, 56, 90].map((h, i) => (
        <div key={i} className="skeleton rounded-lg" style={{ height: h }} aria-hidden="true" />
      ))}
    </div>
  );
}
