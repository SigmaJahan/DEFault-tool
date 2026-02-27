"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Loader2, CheckCircle, XCircle, ZoomIn, ZoomOut,
  Cpu, ShieldAlert, Layers3, Microscope,
} from "lucide-react";
import type { AnalysisMode, PipelineStage } from "@/types/analysis";
import { CATEGORY_META } from "@/types/analysis";

// ── Types ─────────────────────────────────────────────────────────────────

type StageStatus = "idle" | "active" | "done" | "skipped";

interface PipelineVisualizerProps {
  mode: AnalysisMode;
  currentStage: PipelineStage;
  detectionProb?: number;
  detectionPositive?: boolean;
  flaggedCategories?: string[];
  topRootCause?: string;
}

// ── Stage status resolver ─────────────────────────────────────────────────

function useStageStatuses(mode: AnalysisMode, currentStage: PipelineStage) {
  const isRunning = mode === "analyzing_static" || mode === "analyzing_full" || mode === "training";
  const isDone    = mode === "done_static" || mode === "done_full" || mode === "done_training";

  function get(id: PipelineStage): StageStatus {
    if (mode === "idle" || mode === "error") return "idle";
    if (isRunning && currentStage === id) return "active";
    if (isRunning && id < currentStage)   return "done";
    if (mode === "done_static") return id === 3 ? "done" : "skipped";
    if (isDone) return "done";
    return "idle";
  }
  return { s1: get(1), s2: get(2), s3: get(3), isRunning, isDone };
}

// ── Status icon ───────────────────────────────────────────────────────────

function StatusIcon({ status, size = 14 }: { status: StageStatus; size?: number }) {
  if (status === "active")
    return <Loader2 size={size} className="animate-spin" style={{ color: "var(--accent)" }} />;
  if (status === "done")
    return <CheckCircle size={size} style={{ color: "var(--safe-green)" }} />;
  if (status === "skipped")
    return <XCircle size={size} style={{ color: "var(--text-muted)", opacity: 0.4 }} />;
  return (
    <div style={{
      width: size, height: size, borderRadius: "50%",
      border: "1.5px solid var(--border-subtle)",
      background: "var(--bg-overlay)",
      flexShrink: 0,
    }} />
  );
}

// ── Vertical connector line ───────────────────────────────────────────────

function VLine({ lit = false, height = 22 }: { lit?: boolean; height?: number }) {
  return (
    <div style={{ display: "flex", justifyContent: "center", height }}>
      <div style={{
        width: 2, height: "100%",
        background: lit ? "var(--accent)" : "var(--border-subtle)",
        borderRadius: 1,
        transition: "background 0.4s ease",
      }} />
    </div>
  );
}

// ── Decision branch (between Stage 1 and Stage 2) ────────────────────────

function DecisionBranch({
  show, isFaulty,
}: { show: boolean; isFaulty: boolean }) {
  if (!show) return <VLine height={22} />;

  const faultColor   = isFaulty  ? "var(--fault-red)"  : "var(--border-subtle)";
  const correctColor = !isFaulty ? "var(--safe-green)"  : "var(--border-subtle)";

  return (
    <div style={{ position: "relative", height: 48 }}>
      {/* Vertical up */}
      <div style={{ position: "absolute", top: 0, left: "50%", transform: "translateX(-50%)", width: 2, height: 16, background: "var(--border-subtle)", borderRadius: 1 }} />
      {/* Horizontal bar */}
      <div style={{ position: "absolute", top: 16, left: "22%", right: "22%", height: 2, background: "var(--border-subtle)" }} />
      {/* Fault leg */}
      <div style={{ position: "absolute", top: 16, left: "22%", width: 2, height: 20, background: faultColor, borderRadius: 1, transition: "background 0.3s" }} />
      {/* Correct leg */}
      <div style={{ position: "absolute", top: 16, right: "22%", width: 2, height: 20, background: correctColor, borderRadius: 1, transition: "background 0.3s" }} />
      {/* Labels */}
      <div style={{
        position: "absolute", top: 17, left: "4%",
        fontSize: 8, fontWeight: 700, letterSpacing: "0.07em", textTransform: "uppercase",
        color: isFaulty ? "var(--fault-red)" : "var(--text-muted)",
        transition: "color 0.3s",
      }}>Faulty</div>
      <div style={{
        position: "absolute", top: 17, right: "4%",
        fontSize: 8, fontWeight: 700, letterSpacing: "0.07em", textTransform: "uppercase",
        color: !isFaulty ? "var(--safe-green)" : "var(--text-muted)",
        transition: "color 0.3s",
      }}>Correct</div>
    </div>
  );
}

// ── Node card ─────────────────────────────────────────────────────────────

function NodeCard({
  stageNum, icon, title, question, status, result, dimmed = false,
}: {
  stageNum?: number;
  icon: React.ReactNode;
  title: string;
  question: string;
  status: StageStatus;
  result?: React.ReactNode;
  dimmed?: boolean;
}) {
  const isActive = status === "active";
  const isDone   = status === "done";

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.22 }}
      style={{
        border: `1px solid ${isActive ? "var(--accent)" : isDone ? "var(--border)" : "var(--border-subtle)"}`,
        borderRadius: 10,
        background: isActive ? "var(--bg-overlay)" : "var(--bg-card)",
        boxShadow: isActive ? "0 0 0 1px var(--accent), 0 4px 20px rgba(88,166,255,0.08)" : "none",
        padding: "9px 12px",
        opacity: status === "skipped" ? 0.42 : dimmed ? 0.65 : 1,
        transition: "all 0.3s ease",
      }}
    >
      {/* Header row */}
      <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 4 }}>
        {stageNum !== undefined && (
          <span style={{
            fontSize: 8, fontWeight: 800, letterSpacing: "0.06em",
            padding: "2px 5px", borderRadius: 4,
            background: isActive ? "var(--accent)" : "var(--bg-overlay)",
            color: isActive ? "#fff" : "var(--text-muted)",
            border: `1px solid ${isActive ? "var(--accent)" : "var(--border-subtle)"}`,
            flexShrink: 0,
          }}>
            S{stageNum}
          </span>
        )}
        <span style={{ color: "var(--text-muted)", flexShrink: 0, lineHeight: 1 }}>{icon}</span>
        <span style={{
          fontSize: 11, fontWeight: 600, flex: 1,
          color: isActive ? "var(--accent)" : isDone ? "var(--text-primary)" : "var(--text-muted)",
        }}>
          {title}
        </span>
        <StatusIcon status={status} size={13} />
      </div>

      {/* Subline */}
      <div style={{ fontSize: 10, color: "var(--text-muted)", paddingLeft: 2, marginBottom: 2 }}>
        {question}
      </div>

      {/* Skipped hint */}
      {status === "skipped" && (
        <div style={{ fontSize: 9, color: "var(--text-muted)", marginTop: 4, fontStyle: "italic" }}>
          Requires training data
        </div>
      )}

      {/* Result */}
      <AnimatePresence>
        {result && (
          <motion.div
            key="result"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.28, delay: 0.08 }}
            style={{ overflow: "hidden" }}
          >
            <div style={{
              borderTop: "1px solid var(--border-subtle)",
              marginTop: 8, paddingTop: 8,
            }}>
              {result}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// ── Category chips ────────────────────────────────────────────────────────

function CategoryChips({ categories }: { categories: string[] }) {
  if (!categories.length) {
    return (
      <span style={{ fontSize: 10, color: "var(--safe-green)", display: "flex", alignItems: "center", gap: 5 }}>
        <CheckCircle size={11} />
        No fault categories flagged
      </span>
    );
  }
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
      {categories.map((cat) => {
        const meta = CATEGORY_META[cat.toLowerCase()];
        return (
          <span key={cat} style={{
            fontSize: 9, fontWeight: 700,
            padding: "2px 6px", borderRadius: 4,
            background: "var(--fault-red-bg)",
            border: "1px solid rgba(248,81,73,0.3)",
            color: meta?.color ?? "var(--fault-red)",
          }}>
            {meta?.icon} {meta?.label ?? cat}
          </span>
        );
      })}
    </div>
  );
}

// ── Correct terminal node ─────────────────────────────────────────────────

function CorrectTerminal() {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.25 }}
      style={{
        padding: "10px 16px", borderRadius: 10, textAlign: "center",
        border: "1px solid var(--safe-green)", background: "var(--safe-green-bg)",
      }}
    >
      <CheckCircle size={22} style={{ color: "var(--safe-green)", margin: "0 auto 5px" }} />
      <div style={{ fontSize: 12, fontWeight: 700, color: "var(--safe-green)" }}>Model Looks Correct</div>
      <div style={{ fontSize: 9, color: "var(--text-muted)", marginTop: 3 }}>
        No fault detected in Stage 1
      </div>
    </motion.div>
  );
}

// ── Main Component ────────────────────────────────────────────────────────

export function PipelineVisualizer({
  mode, currentStage,
  detectionProb, detectionPositive,
  flaggedCategories, topRootCause,
}: PipelineVisualizerProps) {
  const [zoom, setZoom] = useState(1);
  const wrapperRef = useRef<HTMLDivElement>(null);

  const { s1, s2, s3, isRunning, isDone } = useStageStatuses(mode, currentStage);
  const isStaticOnly   = mode === "done_static";
  const showBranch     = isDone && !isStaticOnly;
  const isFaulty       = detectionPositive === true;
  // Stages 2 & 3 still always run in full/training modes even if not faulty
  const showFullTree   = !showBranch || isFaulty || mode === "done_full" || mode === "done_training";

  const clampZoom = useCallback((delta: number) => {
    setZoom(z => Math.max(0.5, Math.min(1.8, parseFloat((z + delta).toFixed(1)))));
  }, []);

  // Ctrl/Cmd + scroll to zoom
  useEffect(() => {
    const el = wrapperRef.current;
    if (!el) return;
    function onWheel(e: WheelEvent) {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        clampZoom(e.deltaY > 0 ? -0.1 : 0.1);
      }
    }
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [clampZoom]);

  // Stage 1 result
  const s1Result = s1 === "done" ? (
    <div style={{ display: "flex", alignItems: "center", gap: 7, flexWrap: "wrap" }}>
      {detectionProb !== undefined && (
        <span style={{ fontFamily: "var(--font-mono, monospace)", fontSize: 10, color: "var(--text-code)" }}>
          P = {(detectionProb * 100).toFixed(1)}%
        </span>
      )}
      {detectionPositive !== undefined && (
        <span style={{
          fontSize: 9, fontWeight: 700, padding: "2px 8px", borderRadius: 20,
          background: detectionPositive ? "var(--fault-red-bg)" : "var(--safe-green-bg)",
          color: detectionPositive ? "var(--fault-red)" : "var(--safe-green)",
          border: `1px solid ${detectionPositive ? "var(--fault-red)" : "var(--safe-green)"}`,
        }}>
          {detectionPositive ? "FAULT DETECTED" : "LOOKS CORRECT"}
        </span>
      )}
      {isStaticOnly && (
        <span style={{ fontSize: 9, color: "var(--text-muted)", fontStyle: "italic" }}>
          Train model to unlock
        </span>
      )}
    </div>
  ) : undefined;

  // Stage 2 result
  const s2Result = (s2 === "done" && flaggedCategories !== undefined)
    ? <CategoryChips categories={flaggedCategories} />
    : undefined;

  // Stage 3 result
  const s3Result = s3 === "done" ? (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      {topRootCause ? (
        <>
          <span style={{ fontSize: 9, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
            Top cause:
          </span>
          <span style={{ fontFamily: "var(--font-mono, monospace)", fontSize: 10, color: "var(--text-code)" }}>
            {topRootCause}
          </span>
        </>
      ) : (
        <span style={{ fontSize: 10, color: "var(--text-muted)" }}>See SHAP chart for details</span>
      )}
    </div>
  ) : undefined;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>

      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span style={{
          fontSize: 9, fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase",
          color: "var(--text-muted)",
        }}>
          Fault Diagnosis Tree
        </span>
        {/* Zoom controls */}
        <div style={{ display: "flex", alignItems: "center", gap: 2 }}>
          <button
            className="btn-ghost"
            style={{ padding: "3px 5px", borderRadius: 5 }}
            onClick={() => clampZoom(-0.1)}
            disabled={zoom <= 0.5}
            title="Zoom out (Ctrl+Scroll)"
            aria-label="Zoom out"
          >
            <ZoomOut size={10} />
          </button>
          <button
            style={{
              fontSize: 9, fontFamily: "monospace",
              minWidth: 36, textAlign: "center",
              padding: "2px 4px", borderRadius: 4,
              background: zoom !== 1 ? "var(--accent)" : "var(--bg-overlay)",
              color: zoom !== 1 ? "#fff" : "var(--text-muted)",
              border: `1px solid ${zoom !== 1 ? "var(--accent)" : "var(--border-subtle)"}`,
              cursor: "pointer",
              transition: "all 0.2s",
            }}
            onClick={() => setZoom(1)}
            title="Reset zoom"
            aria-label={`Current zoom ${Math.round(zoom * 100)}%. Click to reset.`}
          >
            {Math.round(zoom * 100)}%
          </button>
          <button
            className="btn-ghost"
            style={{ padding: "3px 5px", borderRadius: 5 }}
            onClick={() => clampZoom(0.1)}
            disabled={zoom >= 1.8}
            title="Zoom in (Ctrl+Scroll)"
            aria-label="Zoom in"
          >
            <ZoomIn size={10} />
          </button>
        </div>
      </div>

      {/* Zoomable scrollable viewport */}
      <div
        ref={wrapperRef}
        style={{
          overflow: "auto",
          borderRadius: 10,
          border: "1px solid var(--border-subtle)",
          background: "var(--bg-base)",
          minHeight: 180,
          maxHeight: 440,
        }}
        role="region"
        aria-label="Fault diagnosis tree visualization"
      >
        <div style={{
          transform: `scale(${zoom})`,
          transformOrigin: "top center",
          transition: "transform 0.18s ease",
          padding: "14px 16px 16px",
          minWidth: 230,
        }}>

          {/* ── Root node ─────────────────────────── */}
          <div style={{ display: "flex", justifyContent: "center", marginBottom: 0 }}>
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.2 }}
              style={{
                display: "flex", alignItems: "center", gap: 6,
                fontSize: 9, fontWeight: 800, letterSpacing: "0.1em", textTransform: "uppercase",
                color: mode === "idle" ? "var(--text-muted)" : "var(--text-secondary)",
                padding: "5px 14px", borderRadius: 6,
                border: "1px solid var(--border-subtle)",
                background: "var(--bg-card)",
                transition: "all 0.3s",
              }}
            >
              <Cpu size={11} aria-hidden="true" />
              DNN Fault Diagnosis
            </motion.div>
          </div>

          <VLine lit={s1 === "active" || s1 === "done"} height={20} />

          {/* ── Stage 1: Fault Detection ──────────── */}
          <NodeCard
            stageNum={1}
            icon={<ShieldAlert size={12} />}
            title="Fault Detection"
            question="Is the DNN faulty?"
            status={s1}
            result={s1Result}
          />

          {/* ── Decision branch (after Stage 1) ───── */}
          <DecisionBranch show={showBranch} isFaulty={isFaulty} />

          {/* ── Fault path or Correct terminal ────── */}
          {showBranch && !isFaulty ? (
            /* Not faulty — terminal node, then show stages 2 & 3 dimmed */
            <>
              <CorrectTerminal />
              <VLine height={20} />
              {/* Stages 2 & 3 still ran; show them dimmed */}
              <NodeCard
                stageNum={2}
                icon={<Layers3 size={12} />}
                title="Fault Categorization"
                question="What type of fault?"
                status={s2}
                result={s2Result}
                dimmed
              />
              <VLine lit={s3 === "done"} height={20} />
              <NodeCard
                stageNum={3}
                icon={<Microscope size={12} />}
                title="Root Cause Analysis"
                question="Why is the DNN faulty?"
                status={s3}
                result={s3Result}
                dimmed
              />
            </>
          ) : (
            /* Faulty path (or idle/running/static) — full tree */
            <>
              <NodeCard
                stageNum={2}
                icon={<Layers3 size={12} />}
                title="Fault Categorization"
                question="What type of fault?"
                status={s2}
                result={s2Result}
              />
              <VLine lit={s3 === "active" || s3 === "done"} height={20} />
              <NodeCard
                stageNum={3}
                icon={<Microscope size={12} />}
                title="Root Cause Analysis"
                question="Why is the DNN faulty?"
                status={s3}
                result={s3Result}
              />
            </>
          )}

        </div>
      </div>

      {/* Hint */}
      <div style={{ fontSize: 9, color: "var(--text-muted)", textAlign: "right" }}>
        Ctrl+scroll or use +/- to zoom
      </div>
    </div>
  );
}
