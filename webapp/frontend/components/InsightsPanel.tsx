"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight, Copy, Check, AlertTriangle, CheckCircle } from "lucide-react";
import type { ShapFeature, StaticAnalysisResult, StageTwoResult } from "@/types/analysis";
import { CATEGORY_META } from "@/types/analysis";

interface InsightsPanelProps {
  stage3?: StaticAnalysisResult | null;
  stage2?: StageTwoResult | null;
  warnings?: string[];
}

export function InsightsPanel({ stage3, stage2, warnings }: InsightsPanelProps) {
  const hasStage3 = stage3 != null && stage3.top_features.length > 0;
  const hasFlagged = stage2 != null && stage2.flagged.length > 0;

  if (!hasStage3 && !hasFlagged && !warnings?.length) {
    return (
      <div
        className="text-[11px] text-center py-6"
        style={{ color: "var(--text-muted)" }}
      >
        No insights yet. Run an analysis first.
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {/* Header */}
      <div
        className="text-xs font-semibold uppercase tracking-wider mb-3"
        style={{ color: "var(--text-secondary)" }}
      >
        Insights &amp; Fix Suggestions
      </div>

      {/* Warnings */}
      {warnings?.map((w, i) => (
        <WarningBanner key={i} message={w} />
      ))}

      {/* Static analysis verdict */}
      {stage3 && (
        <VerdictCard
          probabilityBuggy={stage3.probability_buggy}
          threshold={stage3.threshold}
          predicted={stage3.predicted_buggy}
        />
      )}

      {/* Flagged categories from stage 2 */}
      {hasFlagged && stage2 && (
        <div className="space-y-1.5">
          {stage2.flagged.map((cat) => {
            const category = stage2.categories.find(
              (c) => c.name.toLowerCase() === cat.toLowerCase()
            );
            if (!category) return null;
            return (
              <CategoryInsightCard
                key={cat}
                name={cat}
                probability={category.probability}
                threshold={category.threshold}
              />
            );
          })}
        </div>
      )}

      {/* SHAP feature hints */}
      {hasStage3 && stage3 && (
        <div className="space-y-1.5 mt-2">
          {stage3.top_features.map((f) => (
            <ShapHintCard key={f.feature} feature={f} />
          ))}
        </div>
      )}
    </div>
  );
}

// ── Warning banner ─────────────────────────────────────────────────────────

function WarningBanner({ message }: { message: string }) {
  return (
    <div
      className="flex items-start gap-2 text-[11px] px-3 py-2 rounded-md"
      style={{
        background: "var(--warn-yellow-bg)",
        border: "1px solid var(--warn-yellow)",
        color: "var(--warn-yellow)",
      }}
    >
      <AlertTriangle size={12} className="flex-shrink-0 mt-0.5" />
      <span>{message}</span>
    </div>
  );
}

// ── Verdict card ───────────────────────────────────────────────────────────

function VerdictCard({
  probabilityBuggy,
  threshold,
  predicted,
}: {
  probabilityBuggy: number;
  threshold: number;
  predicted: boolean;
}) {
  const pct = (probabilityBuggy * 100).toFixed(1);
  const bgColor = predicted ? "var(--fault-red-bg)" : "var(--safe-green-bg)";
  const borderColor = predicted ? "var(--fault-red)" : "var(--safe-green)";
  const textColor = predicted ? "var(--fault-red)" : "var(--safe-green)";
  const Icon = predicted ? AlertTriangle : CheckCircle;

  return (
    <div
      className="flex items-center gap-3 px-3 py-2.5 rounded-md"
      style={{ background: bgColor, border: `1px solid ${borderColor}` }}
    >
      <Icon size={18} style={{ color: textColor, flexShrink: 0 }} />
      <div>
        <div className="text-[12px] font-semibold" style={{ color: textColor }}>
          {predicted ? "Fault Detected" : "Model Looks Correct"}
        </div>
        <div className="text-[10px] font-mono" style={{ color: "var(--text-muted)" }}>
          P(buggy) = {pct}% · threshold = {(threshold * 100).toFixed(0)}%
        </div>
      </div>
    </div>
  );
}

// ── Category insight card ──────────────────────────────────────────────────

function CategoryInsightCard({
  name,
  probability,
  threshold,
}: {
  name: string;
  probability: number;
  threshold: number;
}) {
  const meta = CATEGORY_META[name.toLowerCase()];
  const label = meta?.label ?? name;
  const color = meta?.color ?? "var(--fault-red)";
  const description = meta?.description ?? "";

  return (
    <div
      className="px-3 py-2 rounded-md text-[11px]"
      style={{
        background: "rgba(248,81,73,0.08)",
        border: "1px solid rgba(248,81,73,0.25)",
      }}
    >
      <div className="flex items-center gap-2 mb-0.5">
        <span style={{ color }}>{meta?.icon ?? "⚠️"}</span>
        <span className="font-semibold" style={{ color }}>
          {label} Fault
        </span>
        <span
          className="ml-auto font-mono text-[10px]"
          style={{ color: "var(--text-muted)" }}
        >
          {(probability * 100).toFixed(1)}% &gt; {(threshold * 100).toFixed(0)}% τ
        </span>
      </div>
      {description && (
        <div style={{ color: "var(--text-muted)" }}>{description}</div>
      )}
    </div>
  );
}

// ── SHAP hint expandable card ──────────────────────────────────────────────

function ShapHintCard({ feature }: { feature: ShapFeature }) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const isPos = feature.shap_value >= 0;
  const color = isPos ? "var(--shap-pos)" : "var(--shap-neg)";
  const bgColor = isPos ? "var(--fault-red-bg)" : "var(--safe-green-bg)";

  function handleCopy() {
    navigator.clipboard.writeText(feature.hint).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }

  return (
    <div
      className="rounded-md overflow-hidden"
      style={{ border: "1px solid var(--border)" }}
    >
      {/* Header row */}
      <button
        className="w-full flex items-center gap-2 px-3 py-2 text-left"
        style={{ background: bgColor }}
        onClick={() => setOpen((v) => !v)}
      >
        {open ? (
          <ChevronDown size={10} style={{ color: "var(--text-muted)", flexShrink: 0 }} />
        ) : (
          <ChevronRight size={10} style={{ color: "var(--text-muted)", flexShrink: 0 }} />
        )}
        <span
          className="font-mono text-[10px] flex-1 text-left truncate"
          style={{ color: "var(--text-secondary)" }}
        >
          {feature.feature}
        </span>
        <span
          className="font-mono text-[10px] flex-shrink-0"
          style={{ color }}
        >
          {(feature.shap_value >= 0 ? "+" : "") + feature.shap_value.toFixed(4)}
        </span>
        <span
          className="text-[9px] flex-shrink-0 ml-1"
          style={{ color: "var(--text-muted)" }}
        >
          val={feature.feature_value.toFixed(2)}
        </span>
      </button>

      {/* Expanded hint */}
      {open && (
        <div
          className="px-3 py-2.5"
          style={{ background: "var(--bg-card)", borderTop: "1px solid var(--border-subtle)" }}
        >
          <div
            className="text-[11px] leading-relaxed mb-2"
            style={{ color: "var(--text-secondary)" }}
          >
            {feature.hint}
          </div>
          <button
            className="flex items-center gap-1.5 text-[10px] px-2 py-1 rounded"
            style={{
              background: "var(--bg-overlay)",
              border: "1px solid var(--border)",
              color: copied ? "var(--safe-green)" : "var(--text-muted)",
            }}
            onClick={handleCopy}
          >
            {copied ? (
              <Check size={10} />
            ) : (
              <Copy size={10} />
            )}
            {copied ? "Copied!" : "Copy hint"}
          </button>
        </div>
      )}
    </div>
  );
}
