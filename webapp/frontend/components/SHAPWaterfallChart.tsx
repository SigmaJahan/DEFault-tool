"use client";

import { useMemo } from "react";
import type { ShapFeature } from "@/types/analysis";

interface SHAPWaterfallChartProps {
  features: ShapFeature[];
  baseValue: number;
  predictedProbability: number;
}

const BAR_HEIGHT = 22;
const BAR_GAP = 6;
const LABEL_WIDTH = 160;
const VALUE_WIDTH = 56;
const CHART_PAD_TOP = 8;
const CHART_PAD_BOTTOM = 32; // for baseline label
const MIN_BAR_PX = 3;

function fmt(v: number) {
  return (v >= 0 ? "+" : "") + v.toFixed(4);
}

function fmtVal(v: number) {
  if (Math.abs(v) < 0.001) return v.toExponential(1);
  if (Number.isInteger(v)) return String(v);
  return v.toFixed(3);
}

export function SHAPWaterfallChart({
  features,
  baseValue,
  predictedProbability,
}: SHAPWaterfallChartProps) {
  const rows = useMemo(() => {
    // Build cumulative waterfall rows (top = most impactful)
    const sorted = [...features].sort((a, b) => b.abs_shap - a.abs_shap);
    let cumulative = baseValue;
    return sorted.map((f) => {
      const start = cumulative;
      const end = cumulative + f.shap_value;
      cumulative = end;
      return { ...f, start, end };
    });
  }, [features, baseValue]);

  // Domain: span all bar endpoints + base + prediction
  const allVals = [
    baseValue,
    predictedProbability,
    ...rows.flatMap((r) => [r.start, r.end]),
  ];
  const domainMin = Math.max(0, Math.min(...allVals) - 0.05);
  const domainMax = Math.min(1, Math.max(...allVals) + 0.05);
  const domainSpan = domainMax - domainMin || 0.01;

  const n = rows.length;
  const svgHeight =
    CHART_PAD_TOP +
    n * (BAR_HEIGHT + BAR_GAP) +
    BAR_HEIGHT + // final bar
    BAR_GAP +
    CHART_PAD_BOTTOM;

  const toX = (v: number, width: number) =>
    ((v - domainMin) / domainSpan) * width;

  return (
    <div>
      <div
        className="text-xs font-semibold uppercase tracking-wider mb-3"
        style={{ color: "var(--text-secondary)" }}
      >
        SHAP Waterfall: Root Cause
      </div>
      <div className="text-[10px] mb-2 flex gap-4">
        <span className="flex items-center gap-1">
          <span
            className="inline-block w-3 h-2 rounded-sm"
            style={{ background: "var(--shap-pos)" }}
          />
          <span style={{ color: "var(--text-muted)" }}>pushes toward buggy</span>
        </span>
        <span className="flex items-center gap-1">
          <span
            className="inline-block w-3 h-2 rounded-sm"
            style={{ background: "var(--shap-neg)" }}
          />
          <span style={{ color: "var(--text-muted)" }}>pushes toward correct</span>
        </span>
      </div>

      <div className="w-full overflow-x-auto">
        <div style={{ minWidth: 420 }}>
          {/* Header row */}
          <div
            className="flex text-[9px] uppercase tracking-wider pb-1 mb-1"
            style={{
              borderBottom: "1px solid var(--border-subtle)",
              color: "var(--text-muted)",
            }}
          >
            <div style={{ width: LABEL_WIDTH, flexShrink: 0 }}>Feature</div>
            <div className="flex-1" />
            <div style={{ width: VALUE_WIDTH, flexShrink: 0, textAlign: "right" }}>
              SHAP
            </div>
          </div>

          {/* Rows */}
          {rows.map((row, i) => (
            <WaterfallRow
              key={row.feature}
              row={row}
              index={i}
              domainMin={domainMin}
              domainSpan={domainSpan}
              labelWidth={LABEL_WIDTH}
              valueWidth={VALUE_WIDTH}
              barHeight={BAR_HEIGHT}
              barGap={BAR_GAP}
              minBarPx={MIN_BAR_PX}
              toX={toX}
              svgWidth={undefined}
            />
          ))}

          {/* Final prediction bar */}
          <FinalBar
            predictedProbability={predictedProbability}
            baseValue={baseValue}
            domainMin={domainMin}
            domainSpan={domainSpan}
            labelWidth={LABEL_WIDTH}
            valueWidth={VALUE_WIDTH}
            barHeight={BAR_HEIGHT}
            barGap={BAR_GAP}
          />

          {/* Base value label */}
          <div
            className="text-[10px] mt-1 font-mono"
            style={{ color: "var(--text-muted)", paddingLeft: LABEL_WIDTH }}
          >
            E[f(x)] = {baseValue.toFixed(4)}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Individual waterfall row ───────────────────────────────────────────────

interface RowData {
  feature: string;
  shap_value: number;
  abs_shap: number;
  feature_value: number;
  hint: string;
  start: number;
  end: number;
}

function WaterfallRow({
  row,
  domainMin,
  domainSpan,
  labelWidth,
  valueWidth,
  barHeight,
  barGap,
  minBarPx,
}: {
  row: RowData;
  index: number;
  domainMin: number;
  domainSpan: number;
  labelWidth: number;
  valueWidth: number;
  barHeight: number;
  barGap: number;
  minBarPx: number;
  toX: (v: number, w: number) => number;
  svgWidth: number | undefined;
}) {
  const isPos = row.shap_value >= 0;
  const color = isPos ? "var(--shap-pos)" : "var(--shap-neg)";
  const bgColor = isPos ? "var(--fault-red-bg)" : "var(--safe-green-bg)";

  return (
    <div
      className="flex items-center group"
      style={{ marginBottom: barGap, height: barHeight }}
      title={row.hint}
    >
      {/* Label */}
      <div
        className="font-mono text-[10px] truncate pr-2 flex-shrink-0"
        style={{ width: labelWidth, color: "var(--text-secondary)" }}
      >
        {row.feature}
        <span style={{ color: "var(--text-muted)", marginLeft: 4 }}>
          = {fmtVal(row.feature_value)}
        </span>
      </div>

      {/* Bar area */}
      <div className="flex-1 relative flex items-center" style={{ height: barHeight }}>
        <BarSvg
          start={row.start}
          end={row.end}
          domainMin={domainMin}
          domainSpan={domainSpan}
          barHeight={barHeight}
          color={color}
          bgColor={bgColor}
          minBarPx={minBarPx}
        />
      </div>

      {/* SHAP value */}
      <div
        className="font-mono text-[10px] flex-shrink-0 text-right"
        style={{ width: valueWidth, color }}
      >
        {fmt(row.shap_value)}
      </div>
    </div>
  );
}

function BarSvg({
  start,
  end,
  domainMin,
  domainSpan,
  barHeight,
  color,
  bgColor,
  minBarPx,
}: {
  start: number;
  end: number;
  domainMin: number;
  domainSpan: number;
  barHeight: number;
  color: string;
  bgColor: string;
  minBarPx: number;
}) {
  return (
    <svg
      className="w-full"
      height={barHeight}
      preserveAspectRatio="none"
      viewBox={`0 0 100 ${barHeight}`}
    >
      {/* Track */}
      <rect x={0} y={barHeight / 2 - 1} width={100} height={1.5} fill="var(--border-subtle)" />

      {/* Connector line from base */}
      {(() => {
        const startPct = ((start - domainMin) / domainSpan) * 100;
        const endPct = ((end - domainMin) / domainSpan) * 100;
        const left = Math.min(startPct, endPct);
        const right = Math.max(startPct, endPct);
        const widthPct = Math.max((right - left), (minBarPx / 1) * 0.5);
        const innerH = barHeight * 0.55;
        const y = (barHeight - innerH) / 2;
        return (
          <>
            <rect
              x={`${left}%`}
              y={y}
              width={`${widthPct}%`}
              height={innerH}
              fill={bgColor}
              rx={2}
            />
            <rect
              x={`${left}%`}
              y={y}
              width={`${widthPct}%`}
              height={innerH}
              fill={color}
              fillOpacity={0.75}
              rx={2}
            />
          </>
        );
      })()}

      {/* Cumulative end marker */}
      {(() => {
        const pct = ((end - domainMin) / domainSpan) * 100;
        return (
          <line
            x1={`${pct}%`}
            y1={0}
            x2={`${pct}%`}
            y2={barHeight}
            stroke={color}
            strokeWidth={1.5}
          />
        );
      })()}
    </svg>
  );
}

function FinalBar({
  predictedProbability,
  baseValue,
  domainMin,
  domainSpan,
  labelWidth,
  valueWidth,
  barHeight,
  barGap,
}: {
  predictedProbability: number;
  baseValue: number;
  domainMin: number;
  domainSpan: number;
  labelWidth: number;
  valueWidth: number;
  barHeight: number;
  barGap: number;
}) {
  const isPos = predictedProbability > baseValue;
  const color = isPos ? "var(--shap-pos)" : "var(--shap-neg)";

  const basePct = ((baseValue - domainMin) / domainSpan) * 100;
  const predPct = ((predictedProbability - domainMin) / domainSpan) * 100;
  const left = Math.min(basePct, predPct);
  const width = Math.abs(predPct - basePct);
  const innerH = barHeight * 0.65;
  const y = (barHeight - innerH) / 2;

  return (
    <div
      className="flex items-center"
      style={{ marginTop: barGap + 4, height: barHeight, borderTop: "1px dashed var(--border-subtle)", paddingTop: barGap }}
    >
      <div
        className="font-mono text-[10px] flex-shrink-0 pr-2"
        style={{ width: labelWidth, color: "var(--text-muted)" }}
      >
        f(x) · predicted
      </div>
      <div className="flex-1 relative" style={{ height: barHeight }}>
        <svg className="w-full" height={barHeight} preserveAspectRatio="none" viewBox={`0 0 100 ${barHeight}`}>
          <rect x={0} y={barHeight / 2 - 0.75} width={100} height={1.5} fill="var(--border-subtle)" />
          <rect x={`${left}%`} y={y} width={`${width}%`} height={innerH} fill={color} fillOpacity={0.9} rx={2} />
          <line
            x1={`${predPct}%`} y1={0}
            x2={`${predPct}%`} y2={barHeight}
            stroke={color} strokeWidth={2}
          />
        </svg>
      </div>
      <div
        className="font-mono text-[10px] flex-shrink-0 text-right"
        style={{ width: valueWidth, color }}
      >
        {predictedProbability.toFixed(4)}
      </div>
    </div>
  );
}
