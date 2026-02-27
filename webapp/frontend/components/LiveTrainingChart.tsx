"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ReferenceDot,
  ResponsiveContainer,
} from "recharts";
import type { EpochMetric } from "@/types/analysis";

interface LiveTrainingChartProps {
  epochs: EpochMetric[];
  totalEpochs: number;
  isRunning: boolean;
  isDummyData: boolean;
}

interface Annotation {
  epoch: number;
  label: string;
  color: string;
  short: string;
}

function detectAnnotations(epochs: EpochMetric[]): Annotation[] {
  const result: Annotation[] = [];
  const added = new Set<string>(); // prevent duplicate labels

  function add(a: Annotation) {
    if (!added.has(a.label)) {
      result.push(a);
      added.add(a.label);
    }
  }

  // Overfitting: train_acc > val_acc + 0.03 for 2+ consecutive epochs
  let ofStreak = 0;
  for (let i = 0; i < epochs.length; i++) {
    if (epochs[i].acc - epochs[i].val_acc > 0.03) {
      ofStreak++;
      if (ofStreak === 2) {
        add({ epoch: i + 1, label: "Overfitting", short: "overfit", color: "var(--fault-red)" });
      }
    } else {
      ofStreak = 0;
    }
  }

  // Loss spike: loss[i] > loss[i-1] * 1.2
  for (let i = 1; i < epochs.length; i++) {
    if (epochs[i - 1].loss > 0 && epochs[i].loss > epochs[i - 1].loss * 1.2) {
      add({ epoch: i + 1, label: "Loss Spike", short: "spike", color: "var(--warn-yellow)" });
    }
  }

  // Plateau: |val_acc[i] - val_acc[i-1]| < 0.001 for 3+ consecutive epochs
  let platStreak = 1;
  for (let i = 1; i < epochs.length; i++) {
    if (Math.abs(epochs[i].val_acc - epochs[i - 1].val_acc) < 0.001) {
      platStreak++;
      if (platStreak === 3) {
        add({ epoch: i + 1, label: "Plateau", short: "plateau", color: "#79c0ff" });
      }
    } else {
      platStreak = 1;
    }
  }

  // Underfitting: train_acc < 0.6 for all last 5 epochs
  if (epochs.length >= 5 && epochs.slice(-5).every((e) => e.acc < 0.6)) {
    add({ epoch: epochs[epochs.length - 1].epoch, label: "Underfitting", short: "underfit", color: "#d2a8ff" });
  }

  return result;
}

const TOOLTIP_STYLE = {
  background: "var(--bg-overlay)",
  border: "1px solid var(--border)",
  borderRadius: 6,
  fontSize: 11,
  color: "var(--text-primary)",
};

export function LiveTrainingChart({ epochs, totalEpochs, isRunning, isDummyData }: LiveTrainingChartProps) {
  const annotations = detectAnnotations(epochs);
  const latest = epochs[epochs.length - 1];
  const progress = epochs.length / Math.max(totalEpochs, 1);

  // Chart data — add epoch field for XAxis
  const data = epochs.map((e) => ({
    epoch:    e.epoch,
    loss:     +e.loss.toFixed(4),
    val_loss: +e.val_loss.toFixed(4),
    acc:      +e.acc.toFixed(4),
    val_acc:  +e.val_acc.toFixed(4),
  }));

  return (
    <div className="flex flex-col gap-2">
      {/* Header row */}
      <div className="flex items-center justify-between">
        <span className="text-[9px] uppercase tracking-widest font-semibold" style={{ color: "var(--text-muted)" }}>
          Live Training
        </span>
        <div className="flex items-center gap-2">
          {isDummyData && (
            <span
              className="text-[9px] px-1.5 py-0.5 rounded font-semibold"
              style={{
                background: "var(--warn-yellow-bg)",
                border: "1px solid var(--warn-yellow)",
                color: "var(--warn-yellow)",
              }}
            >
              SYNTHETIC DATA
            </span>
          )}
          {isRunning && (
            <span className="text-[10px]" style={{ color: "var(--text-secondary)" }}>
              Epoch {epochs.length}/{totalEpochs}
            </span>
          )}
          {!isRunning && epochs.length > 0 && (
            <span className="text-[10px]" style={{ color: "var(--safe-green)" }}>
              ✓ {epochs.length} epochs
            </span>
          )}
        </div>
      </div>

      {/* Progress bar */}
      <div
        className="rounded-full overflow-hidden"
        style={{ height: 3, background: "var(--bg-overlay)" }}
      >
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{
            width: `${Math.round(progress * 100)}%`,
            background: isRunning
              ? "linear-gradient(90deg, var(--accent), #58a6ff88)"
              : "var(--safe-green)",
          }}
        />
      </div>

      {/* Loss chart */}
      <div>
        <div className="flex items-center gap-3 mb-1">
          <span className="text-[9px] uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>Loss</span>
          <LegendDot color="var(--fault-red)" label="train" />
          <LegendDot color="var(--warn-yellow)" label="val" dashed />
        </div>
        <ResponsiveContainer width="100%" height={110}>
          <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
            <XAxis
              dataKey="epoch"
              tick={{ fontSize: 9, fill: "var(--text-muted)" }}
              tickLine={false}
              axisLine={false}
              interval="preserveEnd"
            />
            <YAxis
              tick={{ fontSize: 9, fill: "var(--text-muted)" }}
              tickLine={false}
              axisLine={false}
              width={32}
            />
            <Tooltip contentStyle={TOOLTIP_STYLE} labelFormatter={(v) => `Epoch ${v}`} />
            {annotations.map((a) => (
              <ReferenceLine key={`loss-a-${a.short}`} x={a.epoch} stroke={a.color} strokeDasharray="3 3" strokeWidth={1} />
            ))}
            <Line
              type="monotone"
              dataKey="loss"
              stroke="var(--fault-red)"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="val_loss"
              stroke="var(--warn-yellow)"
              strokeWidth={1.5}
              strokeDasharray="4 2"
              dot={false}
              isAnimationActive={false}
            />
            {/* Live cursor dot */}
            {isRunning && latest && (
              <ReferenceDot
                x={latest.epoch}
                y={+latest.loss.toFixed(4)}
                r={4}
                fill="var(--fault-red)"
                stroke="var(--bg-card)"
                strokeWidth={1.5}
                className="animate-pulse-live"
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Accuracy chart */}
      <div>
        <div className="flex items-center gap-3 mb-1">
          <span className="text-[9px] uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>Accuracy</span>
          <LegendDot color="var(--safe-green)" label="train" />
          <LegendDot color="#79c0ff" label="val" dashed />
        </div>
        <ResponsiveContainer width="100%" height={110}>
          <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
            <XAxis
              dataKey="epoch"
              tick={{ fontSize: 9, fill: "var(--text-muted)" }}
              tickLine={false}
              axisLine={false}
              interval="preserveEnd"
            />
            <YAxis
              tick={{ fontSize: 9, fill: "var(--text-muted)" }}
              tickLine={false}
              axisLine={false}
              width={32}
              domain={[0, 1]}
            />
            <Tooltip contentStyle={TOOLTIP_STYLE} labelFormatter={(v) => `Epoch ${v}`} />
            {annotations.map((a) => (
              <ReferenceLine key={`acc-a-${a.short}`} x={a.epoch} stroke={a.color} strokeDasharray="3 3" strokeWidth={1} />
            ))}
            <Line
              type="monotone"
              dataKey="acc"
              stroke="var(--safe-green)"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="val_acc"
              stroke="#79c0ff"
              strokeWidth={1.5}
              strokeDasharray="4 2"
              dot={false}
              isAnimationActive={false}
            />
            {isRunning && latest && (
              <ReferenceDot
                x={latest.epoch}
                y={+latest.acc.toFixed(4)}
                r={4}
                fill="var(--safe-green)"
                stroke="var(--bg-card)"
                strokeWidth={1.5}
                className="animate-pulse-live"
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Fault annotation cards */}
      {annotations.length > 0 && (
        <div className="flex flex-col gap-1 mt-1">
          {annotations.map((a) => (
            <div
              key={a.label}
              className="flex items-center gap-2 px-2 py-1 rounded text-[10px]"
              style={{
                borderLeft: `3px solid ${a.color}`,
                background: "var(--bg-overlay)",
                color: "var(--text-secondary)",
              }}
            >
              <span style={{ color: a.color, fontWeight: 600 }}>{a.label}</span>
              <span style={{ color: "var(--text-muted)" }}>detected at epoch {a.epoch}</span>
            </div>
          ))}
        </div>
      )}

      {/* Latest metrics summary row */}
      {latest && (
        <div
          className="grid grid-cols-4 gap-1 mt-0.5 text-center rounded py-1.5"
          style={{ background: "var(--bg-overlay)" }}
        >
          {[
            { label: "Loss",     value: latest.loss.toFixed(3),    color: "var(--fault-red)" },
            { label: "Val Loss", value: latest.val_loss.toFixed(3), color: "var(--warn-yellow)" },
            { label: "Acc",      value: (latest.acc * 100).toFixed(1) + "%",     color: "var(--safe-green)" },
            { label: "Val Acc",  value: (latest.val_acc * 100).toFixed(1) + "%", color: "#79c0ff" },
          ].map((m) => (
            <div key={m.label} className="flex flex-col">
              <span className="text-[8px] uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
                {m.label}
              </span>
              <span className="text-[11px] font-mono font-semibold" style={{ color: m.color }}>
                {m.value}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Small helpers ─────────────────────────────────────────────────────────

function LegendDot({ color, label, dashed = false }: { color: string; label: string; dashed?: boolean }) {
  return (
    <div className="flex items-center gap-1">
      <svg width={14} height={6}>
        <line
          x1="0" y1="3" x2="14" y2="3"
          stroke={color}
          strokeWidth={2}
          strokeDasharray={dashed ? "3 2" : undefined}
        />
      </svg>
      <span className="text-[9px]" style={{ color: "var(--text-muted)" }}>{label}</span>
    </div>
  );
}
