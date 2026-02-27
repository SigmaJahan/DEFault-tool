"use client";

import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine,
} from "recharts";
import type { TrainingHistory, TrainingSummary } from "@/types/analysis";

interface TrainingCurveChartProps {
  history: TrainingHistory;
  summary: TrainingSummary;
}

interface TooltipPayload {
  color: string;
  name: string;
  value: number;
}

function CustomTooltip({ active, payload, label }: { active?: boolean; payload?: TooltipPayload[]; label?: number }) {
  if (!active || !payload?.length) return null;
  return (
    <div
      className="rounded-lg p-2.5 text-xs border shadow-xl"
      style={{ background: "var(--bg-overlay)", borderColor: "var(--border)" }}
    >
      <div className="font-medium mb-1.5" style={{ color: "var(--text-secondary)" }}>Epoch {label}</div>
      {payload.map((p) => (
        <div key={p.name} className="flex items-center gap-2 mb-0.5">
          <div className="w-2 h-2 rounded-full" style={{ background: p.color }} />
          <span style={{ color: "var(--text-secondary)" }}>{p.name}:</span>
          <span style={{ color: "var(--text-primary)", fontFamily: "var(--font-mono)" }}>
            {p.value.toFixed(4)}
          </span>
        </div>
      ))}
    </div>
  );
}

export function TrainingCurveChart({ history, summary }: TrainingCurveChartProps) {
  const epochs = history.loss.length;

  const data = Array.from({ length: epochs }, (_, i) => ({
    epoch: i + 1,
    "Train Loss":   history.loss[i],
    "Val Loss":     history.val_loss[i],
    "Train Acc":    history.train_acc[i],
    "Val Acc":      history.val_acc[i],
  }));

  // Detect annotations from summary
  const annotations: { epoch: number; label: string; color: string }[] = [];
  if (summary.loss_oscillation > 0.05) {
    // Find epoch with max loss oscillation
    let maxIdx = 1;
    let maxDiff = 0;
    for (let i = 1; i < history.loss.length; i++) {
      const d = Math.abs(history.loss[i] - history.loss[i - 1]);
      if (d > maxDiff) { maxDiff = d; maxIdx = i + 1; }
    }
    annotations.push({ epoch: maxIdx, label: "Loss oscillation", color: "var(--fault-red)" });
  }
  if (summary.acc_gap < -0.1) {
    // Find where the gap widened
    let maxGapIdx = epochs;
    let maxGap = 0;
    for (let i = 0; i < history.train_acc.length; i++) {
      const gap = history.train_acc[i] - history.val_acc[i];
      if (gap > maxGap) { maxGap = gap; maxGapIdx = i + 1; }
    }
    annotations.push({ epoch: maxGapIdx, label: "Acc gap widens", color: "var(--warn-yellow)" });
  }

  const tickStyle = { fill: "var(--text-secondary)", fontSize: 10, fontFamily: "JetBrains Mono, monospace" };

  return (
    <div>
      <div className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: "var(--text-secondary)" }}>
        Training Curves
      </div>

      {/* Loss chart */}
      <div className="mb-1">
        <div className="text-[11px] mb-1" style={{ color: "var(--text-muted)" }}>Loss</div>
        <ResponsiveContainer width="100%" height={120}>
          <LineChart data={data} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
            <XAxis dataKey="epoch" tick={tickStyle} tickLine={false} axisLine={false} />
            <YAxis tick={tickStyle} tickLine={false} axisLine={false} width={48} />
            <Tooltip content={<CustomTooltip />} />
            {annotations.map((ann) => (
              <ReferenceLine
                key={`loss-${ann.epoch}`}
                x={ann.epoch}
                stroke={ann.color}
                strokeDasharray="4 3"
                label={{ value: ann.label, position: "top", fill: ann.color, fontSize: 9 }}
              />
            ))}
            <Line dataKey="Train Loss" stroke="#ff7b72" strokeWidth={1.5} dot={false} />
            <Line dataKey="Val Loss"   stroke="#ffa657" strokeWidth={1.5} dot={false} strokeDasharray="5 3" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Accuracy chart */}
      <div>
        <div className="text-[11px] mb-1" style={{ color: "var(--text-muted)" }}>Accuracy</div>
        <ResponsiveContainer width="100%" height={120}>
          <LineChart data={data} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
            <XAxis dataKey="epoch" tick={tickStyle} tickLine={false} axisLine={false} />
            <YAxis tick={tickStyle} tickLine={false} axisLine={false} domain={[0, 1]} width={48} />
            <Tooltip content={<CustomTooltip />} />
            {annotations
              .filter((a) => a.label.toLowerCase().includes("acc"))
              .map((ann) => (
                <ReferenceLine
                  key={`acc-${ann.epoch}`}
                  x={ann.epoch}
                  stroke={ann.color}
                  strokeDasharray="4 3"
                  label={{ value: ann.label, position: "top", fill: ann.color, fontSize: 9 }}
                />
              ))}
            <Line dataKey="Train Acc" stroke="#58a6ff" strokeWidth={1.5} dot={false} />
            <Line dataKey="Val Acc"   stroke="#79c0ff" strokeWidth={1.5} dot={false} strokeDasharray="5 3" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Summary stats */}
      <div
        className="grid grid-cols-2 gap-x-3 gap-y-1 mt-3 pt-3 text-[11px]"
        style={{ borderTop: "1px solid var(--border-subtle)" }}
      >
        {[
          { label: "Epochs", value: summary.epochs },
          { label: "Final train acc", value: `${(summary.final_train_acc * 100).toFixed(1)}%` },
          { label: "Final val acc",   value: `${(summary.final_val_acc * 100).toFixed(1)}%` },
          { label: "Acc gap", value: `${(summary.acc_gap * 100).toFixed(1)}%`, warn: summary.acc_gap < -0.1 },
          { label: "Loss oscillation", value: summary.loss_oscillation.toFixed(4), warn: summary.loss_oscillation > 0.05 },
          { label: "↓acc epochs", value: summary.decrease_acc_count, warn: summary.decrease_acc_count > 2 },
        ].map(({ label, value, warn }) => (
          <div key={label} className="flex justify-between">
            <span style={{ color: "var(--text-muted)" }}>{label}</span>
            <span
              className="font-mono"
              style={{ color: warn ? "var(--warn-yellow)" : "var(--text-secondary)" }}
            >
              {value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
