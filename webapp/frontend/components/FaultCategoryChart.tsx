"use client";

import { motion } from "framer-motion";
import type { CategoryResult } from "@/types/analysis";
import { CATEGORY_META } from "@/types/analysis";

interface FaultCategoryChartProps {
  categories: CategoryResult[];
}

export function FaultCategoryChart({ categories }: FaultCategoryChartProps) {
  // Ensure consistent order, sorted by probability descending
  const sorted = [...categories].sort((a, b) => b.probability - a.probability);

  return (
    <div className="flex flex-col gap-1.5">
      {sorted.map((cat, i) => {
        const meta = CATEGORY_META[cat.name];
        const pct = Math.max(cat.probability * 100, 0);
        const isFlag = cat.predicted_positive;
        const threshPct = cat.threshold * 100;

        return (
          <motion.div
            key={cat.name}
            initial={{ opacity: 0, x: -12 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.05 }}
            className="group"
          >
            {/* Label row */}
            <div className="flex items-center justify-between mb-0.5">
              <div className="flex items-center gap-1.5">
                <span className="text-xs">{meta.icon}</span>
                <span
                  className="text-xs font-medium"
                  style={{ color: isFlag ? meta.color : "var(--text-secondary)" }}
                >
                  {meta.label}
                </span>
                {isFlag && (
                  <span
                    className="text-[10px] px-1.5 py-0 rounded-full font-semibold"
                    style={{
                      background: `${meta.color}20`,
                      color: meta.color,
                      border: `1px solid ${meta.color}60`,
                    }}
                  >
                    FLAGGED
                  </span>
                )}
              </div>
              <span
                className="text-xs font-mono"
                style={{ color: isFlag ? meta.color : "var(--text-muted)" }}
              >
                {(cat.probability * 100).toFixed(1)}%
              </span>
            </div>

            {/* Bar track */}
            <div
              className="relative h-2 rounded-full overflow-visible"
              style={{ background: "var(--border-subtle)" }}
            >
              {/* Filled bar */}
              <motion.div
                className="absolute top-0 left-0 h-full rounded-full"
                initial={{ width: 0 }}
                animate={{ width: `${pct}%` }}
                transition={{ duration: 0.5, ease: "easeOut", delay: i * 0.05 + 0.1 }}
                style={{
                  background: isFlag
                    ? `linear-gradient(90deg, ${meta.color}80, ${meta.color})`
                    : "var(--border)",
                }}
              />

              {/* Threshold marker */}
              <div
                className="absolute top-[-3px] w-px h-[14px] rounded-sm"
                style={{
                  left: `${threshPct}%`,
                  background: "var(--warn-yellow)",
                  opacity: 0.7,
                }}
                title={`Threshold: ${(cat.threshold * 100).toFixed(1)}%`}
              />
            </div>
          </motion.div>
        );
      })}

      {/* Legend */}
      <div className="flex items-center gap-3 mt-2 pt-2" style={{ borderTop: "1px solid var(--border-subtle)" }}>
        <div className="flex items-center gap-1">
          <div className="w-3 h-0.5 rounded" style={{ background: "var(--warn-yellow)" }} />
          <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>threshold (τ)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-1.5 rounded" style={{ background: "var(--accent)" }} />
          <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>below threshold</span>
        </div>
      </div>
    </div>
  );
}
