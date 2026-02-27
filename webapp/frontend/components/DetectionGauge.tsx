"use client";

import { useMemo } from "react";
import { motion } from "framer-motion";

interface DetectionGaugeProps {
  probability: number;
  threshold: number;
  isFault: boolean;
}

export function DetectionGauge({ probability, threshold, isFault }: DetectionGaugeProps) {
  const pct = Math.min(Math.max(probability, 0), 1);

  // Arc geometry
  const cx = 80, cy = 80, r = 60;
  const startAngle = -225;   // degrees, from top-left
  const sweepAngle = 270;    // total sweep

  function polarToXY(angleDeg: number, radius: number) {
    const rad = (angleDeg * Math.PI) / 180;
    return { x: cx + radius * Math.cos(rad), y: cy + radius * Math.sin(rad) };
  }

  const arcPath = useMemo(() => {
    function describeArc(pct: number) {
      const endAngle = startAngle + sweepAngle * pct;
      const start = polarToXY(startAngle, r);
      const end = polarToXY(endAngle, r);
      const largeArc = sweepAngle * pct > 180 ? 1 : 0;
      return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 1 ${end.x} ${end.y}`;
    }
    return describeArc(pct);
  }, [pct]);

  const trackPath = useMemo(() => {
    const start = polarToXY(startAngle, r);
    const end = polarToXY(startAngle + sweepAngle, r);
    return `M ${start.x} ${start.y} A ${r} ${r} 0 1 1 ${end.x} ${end.y}`;
  }, []);

  // Color based on probability
  const arcColor = isFault
    ? pct > 0.8 ? "#f85149" : pct > 0.6 ? "#ff7b72" : "#ffa657"
    : "#3fb950";

  return (
    <div className="flex flex-col items-center">
      <svg width={160} height={140} viewBox="0 0 160 140">
        {/* Track */}
        <path d={trackPath} fill="none" stroke="var(--border)" strokeWidth={10} strokeLinecap="round" />

        {/* Threshold marker */}
        {(() => {
          const threshAngle = startAngle + sweepAngle * threshold;
          const inner = polarToXY(threshAngle, r - 7);
          const outer = polarToXY(threshAngle, r + 7);
          return (
            <line
              x1={inner.x} y1={inner.y}
              x2={outer.x} y2={outer.y}
              stroke="var(--warn-yellow)"
              strokeWidth={2}
              strokeDasharray="3 2"
            />
          );
        })()}

        {/* Filled arc */}
        <motion.path
          d={arcPath}
          fill="none"
          stroke={arcColor}
          strokeWidth={10}
          strokeLinecap="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        />

        {/* Center percentage */}
        <text
          x={cx} y={cy - 4}
          textAnchor="middle"
          fill="var(--text-primary)"
          fontSize={22}
          fontWeight="700"
          fontFamily="JetBrains Mono, monospace"
        >
          {(pct * 100).toFixed(0)}%
        </text>
        <text
          x={cx} y={cy + 14}
          textAnchor="middle"
          fill="var(--text-secondary)"
          fontSize={9}
          fontFamily="Inter, sans-serif"
        >
          fault probability
        </text>

        {/* Min/Max labels */}
        <text x={20} y={132} fill="var(--text-muted)" fontSize={9} textAnchor="middle">0%</text>
        <text x={140} y={132} fill="var(--text-muted)" fontSize={9} textAnchor="middle">100%</text>

        {/* Threshold label */}
        {(() => {
          const threshAngle = startAngle + sweepAngle * threshold;
          const pos = polarToXY(threshAngle, r - 20);
          return (
            <text
              x={pos.x} y={pos.y}
              textAnchor="middle"
              fill="var(--warn-yellow)"
              fontSize={8}
              fontFamily="JetBrains Mono, monospace"
            >
              τ={threshold.toFixed(2)}
            </text>
          );
        })()}
      </svg>

      {/* Status badge */}
      <motion.div
        initial={{ opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="text-xs font-semibold px-3 py-1 rounded-full mt-1"
        style={{
          background: isFault ? "var(--fault-red-bg)" : "var(--safe-green-bg)",
          color: isFault ? "var(--fault-red)" : "var(--safe-green)",
          border: `1px solid ${isFault ? "var(--fault-red)" : "var(--safe-green)"}`,
          letterSpacing: "0.05em",
        }}
      >
        {isFault ? "FAULT DETECTED" : "MODEL LOOKS CORRECT"}
      </motion.div>
    </div>
  );
}
