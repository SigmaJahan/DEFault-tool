"use client";

import { useState, useCallback } from "react";
import { ChevronRight, ChevronDown } from "lucide-react";
import type { FaultTaxonomyNode } from "@/types/analysis";
import { CATEGORY_META } from "@/types/analysis";

interface FaultTaxonomyTreeProps {
  root: FaultTaxonomyNode;
  flaggedCategories: string[];   // from stage2_categories.flagged
  topShapFeatures: string[];     // feature names from stage3_static.top_features
}

export function FaultTaxonomyTree({
  root,
  flaggedCategories,
  topShapFeatures,
}: FaultTaxonomyTreeProps) {
  // Normalise for comparison
  const flaggedSet = new Set(flaggedCategories.map((c) => c.toLowerCase()));
  const shapSet = new Set(topShapFeatures.map((f) => f.toLowerCase()));

  return (
    <div>
      <div
        className="text-xs font-semibold uppercase tracking-wider mb-3"
        style={{ color: "var(--text-secondary)" }}
      >
        Fault Taxonomy
      </div>
      <div className="text-[10px] mb-2 flex gap-3 flex-wrap">
        {flaggedCategories.length > 0 && flaggedCategories.map((cat) => {
          const meta = CATEGORY_META[cat.toLowerCase()];
          return (
            <span
              key={cat}
              className="px-1.5 py-0.5 rounded text-[9px] font-medium uppercase"
              style={{
                background: "var(--fault-red-bg)",
                color: "var(--fault-red)",
                border: "1px solid var(--fault-red)",
              }}
            >
              {meta?.icon} {meta?.label ?? cat}
            </span>
          );
        })}
        {flaggedCategories.length === 0 && (
          <span style={{ color: "var(--text-muted)" }}>No categories flagged</span>
        )}
      </div>

      <TreeNode
        node={root}
        depth={0}
        flaggedSet={flaggedSet}
        shapSet={shapSet}
        defaultOpen
      />
    </div>
  );
}

interface TreeNodeProps {
  node: FaultTaxonomyNode;
  depth: number;
  flaggedSet: Set<string>;
  shapSet: Set<string>;
  defaultOpen?: boolean;
}

function TreeNode({ node, depth, flaggedSet, shapSet, defaultOpen = false }: TreeNodeProps) {
  const [open, setOpen] = useState(defaultOpen || depth < 1);

  const isFlagged =
    node.fault_category != null &&
    flaggedSet.has(node.fault_category.toLowerCase());

  const isShapHighlighted = shapSet.has(node.id.toLowerCase()) || shapSet.has(node.label.toLowerCase());

  const hasChildren = node.children.length > 0;

  const toggleOpen = useCallback(() => {
    if (hasChildren) setOpen((v) => !v);
  }, [hasChildren]);

  const meta = node.fault_category
    ? CATEGORY_META[node.fault_category.toLowerCase()]
    : null;

  const indentPx = depth * 14;

  return (
    <div>
      <div
        className="flex items-start gap-1 py-0.5 rounded cursor-pointer group"
        style={{ paddingLeft: indentPx, paddingRight: 4 }}
        onClick={toggleOpen}
      >
        {/* Expand toggle */}
        <div
          className="flex-shrink-0 mt-0.5"
          style={{ width: 14, color: "var(--text-muted)" }}
        >
          {hasChildren ? (
            open ? (
              <ChevronDown size={10} />
            ) : (
              <ChevronRight size={10} />
            )
          ) : (
            <span style={{ display: "inline-block", width: 10 }} />
          )}
        </div>

        {/* Node content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5 flex-wrap">
            {/* Category color dot */}
            {meta && (
              <span
                className="inline-block w-2 h-2 rounded-full flex-shrink-0"
                style={{ background: meta.color }}
              />
            )}

            {/* Label */}
            <span
              className="text-[11px] font-medium truncate"
              style={{
                color: isFlagged
                  ? "var(--fault-red)"
                  : isShapHighlighted
                  ? "var(--accent)"
                  : depth === 0
                  ? "var(--text-primary)"
                  : "var(--text-secondary)",
              }}
            >
              {node.label}
            </span>

            {/* Flagged badge */}
            {isFlagged && (
              <span
                className="text-[8px] px-1 py-0.5 rounded uppercase font-bold flex-shrink-0"
                style={{
                  background: "var(--fault-red-bg)",
                  color: "var(--fault-red)",
                }}
              >
                flagged
              </span>
            )}

            {/* SHAP badge */}
            {isShapHighlighted && !isFlagged && (
              <span
                className="text-[8px] px-1 py-0.5 rounded uppercase font-bold flex-shrink-0"
                style={{
                  background: "rgba(88,166,255,0.15)",
                  color: "var(--accent)",
                }}
              >
                top cause
              </span>
            )}
          </div>

          {/* Description shown only for flagged or root nodes */}
          {(isFlagged || depth === 0) && node.description && (
            <div
              className="text-[9px] mt-0.5 leading-relaxed"
              style={{ color: "var(--text-muted)" }}
            >
              {node.description}
            </div>
          )}
        </div>
      </div>

      {/* Children */}
      {hasChildren && open && (
        <div
          style={{
            borderLeft: depth === 0 ? "none" : "1px solid var(--border-subtle)",
            marginLeft: indentPx + 6,
          }}
        >
          {node.children.map((child) => (
            <TreeNode
              key={child.id}
              node={child}
              depth={depth + 1}
              flaggedSet={flaggedSet}
              shapSet={shapSet}
            />
          ))}
        </div>
      )}
    </div>
  );
}
