import { useRef } from "react";

/** Pointer-capture drag handle. ``axis="x"`` is a thin vertical strip
 *  that drags horizontally (col-resize); ``axis="y"`` is a horizontal
 *  strip that drags vertically (row-resize). Emits per-move pixel
 *  deltas — parent does the geometry math. Double-click invokes
 *  ``onDoubleClick`` (used to reset to default split). Pattern lifted
 *  from JobsPanel's split handle; extracted here so the inference
 *  sub-panels can reuse the same gesture + styling (.analyze-split-x /
 *  .analyze-split-y CSS classes). */
export function DragHandle({
  axis,
  ariaLabel,
  onDragDelta,
  onDragEnd,
  onDoubleClick,
}: {
  axis: "x" | "y";
  ariaLabel: string;
  onDragDelta: (delta: number) => void;
  /** Called on pointerup / pointercancel / Home key, and after each
   *  arrow-key nudge. Parent uses this to persist layout once a
   *  gesture has settled — avoids hundreds of localStorage writes
   *  per second during a fast drag. */
  onDragEnd?: () => void;
  /** Tied to the handle's double-click as well as the Home key for
   *  keyboard users — both gestures mean "reset to default." */
  onDoubleClick?: () => void;
}) {
  const lastRef = useRef<{ x: number; y: number; pointerId: number } | null>(
    null,
  );
  return (
    <div
      className={axis === "x" ? "analyze-split-x" : "analyze-split-y"}
      role="separator"
      aria-orientation={axis === "x" ? "vertical" : "horizontal"}
      aria-label={ariaLabel}
      tabIndex={0}
      title="Drag to resize · double-click or Home to reset · arrow keys to nudge (Shift for x4)"
      onPointerDown={(e) => {
        e.preventDefault();
        (e.currentTarget as Element).setPointerCapture(e.pointerId);
        lastRef.current = {
          x: e.clientX,
          y: e.clientY,
          pointerId: e.pointerId,
        };
        document.body.style.cursor =
          axis === "x" ? "col-resize" : "row-resize";
        document.body.style.userSelect = "none";
      }}
      onPointerMove={(e) => {
        const last = lastRef.current;
        if (!last) return;
        const delta = axis === "x" ? e.clientX - last.x : e.clientY - last.y;
        if (delta !== 0) onDragDelta(delta);
        lastRef.current = { ...last, x: e.clientX, y: e.clientY };
      }}
      onPointerUp={(e) => {
        if (!lastRef.current) return;
        lastRef.current = null;
        try {
          (e.currentTarget as Element).releasePointerCapture(e.pointerId);
        } catch {
          /* already released */
        }
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
        onDragEnd?.();
      }}
      onPointerCancel={(e) => {
        lastRef.current = null;
        try {
          (e.currentTarget as Element).releasePointerCapture(e.pointerId);
        } catch {
          /* already released */
        }
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
        onDragEnd?.();
      }}
      onDoubleClick={onDoubleClick}
      onKeyDown={(e) => {
        const step = e.shiftKey ? 32 : 8;
        const decrease = axis === "x" ? "ArrowLeft" : "ArrowUp";
        const increase = axis === "x" ? "ArrowRight" : "ArrowDown";
        if (e.key === decrease) {
          e.preventDefault();
          onDragDelta(-step);
          onDragEnd?.();
        } else if (e.key === increase) {
          e.preventDefault();
          onDragDelta(step);
          onDragEnd?.();
        } else if (e.key === "Home") {
          e.preventDefault();
          onDoubleClick?.();
        }
      }}
    />
  );
}
