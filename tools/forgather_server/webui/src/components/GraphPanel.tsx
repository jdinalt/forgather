import { useQuery } from "@tanstack/react-query";
import { useCallback, useEffect, useRef, useState } from "react";
import { instance } from "@viz-js/viz";

import { api, ConfigInfo, ProjectInfo } from "../api";

interface Props {
  project: ProjectInfo;
  config: ConfigInfo;
}

/** Two-pane graph view: target list (left) + Graphviz node-graph SVG (right).
 *
 *  The target list is populated from /api/config/code-targets (same set the
 *  code panel uses).  Selecting a target re-fetches the DOT graph from
 *  /api/config/graph and re-renders it via viz-js (the same WASM renderer
 *  the templates view uses for trefs). */
export function GraphPanel({ project, config }: Props) {
  const [target, setTarget] = useState<string>("");
  const [includeValues, setIncludeValues] = useState<boolean>(false);

  // Reset when the config changes so we don't request a stale target.
  useEffect(() => {
    setTarget("");
  }, [project.project_dir, config.name]);

  const targetsQ = useQuery({
    queryKey: ["code-targets", project.project_dir, config.name],
    queryFn: () => api.configCodeTargets(project.project_dir, config.name),
  });

  const dotQ = useQuery({
    queryKey: [
      "graph-dot",
      project.project_dir,
      config.name,
      target,
      includeValues,
    ],
    queryFn: () =>
      api.configGraphDot(project.project_dir, config.name, target, includeValues),
    enabled: targetsQ.isSuccess || targetsQ.isError,
  });

  const targets = targetsQ.data ?? [];

  return (
    <div className="graph-view">
      <div className="code-targets">
        <div className="code-targets-header">
          {targets.length} target{targets.length === 1 ? "" : "s"}
        </div>
        <label className="graph-toggle" title="Also render scalars, lists, and dicts as graph nodes">
          <input
            type="checkbox"
            checked={includeValues}
            onChange={(e) => setIncludeValues(e.target.checked)}
          />
          Show values
        </label>
        <ul className="code-target-items">
          <li
            className={target === "" ? "code-target-item active" : "code-target-item"}
            onClick={() => setTarget("")}
            title="Show all targets in one diagram"
          >
            <em>(all targets)</em>
          </li>
          {targets.map((t) => (
            <li
              key={t}
              className={t === target ? "code-target-item active" : "code-target-item"}
              onClick={() => setTarget(t)}
              title={t}
            >
              {t}
            </li>
          ))}
        </ul>
      </div>
      <GraphCanvas
        dot={dotQ.data ?? null}
        loading={dotQ.isLoading || targetsQ.isLoading}
        error={dotQ.error ?? targetsQ.error}
      />
    </div>
  );
}

type ViewBox = [number, number, number, number]; // [x, y, w, h]

const MIN_ZOOM = 0.05;
const MAX_ZOOM = 50;

/** Meet-fit scale: how many CSS pixels per user unit, given a container size
 *  and a viewBox.  Matches preserveAspectRatio="xMidYMid meet". */
function meetScale(cw: number, ch: number, vw: number, vh: number): number {
  return Math.min(cw / vw, ch / vh);
}

function GraphCanvas({
  dot,
  loading,
  error,
}: {
  dot: string | null;
  loading: boolean;
  error: unknown;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const hostRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  // Natural viewBox emitted by Graphviz; never changes after load.
  const naturalVbRef = useRef<ViewBox | null>(null);
  // Live viewBox (also mirrored to a ref so wheel/drag handlers don't
  // need to be re-bound when state changes).
  const [viewBox, setViewBox] = useState<ViewBox | null>(null);
  const viewBoxRef = useRef<ViewBox | null>(null);
  viewBoxRef.current = viewBox;
  const dragRef = useRef<{
    cx: number;
    cy: number;
    vbX: number;
    vbY: number;
  } | null>(null);

  /** Set both the React state and the SVG's `viewBox` attribute.  Updating
   *  the attribute imperatively keeps the rendered graph in sync on every
   *  wheel/drag tick without waiting for React's render cycle. */
  const applyViewBox = useCallback((vb: ViewBox) => {
    setViewBox(vb);
    if (svgRef.current) {
      svgRef.current.setAttribute("viewBox", vb.join(" "));
    }
  }, []);

  // Render new DOT and reset the view to the natural viewBox.
  useEffect(() => {
    if (!dot || !hostRef.current) return;
    let cancelled = false;
    instance().then((viz) => {
      if (cancelled || !hostRef.current) return;
      try {
        const svg = viz.renderSVGElement(dot);
        // Parse the viewBox graphviz emits; that's the user-coordinate
        // system everything else is expressed in.
        const vbAttr = svg.getAttribute("viewBox");
        const parts = vbAttr?.split(/\s+/).map(Number);
        let natural: ViewBox;
        if (parts && parts.length === 4 && parts.every(Number.isFinite)) {
          natural = [parts[0], parts[1], parts[2], parts[3]];
        } else {
          // Fall back to width/height attributes if no viewBox.
          const w = parseFloat(svg.getAttribute("width") || "500") || 500;
          const h = parseFloat(svg.getAttribute("height") || "500") || 500;
          natural = [0, 0, w, h];
        }
        naturalVbRef.current = natural;
        // Make the SVG fill the host so the viewBox controls everything we
        // see.  preserveAspectRatio="xMidYMid meet" is SVG's default but
        // we set it explicitly so the cursor-anchored zoom math is correct.
        svg.setAttribute("width", "100%");
        svg.setAttribute("height", "100%");
        svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
        hostRef.current.innerHTML = "";
        hostRef.current.appendChild(svg);
        svgRef.current = svg;
        applyViewBox([natural[0], natural[1], natural[2], natural[3]]);
      } catch {
        if (hostRef.current) {
          hostRef.current.innerHTML = `<pre class="graph-dot-error">Failed to render graph</pre>`;
        }
      }
    });
    return () => {
      cancelled = true;
    };
  }, [dot, applyViewBox]);

  // Wheel-to-zoom.  Native (non-passive) listener so we can preventDefault
  // to stop the page from scrolling.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const vb = viewBoxRef.current;
      const nat = naturalVbRef.current;
      if (!vb || !nat) return;
      const rect = el.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;
      const [vx, vy, vw, vh] = vb;
      const s = meetScale(rect.width, rect.height, vw, vh);
      // The viewBox is centered inside the container by xMidYMid meet,
      // so account for letterboxing when converting screen -> user coords.
      const ox = (rect.width - vw * s) / 2;
      const oy = (rect.height - vh * s) / 2;
      const ux = vx + (cx - ox) / s;
      const uy = vy + (cy - oy) / s;
      // Each notch scales by ~e^(0.0015 * deltaY); trackpad pixel deltas
      // accumulate smoothly, mouse wheel notches are visible jumps.
      const factor = Math.exp(-e.deltaY * 0.0015);
      const currentZoom = nat[2] / vw; // 1 == natural; >1 == zoomed in
      const newZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, currentZoom * factor));
      const k = newZoom / currentZoom;
      const newVw = vw / k;
      const newVh = vh / k;
      // Recenter so the user point under the cursor stays put.
      const newS = meetScale(rect.width, rect.height, newVw, newVh);
      const newOx = (rect.width - newVw * newS) / 2;
      const newOy = (rect.height - newVh * newS) / 2;
      const newVx = ux - (cx - newOx) / newS;
      const newVy = uy - (cy - newOy) / newS;
      applyViewBox([newVx, newVy, newVw, newVh]);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [applyViewBox]);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return;
    // Don't pan when interacting with the controls overlay.
    if ((e.target as HTMLElement).closest(".graph-controls")) return;
    const vb = viewBoxRef.current;
    if (!vb) return;
    dragRef.current = { cx: e.clientX, cy: e.clientY, vbX: vb[0], vbY: vb[1] };
    e.preventDefault();
  }, []);

  // Drag listeners on window so panning continues if the cursor leaves
  // the canvas mid-gesture.
  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      const d = dragRef.current;
      const vb = viewBoxRef.current;
      const el = containerRef.current;
      if (!d || !vb || !el) return;
      const rect = el.getBoundingClientRect();
      const [, , vw, vh] = vb;
      const s = meetScale(rect.width, rect.height, vw, vh);
      const dx = (e.clientX - d.cx) / s;
      const dy = (e.clientY - d.cy) / s;
      applyViewBox([d.vbX - dx, d.vbY - dy, vw, vh]);
    };
    const onUp = () => {
      dragRef.current = null;
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [applyViewBox]);

  const resetView = useCallback(() => {
    const nat = naturalVbRef.current;
    if (nat) applyViewBox([nat[0], nat[1], nat[2], nat[3]]);
  }, [applyViewBox]);

  const zoomPct =
    viewBox && naturalVbRef.current
      ? Math.round((naturalVbRef.current[2] / viewBox[2]) * 100)
      : 100;

  return (
    <div ref={containerRef} className="graph-canvas" onMouseDown={onMouseDown}>
      {loading && <div className="pane-state">Loading graph...</div>}
      {!loading && error != null && (
        <div className="pane-state err">
          <pre>{String(error)}</pre>
        </div>
      )}
      {!loading && !error && dot === null && (
        <div className="pane-state muted">No graph data.</div>
      )}
      <div ref={hostRef} className="graph-svg-host" />
      <div className="graph-controls">
        <span className="graph-zoom-pct">{zoomPct}%</span>
        <button onClick={resetView} title="Reset zoom and pan">
          Reset
        </button>
      </div>
    </div>
  );
}
