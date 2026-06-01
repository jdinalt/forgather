import { useEffect, useMemo, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import type { DiLoCoStatsRecord } from "../api";

/** Build uPlot's column-major data: [x=step, train_loss, eval_loss].
 *  Records are assumed sorted by step (the server appends in order). */
function buildData(records: DiLoCoStatsRecord[]): uPlot.AlignedData {
  const xs: number[] = [];
  const train: (number | null)[] = [];
  const evals: (number | null)[] = [];
  for (const r of records) {
    if (typeof r.global_step !== "number") continue;
    xs.push(r.global_step);
    train.push(typeof r.train_loss === "number" ? r.train_loss : null);
    evals.push(typeof r.eval_loss === "number" ? r.eval_loss : null);
  }
  return [xs, train, evals];
}

/** X-axis pan/zoom/reset: wheel zooms toward the cursor, drag pans, and a
 *  double-click resets to the full data range. uPlot's own box-zoom is
 *  disabled (cursor.drag.x=false) so drag means pan. */
function panZoomPlugin(): uPlot.Plugin {
  return {
    hooks: {
      ready: (u: uPlot) => {
        const over = u.over;

        const fullRange = (): [number, number] => {
          const xs = u.data[0];
          return xs.length ? [xs[0] as number, xs[xs.length - 1] as number] : [0, 1];
        };

        const mark = (z: boolean) => {
          (u as unknown as { _userZoomed?: boolean })._userZoomed = z;
        };

        over.addEventListener(
          "wheel",
          (e: WheelEvent) => {
            e.preventDefault();
            const rect = over.getBoundingClientRect();
            const leftPx = e.clientX - rect.left;
            const xVal = u.posToVal(leftPx, "x");
            const min = u.scales.x.min ?? fullRange()[0];
            const max = u.scales.x.max ?? fullRange()[1];
            const factor = e.deltaY < 0 ? 0.8 : 1.25; // up = zoom in
            const nMin = xVal - (xVal - min) * factor;
            const nMax = xVal + (max - xVal) * factor;
            mark(true);
            u.setScale("x", { min: nMin, max: nMax });
          },
          { passive: false },
        );

        let panning = false;
        let startClientX = 0;
        let startMin = 0;
        let startMax = 0;
        over.addEventListener("mousedown", (e: MouseEvent) => {
          panning = true;
          startClientX = e.clientX;
          startMin = u.scales.x.min ?? fullRange()[0];
          startMax = u.scales.x.max ?? fullRange()[1];
        });
        const onMove = (e: MouseEvent) => {
          if (!panning) return;
          const dxPx = e.clientX - startClientX;
          // A bare click (no real movement) must not disable live-follow —
          // only an actual drag counts as the user taking over the view.
          if (Math.abs(dxPx) < 3) return;
          mark(true);
          const rect = over.getBoundingClientRect();
          const perPx = (startMax - startMin) / Math.max(1, rect.width);
          const dx = dxPx * perPx;
          u.setScale("x", { min: startMin - dx, max: startMax - dx });
        };
        const onUp = () => {
          panning = false;
        };
        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup", onUp);

        over.addEventListener("dblclick", () => {
          // Re-autoscale to the full data range and resume live-following.
          mark(false);
          u.setData(u.data);
        });

        // Clean up the window-level listeners with the plot.
        (u as unknown as { _panzoomCleanup?: () => void })._panzoomCleanup =
          () => {
            window.removeEventListener("mousemove", onMove);
            window.removeEventListener("mouseup", onUp);
          };
      },
    },
  };
}

/** Embedded train/eval loss curve for the DiLoCo server view. */
export default function LossChart({
  records,
  height = 240,
}: {
  records: DiLoCoStatsRecord[];
  height?: number;
}) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const plotRef = useRef<uPlot | null>(null);
  const data = useMemo(() => buildData(records), [records]);

  // Create the plot once; size it to the container.
  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const opts: uPlot.Options = {
      width: wrap.clientWidth || 600,
      height,
      // Loss curves: a private CA / muted palette consistent with the panel.
      scales: { x: { time: false } },
      legend: { show: true },
      cursor: { drag: { x: false, y: false } },
      series: [
        { label: "step" },
        {
          label: "train",
          stroke: "#7aa2f7",
          width: 1.5,
          spanGaps: true,
        },
        {
          label: "eval",
          stroke: "#ff9e64",
          width: 1.5,
          spanGaps: true,
        },
      ],
      axes: [
        { stroke: "#9aa5ce", grid: { stroke: "#2b2f44" } },
        { stroke: "#9aa5ce", grid: { stroke: "#2b2f44" } },
      ],
      plugins: [panZoomPlugin()],
    };
    const u = new uPlot(opts, data, wrap);
    plotRef.current = u;

    const ro = new ResizeObserver(() => {
      u.setSize({ width: wrap.clientWidth || 600, height });
    });
    ro.observe(wrap);

    return () => {
      ro.disconnect();
      (u as unknown as { _panzoomCleanup?: () => void })._panzoomCleanup?.();
      u.destroy();
      plotRef.current = null;
    };
    // Recreate only on height change; data updates are handled below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [height]);

  // Push new data without recreating the plot. Always use the redrawing
  // setData (resetScales=true) so new points actually paint — setData(.,false)
  // updates the buffer but doesn't reliably repaint until the next user event.
  // When the user has panned/zoomed, restore their x-range right after so the
  // live curve keeps extending without yanking their view. Double-click clears
  // the flag and resumes full live-follow.
  useEffect(() => {
    const u = plotRef.current;
    if (!u) return;
    const zoomed = !!(u as unknown as { _userZoomed?: boolean })._userZoomed;
    const xMin = u.scales.x.min;
    const xMax = u.scales.x.max;
    u.setData(data, true);
    if (zoomed && xMin != null && xMax != null) {
      u.setScale("x", { min: xMin, max: xMax });
    }
  }, [data]);

  return (
    <div
      ref={wrapRef}
      style={{ width: "100%" }}
      title="Scroll to zoom · drag to pan · double-click to reset"
    />
  );
}
