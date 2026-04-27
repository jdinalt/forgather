import { useEffect, useLayoutEffect, useRef, useState } from "react";

interface Props {
  /** Anchor point in viewport coordinates (typically the cursor). */
  x: number;
  y: number;
  onClose: () => void;
  children: React.ReactNode;
}

/** Lightweight floating context menu.
 *
 *  Closes on outside click, Escape, or losing focus. Position is clamped to
 *  the viewport so menus opened near the bottom/right edge don't get
 *  clipped — measured after first paint with useLayoutEffect so we can
 *  shift up/left if needed before the browser commits the frame. */
export function ContextMenu({ x, y, onClose, children }: Props) {
  const menuRef = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ left: number; top: number }>({
    left: x,
    top: y,
  });

  useLayoutEffect(() => {
    const el = menuRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    let left = x;
    let top = y;
    if (left + r.width > window.innerWidth) left = window.innerWidth - r.width - 4;
    if (top + r.height > window.innerHeight) top = window.innerHeight - r.height - 4;
    if (left < 0) left = 0;
    if (top < 0) top = 0;
    setPos({ left, top });
  }, [x, y]);

  useEffect(() => {
    const onDocDown = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("mousedown", onDocDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDocDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [onClose]);

  return (
    <div
      ref={menuRef}
      className="context-menu"
      style={{ position: "fixed", left: pos.left, top: pos.top, zIndex: 1000 }}
      role="menu"
      // Suppress nested context menus opened inside this menu (rare but
      // possible in some browsers when right-clicking quickly).
      onContextMenu={(e) => e.preventDefault()}
    >
      {children}
    </div>
  );
}
