import { useEffect, useRef } from "react";
import type { SlotTrace } from "@/lib/engine/types";

function token(name: string, fallback: string) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

export function KcGrid({
  slot,
  progress,
}: {
  slot: SlotTrace | null;
  progress: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const w = canvas.width;
    const h = canvas.height;
    ctx.fillStyle = token("--color-surface-2", "#1b201d");
    ctx.fillRect(0, 0, w, h);
    if (!slot) return;
    const n = slot.nKc;
    const side = Math.ceil(Math.sqrt(n));
    const cell = Math.min(w, h) / side;
    const shown = Math.floor(slot.kcActive.length * Math.min(1, Math.max(0, progress)));
    const active = new Set(slot.kcActive.slice(0, shown));
    const on = token("--color-kc", "#6ee7b7");
    const off = token("--color-border", "#2a322e");
    for (let i = 0; i < n; i++) {
      const x = (i % side) * cell;
      const y = Math.floor(i / side) * cell;
      if (active.has(i)) {
        ctx.fillStyle = on;
        ctx.globalAlpha = 0.92;
      } else {
        ctx.fillStyle = off;
        ctx.globalAlpha = 0.7;
      }
      ctx.fillRect(x + 0.4, y + 0.4, Math.max(0.6, cell - 0.8), Math.max(0.6, cell - 0.8));
    }
    ctx.globalAlpha = 1;
  }, [slot, progress]);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between">
        <h3 className="text-2xs font-medium tracking-wide text-muted uppercase">Kenyon cells</h3>
        <span className="font-mono text-xs tabular-nums text-kc">
          {slot ? `${slot.nKcActive}/${slot.nKc}` : "—"}
        </span>
      </div>
      <canvas
        ref={canvasRef}
        width={280}
        height={280}
        className="h-auto w-full rounded-md bg-surface-2"
        aria-label="Kenyon cell activity grid"
      />
      <p className="text-xs text-subtle">
        Sparse expansion. About 10% fire. Shared across all 39 specialists — speakers never see this.
      </p>
    </div>
  );
}
