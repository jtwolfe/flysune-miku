import { downsamplePeaks } from "@/lib/engine/audio";
import type { SpeakTrace } from "@/lib/engine/types";
import { useMemo } from "react";

export function Waveform({
  trace,
  progress,
  onSeek,
}: {
  trace: SpeakTrace | null;
  progress: number;
  onSeek?: (t: number) => void;
}) {
  const peaks = useMemo(
    () => (trace ? downsamplePeaks(trace.audio, 240) : new Float32Array(240)),
    [trace],
  );
  const d = useMemo(() => {
    const w = 240;
    const h = 48;
    let path = "";
    for (let i = 0; i < peaks.length; i++) {
      const x = (i / (peaks.length - 1 || 1)) * w;
      const y = 24 - peaks[i]! * 22;
      path += `${i === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)} `;
    }
    for (let i = peaks.length - 1; i >= 0; i--) {
      const x = (i / (peaks.length - 1 || 1)) * w;
      const y = 24 + peaks[i]! * 22;
      path += `L${x.toFixed(1)} ${y.toFixed(1)} `;
    }
    return path + "Z";
  }, [peaks]);
  const dur = trace ? trace.audio.length / trace.sampleRate : 0;

  return (
    <div className="rounded-xl bg-surface p-4 shadow-[var(--shadow-border)]">
      <div className="flex items-baseline justify-between">
        <h3 className="text-2xs font-medium tracking-wide text-muted uppercase">Sentence waveform</h3>
        <span className="font-mono text-2xs tabular-nums text-muted">
          {dur.toFixed(2)} s · {trace?.sampleRate ?? 22050} Hz
        </span>
      </div>
      <button
        type="button"
        className="relative mt-2 block w-full text-left"
        disabled={!trace || !onSeek}
        onClick={(e) => {
          if (!trace || !onSeek) return;
          const rect = e.currentTarget.getBoundingClientRect();
          const t = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
          onSeek(t);
        }}
        aria-label="Seek waveform to a phoneme slot"
      >
        <svg viewBox="0 0 240 48" className="h-12 w-full text-primary/70" aria-hidden>
          <path d={d} fill="currentColor" />
        </svg>
        {trace && (
          <div
            className="pointer-events-none absolute top-0 h-full w-px bg-kc"
            style={{ left: `${Math.min(100, progress * 100)}%` }}
          />
        )}
      </button>
    </div>
  );
}