import { Button } from "@/components/ui/button";
import { downsamplePeaks, playSamples } from "@/lib/engine/audio";
import { PHONEME_INVENTORY } from "@/lib/engine/phonemes";
import type { SlotTrace } from "@/lib/engine/types";
import { Play } from "lucide-react";
import { useMemo } from "react";

function MiniWave({ samples }: { samples: Float32Array }) {
  const peaks = useMemo(() => downsamplePeaks(samples, 96), [samples]);
  const d = useMemo(() => {
    const w = 96;
    const h = 36;
    let path = "";
    for (let i = 0; i < peaks.length; i++) {
      const x = (i / (peaks.length - 1 || 1)) * w;
      const y = (1 - peaks[i]!) * h * 0.5;
      path += `${i === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)} `;
    }
    for (let i = peaks.length - 1; i >= 0; i--) {
      const x = (i / (peaks.length - 1 || 1)) * w;
      const y = (1 + peaks[i]!) * h * 0.5;
      path += `L${x.toFixed(1)} ${y.toFixed(1)} `;
    }
    return path + "Z";
  }, [peaks]);
  return (
    <svg viewBox="0 0 96 36" className="h-9 w-full text-kc" aria-hidden>
      <path d={d} fill="currentColor" opacity={0.85} />
    </svg>
  );
}

export function SpeakerPanel({ slot }: { slot: SlotTrace | null }) {
  if (!slot) {
    return (
      <div className="rounded-xl bg-surface p-4 shadow-[var(--shadow-border)]">
        <h3 className="text-2xs font-medium tracking-wide text-muted uppercase">Speaker fly</h3>
        <p className="mt-3 text-sm text-subtle">
          After the choir votes, one speaker fly renders a 60–250 ms crumb from F0 / formants / noise.
        </p>
      </div>
    );
  }
  const info = PHONEME_INVENTORY[slot.speaker.phoneme];
  const rows: [string, string][] = [
    ["F0", `${slot.speaker.f0.toFixed(1)} Hz`],
    ["F1×", slot.speaker.f1Shift.toFixed(3)],
    ["F2×", slot.speaker.f2Shift.toFixed(3)],
    ["Noise", slot.speaker.noiseLevel.toFixed(3)],
    ["Dur", `${slot.speaker.durationMs.toFixed(0)} ms`],
    ["Atk", `${slot.speaker.attackMs.toFixed(0)} ms`],
    ["Rel", `${slot.speaker.releaseMs.toFixed(0)} ms`],
    ["Rolloff", slot.speaker.harmonicRolloff.toFixed(2)],
  ];
  return (
    <div className="rounded-xl bg-surface p-4 shadow-[var(--shadow-border)]">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-2xs font-medium tracking-wide text-muted uppercase">Speaker fly</h3>
          <p className="mt-1 font-mono text-lg text-fg">
            /{slot.speaker.phoneme}/{" "}
            <span className="text-sm text-muted">{info?.ipa}</span>
          </p>
          <p className="text-xs text-subtle">{info?.description}</p>
        </div>
        <Button
          size="sm"
          variant="secondary"
          onClick={() => void playSamples(slot.crumb)}
          aria-label="Play this crumb"
        >
          <Play className="size-3.5" />
          Crumb
        </Button>
      </div>
      <div className="mt-3">
        <MiniWave samples={slot.crumb} />
      </div>
      <dl className="mt-3 grid grid-cols-3 gap-2">
        {rows.map(([k, v]) => (
          <div key={k} className="rounded-md bg-surface-2 px-2 py-1.5">
            <dt className="text-2xs tracking-wide text-subtle uppercase">{k}</dt>
            <dd className="font-mono text-xs tabular-nums text-fg">{v}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
