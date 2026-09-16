import { cn } from "@/lib/utils";
import type { SlotTrace } from "@/lib/engine/types";
import { PHONEME_INVENTORY } from "@/lib/engine/phonemes";
import { ScrollArea } from "@/components/ui/scroll-area";

export function VoteChoir({
  slot,
  progress,
}: {
  slot: SlotTrace | null;
  progress: number;
}) {
  if (!slot) {
    return (
      <div className="flex h-full min-h-48 items-center justify-center text-sm text-subtle">
        Waiting for a slot
      </div>
    );
  }
  const nShow = Math.max(1, Math.round(slot.votes.length * Math.min(1, Math.max(0.08, progress))));
  const shown = slot.votes.slice(0, nShow);
  const maxAbs = Math.max(0.01, ...shown.map((v) => Math.abs(v.score)));

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2">
      <div className="flex items-baseline justify-between">
        <h3 className="text-2xs font-medium tracking-wide text-muted uppercase">Picker swarm</h3>
        <span className="font-mono text-2xs text-muted">39 flies · softmax</span>
      </div>
      <ScrollArea className="min-h-0 flex-1">
        <ul className="flex flex-col gap-1 pr-2">
          {shown.map((v) => {
            const winner = v.phoneme === slot.pickerPhone;
            const match = v.phoneme === slot.referencePhone;
            const width = `${(Math.abs(v.score) / maxAbs) * 100}%`;
            return (
              <li key={v.phoneme} className="grid grid-cols-[3rem_1fr_auto] items-center gap-2">
                <span
                  className={cn(
                    "font-mono text-xs tabular-nums",
                    winner ? "text-kc" : "text-muted",
                  )}
                >
                  {v.phoneme}
                </span>
                <div className="h-2 overflow-hidden rounded-full bg-surface-2">
                  <div
                    className={cn(
                      "h-full rounded-full",
                      winner ? "bg-kc" : v.isYes ? "bg-primary/50" : "bg-border",
                    )}
                    style={{ width }}
                  />
                </div>
                <span className="w-16 text-right font-mono text-2xs tabular-nums text-subtle">
                  {v.isYes ? "YES" : "NO"}
                  {match ? " · ref" : ""}
                </span>
              </li>
            );
          })}
        </ul>
      </ScrollArea>
      <p className="text-xs text-subtle">
        Winner {slot.pickerPhone} /{PHONEME_INVENTORY[slot.pickerPhone]?.ipa}/ · conf{" "}
        {(slot.confidence * 100).toFixed(0)}%
      </p>
    </div>
  );
}
