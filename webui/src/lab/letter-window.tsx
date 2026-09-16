import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { SlotTrace, TokenTrace } from "@/lib/engine/types";
import { PHONEME_INVENTORY } from "@/lib/engine/phonemes";

export function LetterWindow({
  slot,
  token,
}: {
  slot: SlotTrace | null;
  token: TokenTrace | null;
}) {
  if (!slot) {
    return (
      <div className="rounded-xl bg-surface p-4 shadow-[var(--shadow-border)]">
        <h3 className="text-2xs font-medium tracking-wide text-muted uppercase">Letter context</h3>
        <p className="mt-3 text-sm text-subtle">The 7-character window appears when a slot is live.</p>
      </div>
    );
  }
  const chars = [...slot.letterContext];
  const center = Math.floor(chars.length / 2);
  const ok = slot.pickerPhone === slot.referencePhone;
  const pick = PHONEME_INVENTORY[slot.pickerPhone];
  const ref = PHONEME_INVENTORY[slot.referencePhone];
  return (
    <div className="rounded-xl bg-surface p-4 shadow-[var(--shadow-border)]">
      <div className="flex items-baseline justify-between gap-2">
        <h3 className="text-2xs font-medium tracking-wide text-muted uppercase">Letter context</h3>
        <span className="font-mono text-2xs tabular-nums text-muted">
          {token?.word ?? ""} · slot {slot.index + 1}/{slot.nPhonemes} · pos {slot.phonemePos.toFixed(2)}
        </span>
      </div>
      <div className="mt-3 flex justify-center gap-1">
        {chars.map((c, i) => (
          <span
            key={`${c}-${i}`}
            className={cn(
              "flex size-10 items-center justify-center rounded-md font-mono text-lg",
              i === center
                ? "bg-kc/20 text-kc shadow-[var(--shadow-border)]"
                : "bg-surface-2 text-muted",
            )}
          >
            {c === "_" ? "·" : c}
          </span>
        ))}
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
        <span className="text-subtle">Picker</span>
        <Badge tone={ok ? "ok" : "warn"}>
          {slot.pickerPhone} /{pick?.ipa}/
        </Badge>
        <span className="text-subtle">Reference</span>
        <Badge>
          {slot.referencePhone} /{ref?.ipa}/
        </Badge>
        <span className="text-subtle">Prev cue</span>
        <span className="font-mono text-fg">{slot.previousPhone ?? "none"}</span>
        {token && (
          <Badge tone={token.isKnown ? "ok" : "warn"}>{token.slotCountSource}</Badge>
        )}
      </div>
    </div>
  );
}