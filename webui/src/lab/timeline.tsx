import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { SlotRef } from "@/lib/store";
import type { SpeakTrace } from "@/lib/engine/types";

export function Timeline({
  trace,
  revealedCount,
  selected,
  onSelect,
}: {
  trace: SpeakTrace | null;
  revealedCount: number;
  selected: SlotRef | null;
  onSelect: (s: SlotRef) => void;
}) {
  if (!trace) {
    return (
      <div className="rounded-xl bg-surface p-4 text-sm text-subtle shadow-[var(--shadow-border)]">
        Type a sentence and press Speak. Slot count comes from CMUdict (or simple G2P). The picker
        fills each slot; speaker flies render crumbs. Punctuation is silence.
      </div>
    );
  }
  let flat = 0;
  return (
    <div className="flex flex-col gap-3 overflow-x-auto pb-1">
      {trace.tokens.map((tok, ti) => (
        <div key={`${tok.word}-${ti}`} className="flex items-start gap-2">
          <div className="w-24 shrink-0 pt-1">
            <p className="truncate text-xs text-fg">{tok.word}</p>
            <Badge tone={tok.isKnown ? "ok" : "warn"}>{tok.slotCountSource}</Badge>
          </div>
          <div className="flex min-w-0 flex-1 flex-wrap items-center gap-1">
            {tok.slots.map((slot, si) => {
              const idx = flat++;
              const visible = idx < revealedCount;
              const isSel = selected?.token === ti && selected.slot === si;
              const ok = slot.pickerPhone === slot.referencePhone;
              return (
                <button
                  key={si}
                  type="button"
                  disabled={!visible}
                  onClick={() => onSelect({ token: ti, slot: si })}
                  className={cn(
                    "flex min-h-11 min-w-11 flex-col items-center justify-center rounded-md px-2 font-mono text-xs transition-[background-color,color,box-shadow] duration-150",
                    !visible && "opacity-25",
                    isSel && "shadow-[0_0_0_1px_var(--color-kc)]",
                    visible && ok && "bg-ok/15 text-ok",
                    visible && !ok && "bg-warn/15 text-warn",
                  )}
                  title={`picker ${slot.pickerPhone} · ref ${slot.referencePhone}`}
                >
                  <span>{visible ? slot.pickerPhone : "·"}</span>
                  {visible && (
                    <span className="text-2xs text-subtle">{slot.referencePhone}</span>
                  )}
                </button>
              );
            })}
            {tok.pauseS > 0 && (
              <span className="px-2 font-mono text-2xs text-subtle">
                {Math.round(tok.pauseS * 1000)} ms
              </span>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}