import { cn } from "@/lib/utils";
import type { RevealStage } from "@/lib/store";

const STAGES: { id: RevealStage; label: string; hint: string }[] = [
  { id: "context", label: "Window", hint: "7-letter context" },
  { id: "kc", label: "Kenyon", hint: "sparse expansion" },
  { id: "vote", label: "Choir", hint: "39 YES/NO flies" },
  { id: "speak", label: "Speak", hint: "phoneme id only" },
];

const ORDER: RevealStage[] = ["idle", "context", "kc", "vote", "speak", "done"];

export function StageRail({ stage }: { stage: RevealStage }) {
  const idx = ORDER.indexOf(stage);
  return (
    <ol className="grid grid-cols-4 gap-1 rounded-xl bg-surface p-2 shadow-[var(--shadow-border)]">
      {STAGES.map((s, i) => {
        const active = stage === s.id || (stage === "done" && s.id === "speak");
        const done = idx > ORDER.indexOf(s.id) || stage === "done";
        return (
          <li
            key={s.id}
            className={cn(
              "rounded-md px-2 py-2 text-center transition-[background-color,color] duration-150",
              active && "bg-kc/15 text-kc",
              !active && done && "text-fg",
              !active && !done && "text-subtle",
            )}
          >
            <p className="font-mono text-2xs tracking-wide uppercase">{`0${i + 1}`}</p>
            <p className="text-xs font-medium">{s.label}</p>
            <p className="hidden text-2xs text-subtle sm:block">{s.hint}</p>
          </li>
        );
      })}
    </ol>
  );
}
