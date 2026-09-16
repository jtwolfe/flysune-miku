import { cn } from "@/lib/utils";

export function Badge({
  className,
  tone = "muted",
  ...props
}: React.ComponentProps<"span"> & { tone?: "muted" | "ok" | "warn" | "kc" }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2 py-0.5 font-mono text-2xs font-medium tabular-nums",
        tone === "muted" && "bg-surface-2 text-muted",
        tone === "ok" && "bg-ok/15 text-ok",
        tone === "warn" && "bg-warn/15 text-warn",
        tone === "kc" && "bg-kc/15 text-kc",
        className,
      )}
      {...props}
    />
  );
}
