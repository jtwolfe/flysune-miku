import * as SwitchPrimitive from "@radix-ui/react-switch";
import { cn } from "@/lib/utils";

export function Switch({
  className,
  ...props
}: React.ComponentProps<typeof SwitchPrimitive.Root>) {
  return (
    <SwitchPrimitive.Root
      className={cn(
        "peer inline-flex h-6 w-10 shrink-0 items-center rounded-full bg-surface-2 shadow-[var(--shadow-border)] transition-[background-color] duration-150 ease-out data-[state=checked]:bg-kc/40",
        className,
      )}
      {...props}
    >
      <SwitchPrimitive.Thumb className="pointer-events-none block size-5 translate-x-0.5 rounded-full bg-primary transition-transform duration-150 ease-out data-[state=checked]:translate-x-[18px]" />
    </SwitchPrimitive.Root>
  );
}
