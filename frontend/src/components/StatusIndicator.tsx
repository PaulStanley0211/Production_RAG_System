import type { StreamStage } from "@/hooks/useStreamQuery";

const LABELS: Record<NonNullable<StreamStage>, string> = {
  input_guard: "Checking input...",
  memory_rewrite: "Recalling context...",
  cache_lookup: "Looking up cache...",
  routing: "Routing question...",
  crag_running: "Searching documents...",
  generating: "Generating answer...",
  refusing: "Considering safety...",
};

interface StatusIndicatorProps {
  stage: StreamStage;
  visible: boolean;
}

export function StatusIndicator({ stage, visible }: StatusIndicatorProps) {
  if (!visible || !stage) return null;
  const label = LABELS[stage];
  if (!label) return null;
  return (
    <div className="flex items-center gap-2 text-xs text-muted-foreground" aria-live="polite">
      <span className="relative flex h-1.5 w-1.5">
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-muted-foreground/40" />
        <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-muted-foreground/80" />
      </span>
      <span>{label}</span>
    </div>
  );
}
