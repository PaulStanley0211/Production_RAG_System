import { useEffect, useState } from "react";
import { Moon, Sun, PanelLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Separator } from "@/components/ui/separator";
import { checkReady, type ReadyResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

interface HeaderProps {
  conversationTitle?: string;
  isDark: boolean;
  onToggleDark: () => void;
  onToggleSidebar?: () => void;
}

const READY_POLL_MS = 30_000;

export function Header({ conversationTitle, isDark, onToggleDark, onToggleSidebar }: HeaderProps) {
  const [ready, setReady] = useState<ReadyResponse | null>(null);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();

    const tick = async () => {
      const r = await checkReady(controller.signal);
      if (!cancelled) setReady(r);
    };

    tick();
    const id = window.setInterval(tick, READY_POLL_MS);
    return () => {
      cancelled = true;
      controller.abort();
      window.clearInterval(id);
    };
  }, []);

  const dotColor =
    ready?.status === "ok"
      ? "bg-emerald-500"
      : ready?.status === "degraded"
        ? "bg-amber-500"
        : "bg-rose-500";

  const statusLabel =
    ready?.status === "ok"
      ? "All systems operational"
      : ready?.status === "degraded"
        ? "Degraded"
        : "Unreachable";

  return (
    <header className="flex h-14 shrink-0 items-center gap-3 border-b border-border bg-background/80 px-4 backdrop-blur-sm md:px-6">
      <div className="flex items-center gap-3">
        {onToggleSidebar && (
          <Button
            variant="ghost"
            size="icon"
            className="md:hidden"
            onClick={onToggleSidebar}
            aria-label="Open sidebar"
          >
            <PanelLeft className="h-4 w-4" />
          </Button>
        )}
        <span className="font-serif text-lg font-semibold tracking-tight">
          Production RAG
        </span>
      </div>

      <div className="flex flex-1 items-center justify-center px-4">
        <span
          className="truncate text-sm text-muted-foreground"
          title={conversationTitle ?? ""}
        >
          {conversationTitle ?? ""}
        </span>
      </div>

      <div className="flex items-center gap-2">
        <Popover>
          <PopoverTrigger asChild>
            <button
              type="button"
              className="flex items-center gap-2 rounded-full px-2 py-1 text-xs text-muted-foreground hover:bg-secondary"
              aria-label={`Backend status: ${statusLabel}`}
            >
              <span className={cn("h-2.5 w-2.5 rounded-full", dotColor)} />
              <span className="hidden sm:inline">{statusLabel}</span>
            </button>
          </PopoverTrigger>
          <PopoverContent align="end" className="w-72">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span className={cn("h-2.5 w-2.5 rounded-full", dotColor)} />
                <span className="text-sm font-medium">{statusLabel}</span>
              </div>
              <Separator />
              {ready?.components && ready.components.length > 0 ? (
                <ul className="space-y-1.5 text-xs">
                  {ready.components.map((c) => (
                    <li key={`${c.name}-${c.status}`} className="flex items-start justify-between gap-2">
                      <span className="font-mono text-muted-foreground">{c.name}</span>
                      <span
                        className={cn(
                          "ml-auto",
                          c.status === "ok" || c.status === "up"
                            ? "text-emerald-600 dark:text-emerald-400"
                            : c.status === "degraded"
                              ? "text-amber-600 dark:text-amber-400"
                              : "text-rose-600 dark:text-rose-400",
                        )}
                      >
                        {c.status}
                      </span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-xs text-muted-foreground">No component details available.</p>
              )}
            </div>
          </PopoverContent>
        </Popover>

        <Button
          variant="ghost"
          size="icon"
          onClick={onToggleDark}
          aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
        >
          {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </Button>
      </div>
    </header>
  );
}
