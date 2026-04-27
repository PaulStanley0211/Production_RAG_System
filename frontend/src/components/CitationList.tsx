import { FileText } from "lucide-react";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import type { Citation } from "@/hooks/useStreamQuery";

interface CitationListProps {
  citations: Citation[];
}

export function CitationList({ citations }: CitationListProps) {
  if (!citations || citations.length === 0) return null;

  return (
    <div className="mt-3 flex flex-wrap gap-1.5">
      {citations.map((c, i) => (
        <Popover key={`${c.chunk_id}-${i}`}>
          <PopoverTrigger asChild>
            <button
              type="button"
              className="inline-flex max-w-[18rem] items-center gap-1.5 truncate rounded-full border border-accent bg-accent/40 px-2.5 py-1 text-[11px] font-medium text-accent-foreground transition-colors hover:bg-accent"
              aria-label={`Citation ${i + 1}: ${c.source}`}
            >
              <FileText className="h-3 w-3 shrink-0" />
              <span className="truncate">
                <span className="opacity-60 mr-1">{i + 1}.</span>
                {c.source}
              </span>
            </button>
          </PopoverTrigger>
          <PopoverContent className="w-80" align="start">
            <div className="space-y-2">
              <div className="flex items-start justify-between gap-2">
                <p className="break-all text-sm font-medium">{c.source}</p>
                {typeof c.score === "number" && (
                  <span className="shrink-0 rounded bg-secondary px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
                    {c.score.toFixed(3)}
                  </span>
                )}
              </div>
              <p className="font-mono text-[10px] text-muted-foreground">
                chunk {c.chunk_id}
              </p>
              <div className="max-h-60 overflow-auto scrollbar-thin">
                <p className="whitespace-pre-wrap text-xs leading-relaxed text-foreground/90">
                  {c.snippet}
                </p>
              </div>
            </div>
          </PopoverContent>
        </Popover>
      ))}
    </div>
  );
}
