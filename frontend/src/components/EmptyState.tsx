import { Button } from "@/components/ui/button";

interface EmptyStateProps {
  onPick: (q: string) => void;
}

const EXAMPLES = [
  "What is Project Glasswing?",
  "How does Anthropic compare to other AI safety companies?",
  "Summarize the Azure learning guide",
];

export function EmptyState({ onPick }: EmptyStateProps) {
  return (
    <div className="mx-auto flex h-full max-w-2xl flex-col items-center justify-center px-6 py-10 text-center">
      <h1 className="text-balance font-serif text-3xl font-semibold tracking-tight md:text-4xl">
        Ground your answers in your documents
      </h1>
      <p className="mt-4 max-w-xl text-balance text-sm leading-relaxed text-muted-foreground md:text-base">
        Ask a question and get an answer drawn directly from your indexed sources, with citations
        you can verify. The system retrieves passages from your library, evaluates their relevance,
        and writes a grounded response.
      </p>
      <div className="mt-8 flex flex-wrap justify-center gap-2">
        {EXAMPLES.map((q) => (
          <Button
            key={q}
            variant="secondary"
            size="sm"
            className="rounded-full px-3 text-xs font-normal text-muted-foreground hover:text-foreground"
            onClick={() => onPick(q)}
          >
            {q}
          </Button>
        ))}
      </div>
    </div>
  );
}
