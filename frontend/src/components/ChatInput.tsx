import { ArrowUp, Square } from "lucide-react";
import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

interface ChatInputProps {
  onSend: (text: string) => void;
  onStop?: () => void;
  isStreaming: boolean;
  disabled?: boolean;
}

const LINE_HEIGHT_PX = 22; // ~ 1.4 * 14px (text-sm)
const MAX_LINES = 6;
const MIN_HEIGHT = 44;
const MAX_HEIGHT = MIN_HEIGHT + LINE_HEIGHT_PX * (MAX_LINES - 1);

export function ChatInput({ onSend, onStop, isStreaming, disabled }: ChatInputProps) {
  const [value, setValue] = useState("");
  const taRef = useRef<HTMLTextAreaElement>(null);

  // Auto-grow up to MAX_LINES
  useEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    const next = Math.min(MAX_HEIGHT, Math.max(MIN_HEIGHT, ta.scrollHeight));
    ta.style.height = `${next}px`;
  }, [value]);

  const handleSubmit = () => {
    const trimmed = value.trim();
    if (!trimmed || isStreaming || disabled) return;
    onSend(trimmed);
    setValue("");
    // Reset height after send
    requestAnimationFrame(() => {
      const ta = taRef.current;
      if (ta) ta.style.height = `${MIN_HEIGHT}px`;
    });
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="border-t border-border bg-background px-4 py-3 md:px-6 md:py-4">
      <div className="mx-auto flex max-w-3xl items-end gap-2">
        <div className="relative flex-1">
          <Textarea
            ref={taRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Ask a question about your documents..."
            disabled={isStreaming || disabled}
            rows={1}
            className="min-h-[44px] py-3 pr-12 text-sm leading-snug"
            style={{ height: MIN_HEIGHT, maxHeight: MAX_HEIGHT }}
          />
        </div>
        {isStreaming && onStop ? (
          <Button
            type="button"
            size="icon"
            variant="outline"
            onClick={onStop}
            aria-label="Stop generating"
          >
            <Square className="h-4 w-4" />
          </Button>
        ) : (
          <Button
            type="button"
            size="icon"
            onClick={handleSubmit}
            disabled={!value.trim() || disabled}
            aria-label="Send message"
          >
            <ArrowUp className="h-4 w-4" />
          </Button>
        )}
      </div>
      <p className="mx-auto mt-1.5 max-w-3xl text-[10px] text-muted-foreground">
        Press Enter to send, Shift + Enter for a new line.
      </p>
    </div>
  );
}
