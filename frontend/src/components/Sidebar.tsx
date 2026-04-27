import { Plus, Trash2, MessageSquare } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import type { ConversationSummary } from "@/hooks/useConversation";

interface SidebarProps {
  conversations: ConversationSummary[];
  currentId: string | null;
  onNew: () => void;
  onSwitch: (id: string) => void;
  onDelete: (id: string) => void;
}

export function Sidebar({ conversations, currentId, onNew, onSwitch, onDelete }: SidebarProps) {
  return (
    <aside className="flex h-full w-full flex-col bg-background">
      <div className="px-4 pb-3 pt-4">
        <Button onClick={onNew} className="w-full justify-start gap-2" variant="outline">
          <Plus className="h-4 w-4" />
          New conversation
        </Button>
      </div>

      <div className="px-4 pb-2 pt-1">
        <p className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
          Recent
        </p>
      </div>

      <ScrollArea className="flex-1 px-2">
        {conversations.length === 0 ? (
          <p className="px-3 py-4 text-xs text-muted-foreground">
            No past conversations yet.
          </p>
        ) : (
          <ul className="space-y-0.5 pb-4">
            {conversations.map((c) => {
              const active = c.id === currentId;
              return (
                <li key={c.id}>
                  <div
                    className={cn(
                      "group flex items-center gap-2 rounded-md px-2 py-2 text-sm transition-colors",
                      active ? "bg-secondary text-foreground" : "text-muted-foreground hover:bg-secondary/60",
                    )}
                  >
                    <button
                      type="button"
                      onClick={() => onSwitch(c.id)}
                      className="flex flex-1 items-center gap-2 truncate text-left"
                    >
                      <MessageSquare className="h-3.5 w-3.5 shrink-0" />
                      <span className="truncate">{c.title || "Untitled"}</span>
                    </button>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        onDelete(c.id);
                      }}
                      className="rounded p-1 opacity-0 transition-opacity hover:bg-background hover:text-foreground group-hover:opacity-100"
                      aria-label="Delete conversation"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </ScrollArea>

      <div className="border-t border-border px-4 py-3 text-[11px] text-muted-foreground">
        Conversations are kept on this device.
      </div>
    </aside>
  );
}
