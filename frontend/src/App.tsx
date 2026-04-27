import { useEffect, useMemo, useRef, useState } from "react";
import { Header } from "@/components/Header";
import { Sidebar } from "@/components/Sidebar";
import { ChatMessage } from "@/components/ChatMessage";
import { ChatInput } from "@/components/ChatInput";
import { EmptyState } from "@/components/EmptyState";
import { StatusIndicator } from "@/components/StatusIndicator";
import { Sheet, SheetContent } from "@/components/ui/sheet";
import { useStreamQuery } from "@/hooks/useStreamQuery";
import { useConversation } from "@/hooks/useConversation";

interface AppProps {
  isDark: boolean;
  onToggleDark: () => void;
}

function App({ isDark, onToggleDark }: AppProps) {
  const conv = useConversation();
  const stream = useStreamQuery();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll on new messages or token arrival
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [conv.messages.length, stream.currentAnswer]);

  const currentTitle = useMemo(() => {
    if (!conv.currentConversationId) return "";
    return conv.conversations.find((c) => c.id === conv.currentConversationId)?.title ?? "";
  }, [conv.conversations, conv.currentConversationId]);

  const handleSend = async (text: string) => {
    if (!conv.currentConversationId) {
      conv.createNew();
    }

    conv.addMessage({ role: "user", content: text });

    // Pass the most recent backend-issued conversation_id so the backend
    // can continue the same thread; null on the very first send.
    const backendConvId = stream.conversationId;

    const result = await stream.send(text, backendConvId);

    if (result) {
      conv.addMessage({
        role: "assistant",
        content:
          result.finalAnswer ||
          (result.blocked ? "Response was blocked by safety filters." : ""),
        citations: result.citations,
        blocked: result.blocked,
      });
    } else if (stream.error) {
      conv.addMessage({
        role: "assistant",
        content: `**Error:** ${stream.error.message}`,
      });
    }
  };

  const handlePickExample = (q: string) => {
    void handleSend(q);
  };

  const handleNew = () => {
    conv.createNew();
    stream.reset();
    setSidebarOpen(false);
  };

  const handleSwitch = (id: string) => {
    conv.switchTo(id);
    stream.reset();
    setSidebarOpen(false);
  };

  return (
    <div className="flex h-full w-full flex-col">
      <Header
        conversationTitle={currentTitle}
        isDark={isDark}
        onToggleDark={onToggleDark}
        onToggleSidebar={() => setSidebarOpen(true)}
      />

      <div className="flex flex-1 overflow-hidden">
        {/* Desktop sidebar */}
        <div className="hidden h-full w-64 shrink-0 border-r border-border md:block">
          <Sidebar
            conversations={conv.conversations}
            currentId={conv.currentConversationId}
            onNew={handleNew}
            onSwitch={handleSwitch}
            onDelete={conv.deleteConversation}
          />
        </div>

        {/* Mobile sidebar in a Sheet */}
        <Sheet open={sidebarOpen} onOpenChange={setSidebarOpen}>
          <SheetContent side="left" className="w-72 p-0">
            <Sidebar
              conversations={conv.conversations}
              currentId={conv.currentConversationId}
              onNew={handleNew}
              onSwitch={handleSwitch}
              onDelete={conv.deleteConversation}
            />
          </SheetContent>
        </Sheet>

        {/* Main column */}
        <main className="flex h-full min-w-0 flex-1 flex-col">
          <div ref={scrollRef} className="flex-1 overflow-y-auto scrollbar-thin">
            {conv.messages.length === 0 && !stream.isStreaming ? (
              <EmptyState onPick={handlePickExample} />
            ) : (
              <div className="mx-auto flex max-w-3xl flex-col gap-6 px-4 py-8 md:px-6">
                {conv.messages.map((m) => (
                  <ChatMessage key={m.id} message={m} isDark={isDark} />
                ))}

                {/* In-flight streaming assistant message */}
                {stream.isStreaming && (
                  <div className="flex flex-col gap-2">
                    <StatusIndicator
                      stage={stream.currentStage}
                      visible={stream.currentAnswer.length === 0}
                    />
                    {(stream.currentAnswer.length > 0 || stream.citations.length > 0) && (
                      <ChatMessage
                        streaming
                        isDark={isDark}
                        message={{
                          id: "__streaming__",
                          role: "assistant",
                          content: stream.currentAnswer,
                          citations: stream.citations,
                          createdAt: Date.now(),
                          blocked: stream.blocked,
                        }}
                      />
                    )}
                  </div>
                )}

                {stream.error && !stream.isStreaming && (
                  <div className="rounded-md border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs text-destructive">
                    {stream.error.message}
                  </div>
                )}
              </div>
            )}
          </div>

          <ChatInput
            onSend={handleSend}
            onStop={stream.reset}
            isStreaming={stream.isStreaming}
          />
        </main>
      </div>
    </div>
  );
}

export default App;
