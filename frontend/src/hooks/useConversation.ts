import { useCallback, useEffect, useState } from "react";
import type { Citation } from "@/hooks/useStreamQuery";

const STORAGE_KEY = "rag.conversations.v1";
const MESSAGES_KEY_PREFIX = "rag.messages.v1.";

export interface ConversationSummary {
  id: string;
  title: string;
  updatedAt: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  createdAt: number;
  blocked?: boolean;
}

export interface UseConversationReturn {
  currentConversationId: string | null;
  messages: ChatMessage[];
  conversations: ConversationSummary[];
  createNew: () => string;
  switchTo: (id: string) => void;
  deleteConversation: (id: string) => void;
  addMessage: (msg: Omit<ChatMessage, "id" | "createdAt"> & Partial<Pick<ChatMessage, "id" | "createdAt">>) => ChatMessage;
  setConversationId: (id: string) => void;
  setConversationTitle: (id: string, title: string) => void;
}

function loadSummaries(): ConversationSummary[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((c): c is ConversationSummary => {
        const obj = c as Partial<ConversationSummary>;
        return !!obj && typeof obj.id === "string" && typeof obj.title === "string" && typeof obj.updatedAt === "number";
      });
  } catch {
    return [];
  }
}

function saveSummaries(list: ConversationSummary[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
  } catch {
    /* quota or disabled — ignore */
  }
}

function loadMessagesForId(id: string): ChatMessage[] {
  try {
    const raw = localStorage.getItem(MESSAGES_KEY_PREFIX + id);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((m): m is ChatMessage => {
      const obj = m as Partial<ChatMessage>;
      return (
        !!obj &&
        typeof obj.id === "string" &&
        (obj.role === "user" || obj.role === "assistant") &&
        typeof obj.content === "string" &&
        typeof obj.createdAt === "number"
      );
    });
  } catch {
    return [];
  }
}

function saveMessagesForId(id: string, messages: ChatMessage[]) {
  try {
    localStorage.setItem(MESSAGES_KEY_PREFIX + id, JSON.stringify(messages));
  } catch {
    /* quota or disabled — ignore */
  }
}

function clearMessagesForId(id: string) {
  try {
    localStorage.removeItem(MESSAGES_KEY_PREFIX + id);
  } catch {
    /* ignore */
  }
}

function makeLocalId(): string {
  return `conv_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

function makeMessageId(): string {
  return `msg_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

export function useConversation(): UseConversationReturn {
  const [conversations, setConversations] = useState<ConversationSummary[]>(() => loadSummaries());
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);

  // Persist whenever the list changes
  useEffect(() => {
    saveSummaries(conversations);
  }, [conversations]);

  // Persist messages for the current conversation whenever they change
  useEffect(() => {
    if (currentConversationId && messages.length > 0) {
      saveMessagesForId(currentConversationId, messages);
    }
  }, [currentConversationId, messages]);

  const upsertSummary = useCallback((id: string, title: string) => {
    setConversations((prev) => {
      const idx = prev.findIndex((c) => c.id === id);
      const next: ConversationSummary = { id, title, updatedAt: Date.now() };
      if (idx === -1) return [next, ...prev];
      const copy = prev.slice();
      copy[idx] = { ...copy[idx], title, updatedAt: next.updatedAt };
      // Move to top
      copy.sort((a, b) => b.updatedAt - a.updatedAt);
      return copy;
    });
  }, []);

  const createNew = useCallback(() => {
    const id = makeLocalId();
    setCurrentConversationId(id);
    setMessages([]);
    // Don't persist until first message; placeholder title shown in sidebar would be empty.
    return id;
  }, []);

  const switchTo = useCallback((id: string) => {
    setCurrentConversationId(id);
    setMessages(loadMessagesForId(id));
  }, []);

  const deleteConversation = useCallback((id: string) => {
    clearMessagesForId(id);
    setConversations((prev) => prev.filter((c) => c.id !== id));
    setCurrentConversationId((curr) => (curr === id ? null : curr));
    setMessages((prev) => (currentConversationId === id ? [] : prev));
  }, [currentConversationId]);

  const addMessage = useCallback<UseConversationReturn["addMessage"]>(
    (msg) => {
      const full: ChatMessage = {
        id: msg.id ?? makeMessageId(),
        role: msg.role,
        content: msg.content,
        citations: msg.citations,
        createdAt: msg.createdAt ?? Date.now(),
        blocked: msg.blocked,
      };
      setMessages((prev) => [...prev, full]);

      // First user message becomes the title
      if (full.role === "user" && currentConversationId) {
        const existing = conversations.find((c) => c.id === currentConversationId);
        if (!existing) {
          const title = full.content.length > 60 ? `${full.content.slice(0, 57)}...` : full.content;
          upsertSummary(currentConversationId, title);
        } else {
          // bump updatedAt
          upsertSummary(currentConversationId, existing.title);
        }
      }
      return full;
    },
    [conversations, currentConversationId, upsertSummary],
  );

  const setConversationId = useCallback((id: string) => {
    setCurrentConversationId(id);
  }, []);

  const setConversationTitle = useCallback(
    (id: string, title: string) => {
      upsertSummary(id, title);
    },
    [upsertSummary],
  );

  return {
    currentConversationId,
    messages,
    conversations,
    createNew,
    switchTo,
    deleteConversation,
    addMessage,
    setConversationId,
    setConversationTitle,
  };
}
