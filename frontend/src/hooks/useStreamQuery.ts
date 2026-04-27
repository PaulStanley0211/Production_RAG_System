import { useCallback, useEffect, useRef, useState } from "react";
import { streamQuery } from "@/lib/api";
import { parseSSE } from "@/lib/sse";

export type StreamStage =
  | "input_guard"
  | "memory_rewrite"
  | "cache_lookup"
  | "routing"
  | "crag_running"
  | "generating"
  | "refusing"
  | null;

export interface Citation {
  source: string;
  chunk_id: string;
  score: number;
  snippet: string;
}

export interface StreamError {
  message: string;
  type?: string;
}

export interface UseStreamQueryReturn {
  isStreaming: boolean;
  currentStage: StreamStage;
  currentAnswer: string;
  citations: Citation[];
  conversationId: string | null;
  cacheHit: boolean | null;
  blocked: boolean;
  error: StreamError | null;
  send: (query: string, conversationId: string | null) => Promise<{ conversationId: string | null; finalAnswer: string; citations: Citation[]; blocked: boolean } | null>;
  reset: () => void;
}

export function useStreamQuery(): UseStreamQueryReturn {
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentStage, setCurrentStage] = useState<StreamStage>(null);
  const [currentAnswer, setCurrentAnswer] = useState<string>("");
  const [citations, setCitations] = useState<Citation[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [cacheHit, setCacheHit] = useState<boolean | null>(null);
  const [blocked, setBlocked] = useState<boolean>(false);
  const [error, setError] = useState<StreamError | null>(null);

  const abortRef = useRef<AbortController | null>(null);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setIsStreaming(false);
    setCurrentStage(null);
    setCurrentAnswer("");
    setCitations([]);
    setCacheHit(null);
    setBlocked(false);
    setError(null);
  }, []);

  // Abort on unmount
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  const send = useCallback<UseStreamQueryReturn["send"]>(
    async (query, convId) => {
      // Cancel any in-flight request
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setIsStreaming(true);
      setCurrentStage(null);
      setCurrentAnswer("");
      setCitations([]);
      setCacheHit(null);
      setBlocked(false);
      setError(null);

      let assembledAnswer = "";
      const collectedCitations: Citation[] = [];
      let finalConvId: string | null = convId;
      let finalBlocked = false;

      try {
        const res = await streamQuery(
          { query, conversation_id: convId, stream: true },
          controller.signal,
        );
        if (!res.body) throw new Error("No response body");

        for await (const evt of parseSSE(res.body, controller.signal)) {
          if (controller.signal.aborted) break;
          let payload: unknown;
          try {
            payload = JSON.parse(evt.data);
          } catch {
            continue;
          }

          switch (evt.event) {
            case "status": {
              const stage = (payload as { stage?: string } | null)?.stage;
              if (stage) setCurrentStage(stage as StreamStage);
              break;
            }
            case "citation": {
              const c = payload as Citation;
              collectedCitations.push(c);
              setCitations((prev) => [...prev, c]);
              break;
            }
            case "token": {
              const text = (payload as { text?: string } | null)?.text ?? "";
              if (text) {
                assembledAnswer += text;
                setCurrentAnswer(assembledAnswer);
                // Once tokens arrive we can clear the stage indicator
                setCurrentStage(null);
              }
              break;
            }
            case "redacted": {
              const sanitized = (payload as { sanitized_text?: string } | null)?.sanitized_text;
              if (typeof sanitized === "string") {
                assembledAnswer = sanitized;
                setCurrentAnswer(sanitized);
              }
              break;
            }
            case "done": {
              const data = payload as { conversation_id?: string; cache_hit?: boolean; blocked?: boolean };
              if (typeof data.conversation_id === "string") {
                finalConvId = data.conversation_id;
                setConversationId(data.conversation_id);
              }
              if (typeof data.cache_hit === "boolean") setCacheHit(data.cache_hit);
              if (data.blocked === true) {
                finalBlocked = true;
                setBlocked(true);
              }
              break;
            }
            case "error": {
              const errPayload = payload as { message?: string; type?: string };
              setError({ message: errPayload.message ?? "Stream error", type: errPayload.type });
              break;
            }
            default:
              break;
          }
        }

        return {
          conversationId: finalConvId,
          finalAnswer: assembledAnswer,
          citations: collectedCitations,
          blocked: finalBlocked,
        };
      } catch (err) {
        if (controller.signal.aborted) return null;
        setError({
          message: err instanceof Error ? err.message : String(err),
          type: "network",
        });
        return null;
      } finally {
        if (abortRef.current === controller) {
          abortRef.current = null;
        }
        setIsStreaming(false);
      }
    },
    [],
  );

  return {
    isStreaming,
    currentStage,
    currentAnswer,
    citations,
    conversationId,
    cacheHit,
    blocked,
    error,
    send,
    reset,
  };
}
