/**
 * Thin API client for the FastAPI RAG backend.
 *
 *   streamQuery(req)  – POST /api/query with stream=true, returns the raw Response
 *                       (caller pipes response.body through parseSSE).
 *   checkReady()      – GET /ready, returns the parsed JSON or a synthetic "down"
 *                       record on network failure.
 */

export const API_BASE: string =
  (import.meta.env.VITE_API_URL as string | undefined) ?? "http://localhost:8000";

export interface QueryRequest {
  query: string;
  conversation_id: string | null;
  stream: boolean;
}

export interface ReadyComponent {
  name: string;
  status: string;
  detail?: string;
}

export interface ReadyResponse {
  status: "ok" | "degraded" | "down";
  components: ReadyComponent[];
}

export async function streamQuery(
  req: QueryRequest,
  signal?: AbortSignal,
): Promise<Response> {
  const res = await fetch(`${API_BASE}/api/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(req),
    signal,
  });
  if (!res.ok) {
    let detail = "";
    try {
      detail = await res.text();
    } catch {
      /* ignore */
    }
    throw new Error(`Query failed: ${res.status} ${res.statusText}${detail ? ` — ${detail}` : ""}`);
  }
  if (!res.body) {
    throw new Error("Query response had no body");
  }
  return res;
}

export async function checkReady(signal?: AbortSignal): Promise<ReadyResponse> {
  try {
    const res = await fetch(`${API_BASE}/ready`, { signal });
    if (!res.ok) {
      return { status: "down", components: [{ name: "http", status: "down", detail: `HTTP ${res.status}` }] };
    }
    const data = (await res.json()) as Partial<ReadyResponse>;
    return {
      status: (data.status as ReadyResponse["status"]) ?? "down",
      components: Array.isArray(data.components) ? data.components : [],
    };
  } catch (err) {
    return {
      status: "down",
      components: [{ name: "network", status: "down", detail: err instanceof Error ? err.message : String(err) }],
    };
  }
}
