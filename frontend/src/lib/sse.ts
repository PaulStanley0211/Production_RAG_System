/**
 * Server-Sent Events parser as an async iterator.
 *
 * Reads a fetch ReadableStream<Uint8Array> response body, buffers partial
 * chunks, and yields { event, data } objects as complete SSE events arrive.
 *
 * Handles:
 *  - multi-line `data:` blocks (joined with \n per the spec)
 *  - CRLF or LF line endings
 *  - chunks that split events anywhere
 *  - comments (lines starting with `:`) — ignored
 */

export interface SSEEvent {
  event: string;
  data: string;
  id?: string;
  retry?: number;
}

export async function* parseSSE(
  stream: ReadableStream<Uint8Array>,
  signal?: AbortSignal,
): AsyncGenerator<SSEEvent, void, void> {
  const reader = stream.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  const onAbort = () => {
    reader.cancel().catch(() => {});
  };
  signal?.addEventListener("abort", onAbort, { once: true });

  try {
    while (true) {
      if (signal?.aborted) return;
      const { done, value } = await reader.read();
      if (done) {
        // Flush any final event the server didn't terminate with a blank line.
        if (buffer.trim().length > 0) {
          const evt = parseEventBlock(buffer);
          if (evt) yield evt;
        }
        return;
      }
      buffer += decoder.decode(value, { stream: true });

      // Events are separated by a blank line (\n\n or \r\n\r\n).
      // Normalise CRLF -> LF first.
      buffer = buffer.replace(/\r\n/g, "\n");

      let idx: number;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const evt = parseEventBlock(block);
        if (evt) yield evt;
      }
    }
  } finally {
    signal?.removeEventListener("abort", onAbort);
    try {
      reader.releaseLock();
    } catch {
      /* already released */
    }
  }
}

function parseEventBlock(block: string): SSEEvent | null {
  const lines = block.split("\n");
  let event = "message";
  const dataParts: string[] = [];
  let id: string | undefined;
  let retry: number | undefined;

  for (const rawLine of lines) {
    if (rawLine.length === 0) continue;
    if (rawLine.startsWith(":")) continue; // comment
    const colonIdx = rawLine.indexOf(":");
    let field: string;
    let value: string;
    if (colonIdx === -1) {
      field = rawLine;
      value = "";
    } else {
      field = rawLine.slice(0, colonIdx);
      value = rawLine.slice(colonIdx + 1);
      if (value.startsWith(" ")) value = value.slice(1);
    }
    switch (field) {
      case "event":
        event = value;
        break;
      case "data":
        dataParts.push(value);
        break;
      case "id":
        id = value;
        break;
      case "retry": {
        const n = Number(value);
        if (!Number.isNaN(n)) retry = n;
        break;
      }
      default:
        break;
    }
  }

  if (dataParts.length === 0) return null;
  return { event, data: dataParts.join("\n"), id, retry };
}
