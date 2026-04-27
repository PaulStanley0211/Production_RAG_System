# Production RAG — Frontend

A React + TypeScript + Vite client for the Production RAG FastAPI backend.

## Stack

- Vite + React 18 + TypeScript
- Tailwind CSS (with shadcn/ui style primitives)
- `react-markdown` + `remark-gfm` for assistant answers
- `react-syntax-highlighter` (light build) for code blocks
- `lucide-react` icons
- Native `fetch` + `ReadableStream` to consume Server-Sent Events

## Prerequisites

- Node.js 18.18+ (tested on Node 24)
- The RAG backend running on `http://localhost:8000` (or set `VITE_API_URL` to wherever it lives)

## Quick start

```bash
cd frontend
npm install
cp .env.example .env       # optional — override VITE_API_URL here
npm run dev
```

Then open `http://localhost:5173/`.

## Environment

| Variable        | Default                  | Notes                                              |
| --------------- | ------------------------ | -------------------------------------------------- |
| `VITE_API_URL`  | `http://localhost:8000`  | Base URL of the FastAPI backend (no trailing `/`). |

The dev server listens on port `5173` (matches the backend's `CORS_ORIGINS`).

## Scripts

| Command           | What it does                                  |
| ----------------- | --------------------------------------------- |
| `npm run dev`     | Vite dev server with HMR                      |
| `npm run build`   | Type-check (`tsc -b`) + production bundle     |
| `npm run preview` | Preview the built bundle                      |
| `npm run lint`    | Run ESLint                                    |

## Backend contract

This client expects:

- `POST /api/query` with body `{ query, conversation_id, stream: true }` returning a `text/event-stream`.
  - SSE events: `status`, `citation`, `token`, `redacted`, `done`, `error`.
- `GET /ready` returning `{ status: "ok" | "degraded" | "down", components: [...] }`.

See `src/lib/api.ts` and `src/hooks/useStreamQuery.ts` for the full event handling.

## Project layout

```
src/
  components/      # Header, Sidebar, ChatMessage, ChatInput, EmptyState, StatusIndicator, CitationList
    ui/            # shadcn-style primitives (button, card, input, textarea, badge, sheet, popover, ...)
  hooks/
    useStreamQuery.ts   # SSE consumer
    useConversation.ts  # local conversation list (localStorage)
  lib/
    api.ts         # fetch wrappers
    sse.ts         # SSE async-iterator parser
    utils.ts       # cn() helper
  App.tsx          # layout: sidebar + chat column
  main.tsx         # dark-mode bootstrap + render
  index.css        # Tailwind + theme variables + prose styling
```
