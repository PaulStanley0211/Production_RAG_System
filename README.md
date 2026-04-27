# Production RAG System

A retrieval-augmented generation system with hybrid search, agentic
self-correction, three-layer security, and end-to-end evaluation.

Built as a portfolio piece to demonstrate production patterns beyond the
typical RAG tutorial — corrective retrieval (CRAG), prompt injection
defense, PII redaction, and a measurable evaluation harness.

---

## Highlights

- **Hybrid retrieval** — dense (BGE) + sparse (BM25) with Reciprocal Rank
  Fusion, followed by cross-encoder reranking
- **Agentic self-correction** — CRAG loop that detects weak retrieval and
  decomposes the query into sub-queries, with optional web fallback
- **Three-layer security** — input guard against prompt injection, content
  filter on retrieved chunks, output guard for PII redaction
- **Streaming API** — Server-Sent Events with progressive token, citation,
  and status events
- **Reproducible evaluation** — 30-question retrieval suite and 15-question
  end-to-end suite with LLM-as-judge for faithfulness and relevance.
  Full results in [Results.md](./Results.md)

## Evaluation summary

| Metric | Value |
|---|---|
| Retrieval Hit Rate@5 | **0.96** |
| Retrieval MRR | **0.96** |
| Generation Faithfulness | **0.944** |
| Answer Relevance | **1.000** |
| Behavior Accuracy | **1.000** |

Methodology, per-category breakdowns, and reproduction commands in
[Results.md](./Results.md).

---

## Architecture

```
                     ┌──────────────┐
   User query ──────▶│ Input guard  │  Pattern-based prompt injection check
                     └──────┬───────┘
                            ▼
                     ┌──────────────┐
                     │ Memory       │  Conversation-aware query rewriting
                     │ rewrite      │  (resolves "what about it?")
                     └──────┬───────┘
                            ▼
                     ┌──────────────┐
                     │ Semantic     │  Returns instantly on similar past query
                     │ cache        │  (cosine ≥ 0.92)
                     └──────┬───────┘
                            ▼
                     ┌──────────────┐
                     │ Query router │  factual / comparative / chitchat /
                     └──────┬───────┘  no_retrieval
                            ▼
                     ┌──────────────┐
                     │ CRAG agent   │  Adaptive retrieve → grade → decide:
                     │              │   • generate
                     │              │   • decompose + retry
                     │              │   • web fallback
                     │              │   • refuse
                     └──────┬───────┘
                            ▼
                     ┌──────────────┐
                     │ Content      │  Strip instruction-like patterns
                     │ filter       │  from retrieved chunks
                     └──────┬───────┘
                            ▼
                     ┌──────────────┐
                     │ Generate     │  Sonnet 4.6 with [#N] citations,
                     │ (Sonnet)     │  enforced by prompt
                     └──────┬───────┘
                            ▼
                     ┌──────────────┐
                     │ Output guard │  Email / phone / SSN / Luhn-validated
                     └──────┬───────┘  credit card redaction
                            ▼
                       Streaming response
                       (SSE: token / citation / status / done)
```

## Tech stack

| Layer | Choice |
|---|---|
| API | FastAPI (async) + sse-starlette |
| Vector store | Qdrant 1.12 (named vectors: dense + sparse) |
| Cache & state | Redis 7.4 |
| Embeddings | `BAAI/bge-small-en-v1.5` (dense), `Qdrant/bm25` (sparse) |
| Reranker | `Xenova/ms-marco-MiniLM-L-6-v2` (cross-encoder) |
| LLM | Claude Sonnet 4.6 (generation), Claude Haiku 4.5 (routing/grading/judging) |
| Frontend | React + TypeScript + Vite + Tailwind + shadcn/ui |
| Container | Docker + docker-compose |
| Eval | Custom harness (Hit Rate / MRR / nDCG, LLM-as-judge) |

The tiered model strategy — Sonnet only for final generation, Haiku for
routing, grading, decomposition, and judging — keeps a typical query in
the low-single-cent range in API cost.

---

## Repository layout

```
.
├── app/                  FastAPI runtime — pipeline, services, agents, security
│   ├── routes/           HTTP endpoints (/api/query, /api/search, /health)
│   ├── services/         Cache, memory, router, decomposer, grader, pipeline
│   ├── agents/           CRAG self-correction loop + tool wrappers
│   ├── retrieval/        Hybrid retriever, reranker, filters
│   ├── prompts/          Centralized prompt registry (templates + grading)
│   └── security/         Three guard layers
├── pipeline/             Offline ingest (PDF, HTML, DOCX, text) + chunking + embedding
├── frontend/             React chat UI with SSE streaming
├── eval/                 Evaluation harness — datasets, metrics, runners, CLI
├── test-docs/            Sample corpus for the demo
├── docker-compose.yml    Dev stack (app, qdrant, redis)
├── pyproject.toml        Python dependencies (uv-managed)
├── README.md             You are here
└── Results.md            Detailed evaluation results
```

---

## Quickstart

### Prerequisites
- Docker Desktop
- Python 3.11+ with [uv](https://docs.astral.sh/uv/)
- Node.js 20+ (for the frontend)
- An Anthropic API key

### 1. Clone and configure

```bash
git clone https://github.com/PaulStanley0211/Production_RAG_System.git
cd Production_RAG_System
```

Create a `.env` in the repo root with at minimum:

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

All other settings have sensible defaults — see [Configuration](#configuration)
for what you can override.

### 2. Start the backend stack

```bash
docker compose up -d
```

This starts three containers:
- `rag-app` (FastAPI on `:8000`)
- `rag-qdrant` (vector store on `:6333`)
- `rag-redis` (cache on `:6379`)

Verify with `curl http://localhost:8000/ready`.

### 3. Ingest documents

```bash
uv sync --extra pipeline
python -m pipeline.ingest --source test-docs/
```

This extracts, chunks, embeds, and indexes everything in `test-docs/`.

### 4. Try a query

**Non-streaming:**

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Project Glasswing?"}'
```

**Streaming (SSE):**

```bash
curl -N -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Project Glasswing?", "stream": true}'
```

### 5. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>.

---

## Running the evaluation

The eval harness produces reproducible numbers for retrieval and generation
quality. See [Results.md](./Results.md) for the full methodology.

```bash
# Both suites (~5 minutes)
python -m eval.run

# Retrieval only (~30 seconds, no LLM cost)
python -m eval.run --suite retrieval

# Generation only (~3-5 minutes, uses Haiku as judge)
python -m eval.run --suite generation
```

Reports are written to `eval/reports/` with timestamps for run-to-run
comparison.

For component-level analysis, an ablation runner exercises each major
stage in isolation:

```bash
python -m eval.runners.ablation_runner
```

This walks through six configurations (baseline, no-reranker, no-CRAG,
no-content-filter, dense-only, sparse-only), recreating the app container
between each so the env actually changes, and writes a side-by-side
comparison report.

---

## Configuration

All configuration is environment-driven via `.env`. Key settings:

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | _(required)_ | Claude API access |
| `MODEL_GENERATION` | `claude-sonnet-4-6` | Final answer model |
| `MODEL_ROUTING` | `claude-haiku-4-5-20251001` | Routing/grading/judging |
| `RETRIEVAL_TOP_K_DENSE` | `20` | Dense candidates |
| `RETRIEVAL_TOP_K_SPARSE` | `20` | Sparse candidates |
| `RETRIEVAL_TOP_K_RERANKED` | `5` | Final top-K after rerank |
| `SEMANTIC_CACHE_THRESHOLD` | `0.92` | Cosine threshold for cache hits |
| `CRAG_MAX_ITERATIONS` | `3` | CRAG self-correction retry cap |
| `ENABLE_WEB_FALLBACK` | `false` | Allow web search when corpus fails |
| `GUARD_DENY_ON_INJECTION` | `true` | Block detected prompt injection |
| `GUARD_PII_REDACTION` | `true` | Redact PII from output |

The codebase also exposes ablation flags (`DISABLE_RERANKER`, `DISABLE_CRAG`,
`DISABLE_CONTENT_FILTER`, `RETRIEVAL_MODE`) wired through every layer for
component-level evaluation. Note that `docker compose restart` does not
re-read `.env`; recreate the container with `docker compose up -d
--force-recreate app` after editing.

---

## API reference

### `POST /api/query`

Run a query through the full pipeline.

```json
{
  "query": "your question",
  "conversation_id": "optional-uuid",
  "stream": false
}
```

When `stream: true`, the response is `text/event-stream` with these events:
- `status` — pipeline stage progress
- `citation` — source chunk reference
- `token` — incremental answer text
- `redacted` — fired if PII was caught post-stream
- `done` — final summary
- `error` — graceful failure

### `GET /api/search?q=...&top_k=5`

Debug endpoint for raw retrieval — returns hybrid retrieval + rerank
results without generation. Useful for tuning.

### `GET /health` and `GET /ready`

Liveness and readiness probes.

Full OpenAPI docs at <http://localhost:8000/docs>.

---

## Design notes

A few decisions worth calling out:

- **Hybrid retrieval over dense-only.** Real queries mix semantic intent
  ("how do I deploy this") with literal keywords ("PROD-2024-471"). Dense
  alone misses keywords; sparse alone misses synonyms. Hybrid catches both.
- **Cross-encoder reranking after RRF.** RRF fuses ranked lists by
  position; it never reads the documents. The cross-encoder reads each
  (query, chunk) pair together and produces a true relevance judgment.
  Retrieval finds 20 candidates fast; rerank picks the 5 that actually
  answer.
- **Three-way grading (relevant / partial / irrelevant).** The middle
  category is what makes CRAG work — partial matches trigger query
  decomposition rather than a blanket refusal.
- **Output guard runs post-stream.** PII patterns (`john@doe.com`) can
  span multiple tokens during streaming. Raw tokens are emitted, the
  assembled answer is checked, and a `redacted` event fires if anything
  matched. The frontend can replace the displayed text with the sanitized
  version.

---

## What's not included

This is a portfolio system, not a productionized SaaS. Things deliberately
out of scope:

- **Authentication and rate limiting.** The API is open by design for the demo.
- **Multi-tenancy.** A single Qdrant collection, no per-user isolation.
- **Distributed deployment.** Single-node docker-compose; no Kubernetes manifests.
- **Observability dashboards.** Logging is structured but no Prometheus or OpenTelemetry wiring.
- **A statistically large eval set.** 30 + 15 cases is enough to be
  directionally informative; production deployment would expand to
  hundreds.

The architecture supports all of the above — they're additions, not
rewrites.

---

## License

MIT
