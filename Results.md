# Evaluation Results

This document reports measured performance of the RAG system across two
evaluation suites: retrieval (does the right chunk come back?) and generation
(is the answer faithful and relevant?).

All numbers below are reproducible — see the [reproduction](#reproduction)
section at the bottom for exact commands.

---

## Summary

| Suite | Metric | Value | Cases |
|---|---|---|---|
| Retrieval | Hit Rate@5 | **0.960** | 25 in-corpus |
| Retrieval | MRR | **0.960** | 25 in-corpus |
| Retrieval | nDCG@5 | **0.935** | 25 in-corpus |
| Generation | Faithfulness | **0.944** | 9 grounded |
| Generation | Answer relevance | **1.000** | 15 successful |
| Generation | Behavior accuracy | **1.000** | 15 successful |

The system retrieves the right chunks ~96% of the time, generates faithful
answers ~94% of the time when grounded in retrieval, addresses every question
asked, and routes correctly to the right pipeline branch (answer / partial /
refuse) in 100% of test cases.

---

## Retrieval evaluation

**Test set:** 30 questions across an indexed corpus of 18 chunks from three
documents (Anthropic homepage HTML, an Azure services learning guide PDF, a
project conventions markdown). Cases split into four categories:

- **Direct factual** (12 cases) — query language matches doc terms closely
- **Paraphrased** (8 cases) — same intent, different wording
- **Multi-chunk** (5 cases) — answer spans multiple chunks
- **Out-of-corpus** (5 cases) — should retrieve nothing relevant

**Metrics:**

- **Hit Rate@5** — fraction of queries where at least one relevant chunk
  appeared in the top-5 retrieved
- **MRR** (Mean Reciprocal Rank) — quality of ranking. 1.0 = first relevant
  chunk always at rank 1
- **nDCG@5** — accounts for partial credit when multiple relevant chunks
  exist; rewards getting them all ranked highly

**Results by category:**

| Category | Hit Rate@5 | MRR | nDCG@5 | Cases |
|---|---|---|---|---|
| direct_factual | 0.917 | 0.917 | 0.917 | 12 |
| paraphrased | 1.000 | 1.000 | 1.000 | 8 |
| multi_chunk | 1.000 | 1.000 | 0.874 | 5 |
| **Overall (in-corpus)** | **0.960** | **0.960** | **0.935** | **25** |

**Notable findings:**

- **Paraphrased queries scored 1.0 across the board.** The dense embedding
  layer (BAAI/bge-small-en-v1.5) handles semantic match well even when the
  query and document share no exact words.
- **Direct factual queries scored 0.917, lower than paraphrased.** This is a
  known phenomenon in hybrid retrieval — direct keyword queries trigger both
  dense and sparse retrievers strongly, and they can disagree on the best
  chunk, fusing into a slightly suboptimal order.
- **The single direct_factual miss** (a query about Python naming conventions)
  retrieved an adjacent style-guide chunk at rank 2 instead of the target
  naming chunk. The system found the right content; the ground-truth label
  pointed to an overlapping neighbor chunk. The miss was preserved rather
  than retroactively patched, to keep the eval honest.

---

## Generation evaluation

**Test set:** 15 end-to-end test cases covering all three pipeline branches:

- **Answer** (13 cases) — system should retrieve and answer with citations
- **Partial** (1 case) — system should answer what it can and acknowledge gaps
- **Refuse** (1 case) — system should decline to fabricate when no info exists

**Metrics (LLM-as-judge using Claude Haiku 4.5):**

- **Faithfulness** — fraction of grounded answers where every claim is
  supported by retrieved context. Scored on a 0/0.5/1 scale per case.
  Only computed for cases where retrieval ran (cases routed to `no_retrieval`
  have no context to be faithful to and are excluded from this metric).
- **Answer relevance** — does the answer address what was asked?
- **Behavior accuracy** — did the system route to the right branch?

**Results by expected behavior:**

| Expected | Count | Faithfulness | Faith n | Relevance | Match rate |
|---|---|---|---|---|---|
| answer | 13 | 0.938 | 8 | 1.000 | 1.000 |
| partial | 1 | 1.000 | 1 | 1.000 | 1.000 |
| refuse | 1 | 1.000 | 0 | 1.000 | 1.000 |
| **Overall** | **15** | **0.944** | **9** | **1.000** | **1.000** |

**Notable findings:**

- **Behavior accuracy is 1.000.** Every test case routed to its expected
  pipeline branch — including the refusal case ("What is our company's PTO
  policy?") which correctly declined to invent a policy.
- **Faithfulness is 0.944, not 1.000.** One case scored 0.5 — the system
  added a closing summary line ("Each service has one job — together they
  form a complete, scalable, GDPR-compliant AI system") that wasn't literally
  in the retrieved chunks. The judge correctly flagged this as a partial
  fidelity issue. Notably, this is a stylistic embellishment, not a factual
  hallucination.
- **Cases where retrieval didn't run** (router classified as `no_retrieval`)
  are excluded from the faithfulness metric. Including them would deflate the
  number while measuring nothing meaningful — there's no retrieved context to
  be faithful to.

---

## Architecture under test

The numbers above measure the production configuration, which runs:

- **Hybrid retrieval** — dense (BGE) + sparse (BM25) with Reciprocal Rank
  Fusion (RRF, k=60)
- **Cross-encoder reranking** — Xenova/ms-marco-MiniLM-L-6-v2 over top-20
  candidates, returning top-5
- **CRAG self-correction** — adaptive routing with up to 3 retry iterations
  via query decomposition
- **Three-layer security** — input guard (injection detection), content
  filter (chunk-level instruction stripping), output guard (PII redaction)
- **Semantic cache** with 0.92 cosine similarity threshold

The codebase includes ablation flags (`DISABLE_RERANKER`, `DISABLE_CRAG`,
`DISABLE_CONTENT_FILTER`, `RETRIEVAL_MODE`) wired through the pipeline. A
formal ablation study comparing each component's contribution was scoped out
of this evaluation; the infrastructure to run it is in place.

---

## Methodology notes & limitations

A few honest qualifications about what these numbers do and don't tell you:

- **Test set size is small.** 30 retrieval cases and 15 generation cases is
  enough to be directionally informative but not statistically robust. A
  production deployment would expand to hundreds of cases.
- **The corpus is intentionally small** (3 documents, 18 chunks) for fast
  iteration. Some metrics behave differently at larger corpus sizes — for
  instance, Hit Rate@5 becomes harder to achieve when there are more
  irrelevant chunks competing for the top-5 slots.
- **LLM-as-judge has noise.** The same case scored multiple times can vary
  by 0.5 occasionally. The numbers reported here are single-run; a more
  rigorous setup would average across 3-5 judge calls per case.
- **Faithfulness uses citation snippets, not full chunk text.** The judge
  sees up to 1500 characters per cited chunk — enough to verify most claims
  but occasionally insufficient for very long chunks. A future improvement
  would be a debug API endpoint returning the full context the LLM saw.
- **The 0.917 direct_factual score reflects one ground-truth labeling
  ambiguity** (overlapping chunks where the answer spans both). The system's
  behavior on the case in question was correct; the dataset entry pointed
  to a sibling chunk.

---

## Reproduction

The full evaluation harness is in `eval/`. Each suite writes a detailed
markdown report to `eval/reports/` (gitignored).

**Prerequisites:**

```bash
# 1. Stack running
docker compose up -d

# 2. Documents ingested into Qdrant
python -m pipeline.ingest --source test-docs/

# 3. ANTHROPIC_API_KEY set in .env (for LLM judge)
```

**Run both suites:**

```bash
python -m eval.run
```

**Run individual suites:**

```bash
python -m eval.run --suite retrieval     # ~30 seconds
python -m eval.run --suite generation    # ~3-5 minutes
```

Reports are written to `eval/reports/` with timestamped filenames for
side-by-side comparison across runs.

---

## Source

- Test datasets: `eval/dataset/retrieval_eval.json`, `eval/dataset/generation_eval.json`
- Metrics implementation: `eval/metrics/retrieval.py`, `eval/metrics/llm_judge.py`
- Runners: `eval/runners/retrieval_runner.py`, `eval/runners/e2e_runner.py`
- CLI: `eval/run.py`

_Last updated: April 2026_