"""RAG pipeline orchestrator — the main runtime query flow.

This file conducts every other service. Read it top-to-bottom to understand
exactly what happens to a user query. The flow:

    Query → memory rewrite → cache lookup → route → (decompose) →
            retrieve → grade → generate → store cache + memory → respond

Phase 4 covers all of the above. Phase 5 will insert the CRAG self-correction
loop between grading and generation. Phase 7 will add streaming.
"""

import logging
import re
import uuid
from dataclasses import dataclass

from anthropic import AsyncAnthropic

from app.config import settings
from app.models import Citation, QueryResponse
from app.prompts import PromptName, registry
from app.retrieval.hybrid_retrieval import HybridRetriever
from app.retrieval.reranker import Reranker
from app.services.conversation import ConversationMemory
from app.services.document_grader import DocumentGrader, Grade, GradedChunk
from app.services.query_decomposer import QueryDecomposer
from app.services.query_router import Intent, QueryRouter
from app.services.semantic_cache import SemanticCache

log = logging.getLogger(__name__)

# Regex to extract citation markers like [#1], [#2] from generated text
RE_CITATION_MARKER = re.compile(r"\[#(\d+)\]")


@dataclass
class PipelineDiagnostics:
    """Internal diagnostics for the /api/query response trace."""

    intent: str
    cache_hit: bool
    sub_queries: list[str]
    retrieved_count: int
    relevant_count: int


class RAGPipeline:
    """The runtime query orchestrator."""

    def __init__(
        self,
        anthropic_client: AsyncAnthropic,
        retriever: HybridRetriever,
        reranker: Reranker,
        memory: ConversationMemory,
        cache: SemanticCache,
        router: QueryRouter,
        decomposer: QueryDecomposer,
        grader: DocumentGrader,
    ):
        self.anthropic = anthropic_client
        self.retriever = retriever
        self.reranker = reranker
        self.memory = memory
        self.cache = cache
        self.router = router
        self.decomposer = decomposer
        self.grader = grader

    async def run(
        self,
        query: str,
        conversation_id: str | None = None,
    ) -> QueryResponse:
        """Run the full pipeline for one query. Returns answer + citations."""
        # Generate a conversation_id if none provided — fresh conversation
        conversation_id = conversation_id or str(uuid.uuid4())

        # ---- Step 1: rewrite query using conversation history ----
        rewritten = await self.memory.rewrite_query(query, conversation_id)

        # ---- Step 2: semantic cache lookup ----
        cached = await self.cache.lookup(rewritten)
        if cached:
            return QueryResponse(
                answer=cached.answer,
                citations=[Citation(**c) for c in cached.citations],
                conversation_id=conversation_id,
                cache_hit=True,
            )

        # ---- Step 3: classify intent ----
        intent = await self.router.classify(rewritten)

        # ---- Step 4: branch by intent ----
        if intent in (Intent.CHITCHAT, Intent.NO_RETRIEVAL):
            answer = await self._answer_without_retrieval(rewritten, intent)
            citations: list[Citation] = []
            diagnostics = PipelineDiagnostics(
                intent=intent.value,
                cache_hit=False,
                sub_queries=[],
                retrieved_count=0,
                relevant_count=0,
            )

        else:
            # factual or comparative: do RAG
            sub_queries = await self._maybe_decompose(rewritten, intent)
            graded_chunks = await self._retrieve_and_grade(sub_queries)
            relevant = DocumentGrader.filter_relevant(graded_chunks)

            if not relevant:
                answer = await self._refuse(rewritten)
                citations = []
            else:
                answer, citations = await self._generate_with_context(
                    rewritten, relevant
                )

            diagnostics = PipelineDiagnostics(
                intent=intent.value,
                cache_hit=False,
                sub_queries=sub_queries,
                retrieved_count=len(graded_chunks),
                relevant_count=len(relevant),
            )

        # ---- Step 5: persist to cache + memory (fire-and-forget on failure) ----
        try:
            await self.cache.store(
                rewritten,
                answer,
                citations=[c.model_dump() for c in citations],
            )
            await self.memory.append(conversation_id, query, answer)
        except Exception as e:
            log.warning("Cache/memory store failed (non-fatal): %s", e)

        log.info(
            "Pipeline done: intent=%s subq=%d retrieved=%d relevant=%d",
            diagnostics.intent,
            len(diagnostics.sub_queries),
            diagnostics.retrieved_count,
            diagnostics.relevant_count,
        )

        return QueryResponse(
            answer=answer,
            citations=citations,
            conversation_id=conversation_id,
            cache_hit=False,
        )

    # ---------- Internal pipeline branches ----------

    async def _maybe_decompose(self, query: str, intent: Intent) -> list[str]:
        """Decompose only for comparative intent. Otherwise return [query]."""
        if intent == Intent.COMPARATIVE:
            return await self.decomposer.decompose(query)
        return [query]

    async def _retrieve_and_grade(
        self,
        sub_queries: list[str],
    ) -> list[GradedChunk]:
        """Retrieve + rerank for each sub-query, dedupe, grade all together.

        For multi-sub-query, we use the FIRST sub-query as the grading "query"
        because grading against multiple queries explodes LLM cost. In practice
        this works well — sub-queries are usually facets of one user intent.
        """
        all_reranked = []
        seen_ids = set()

        for sq in sub_queries:
            candidates = await self.retriever.retrieve(sq)
            reranked = self.reranker.rerank(sq, candidates)
            for r in reranked:
                if r.point.id in seen_ids:
                    continue
                seen_ids.add(r.point.id)
                all_reranked.append(r)

        # Grade all unique chunks against the original (or first) sub-query
        grading_query = sub_queries[0] if sub_queries else ""
        return await self.grader.grade(grading_query, all_reranked)

    async def _generate_with_context(
        self,
        query: str,
        relevant: list[GradedChunk],
    ) -> tuple[str, list[Citation]]:
        """Build the prompt with numbered context, generate, parse citations."""
        # Number chunks 1..N for citation markers
        context_lines: list[str] = []
        for idx, c in enumerate(relevant, start=1):
            content = c.point.payload.get("content", "")
            source = c.point.payload.get("source", "unknown")
            context_lines.append(f"[#{idx}] (source: {source})\n{content}")
        context = "\n\n---\n\n".join(context_lines)

        prompt = registry.get(PromptName.FACTUAL_QA).format(
            context=context, query=query
        )

        response = await self.anthropic.messages.create(
            model=settings.model_generation,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.content[0].text.strip()

        # Parse [#N] markers and map back to chunks
        cited_indices = self._parse_cited_indices(answer, max_idx=len(relevant))
        citations = [
            Citation(
                source=relevant[i - 1].point.payload.get("source", "unknown"),
                chunk_id=str(relevant[i - 1].point.id),
                score=relevant[i - 1].rerank_score,
                snippet=(relevant[i - 1].point.payload.get("content") or "")[:300],
            )
            for i in cited_indices
        ]

        return answer, citations

    async def _answer_without_retrieval(self, query: str, intent: Intent) -> str:
        """Direct LLM answer for chitchat / no_retrieval intents."""
        # Small dedicated prompt — friendly and brief
        sys_prompt = (
            "You are a helpful assistant. Respond briefly and naturally. "
            "If the user is just chatting or asking a generic knowledge question, "
            "answer in 1-3 sentences without citations."
        )
        response = await self.anthropic.messages.create(
            model=settings.model_generation,
            max_tokens=300,
            system=sys_prompt,
            messages=[{"role": "user", "content": query}],
        )
        return response.content[0].text.strip()

    async def _refuse(self, query: str) -> str:
        """Graceful refusal when retrieval found nothing relevant."""
        prompt = registry.get(PromptName.REFUSAL).format(query=query)
        response = await self.anthropic.messages.create(
            model=settings.model_generation,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    @staticmethod
    def _parse_cited_indices(answer: str, max_idx: int) -> list[int]:
        """Extract unique [#N] citation markers from the answer."""
        seen = set()
        ordered: list[int] = []
        for match in RE_CITATION_MARKER.finditer(answer):
            idx = int(match.group(1))
            if 1 <= idx <= max_idx and idx not in seen:
                seen.add(idx)
                ordered.append(idx)
        return ordered