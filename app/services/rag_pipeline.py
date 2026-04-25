"""RAG pipeline orchestrator — runtime query flow with CRAG self-correction.

Phase 5 inserts CRAG between routing and generation:

    Query → memory rewrite → cache lookup → route →
            (if RAG path) → CRAG.run() → generate → cache + memory → respond

CRAG.run() internally handles retrieve, grade, decide, decompose, fallback.
The pipeline only sees the final outcome.
"""

import logging
import re
import uuid
from dataclasses import dataclass

from anthropic import AsyncAnthropic

from app.agents.adaptive_router import Action
from app.agents.crag import CRAGAgent
from app.config import settings
from app.models import Citation, QueryResponse
from app.prompts import PromptName, registry
from app.services.conversation import ConversationMemory
from app.services.document_grader import DocumentGrader, GradedChunk
from app.services.query_decomposer import QueryDecomposer
from app.services.query_router import Intent, QueryRouter
from app.services.semantic_cache import SemanticCache

log = logging.getLogger(__name__)

RE_CITATION_MARKER = re.compile(r"\[#(\d+)\]")


@dataclass
class PipelineDiagnostics:
    """Internal diagnostics for the response trace."""

    intent: str
    cache_hit: bool
    crag_iterations: int
    crag_tools_used: list[str]
    sub_queries: list[str]
    retrieved_count: int
    relevant_count: int


class RAGPipeline:
    """The runtime query orchestrator with CRAG self-correction."""

    def __init__(
        self,
        anthropic_client: AsyncAnthropic,
        crag_agent: CRAGAgent,
        memory: ConversationMemory,
        cache: SemanticCache,
        router: QueryRouter,
        decomposer: QueryDecomposer,
        grader: DocumentGrader,
    ):
        self.anthropic = anthropic_client
        self.crag = crag_agent
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
            answer = await self._answer_without_retrieval(rewritten)
            citations: list[Citation] = []
            diagnostics = PipelineDiagnostics(
                intent=intent.value,
                cache_hit=False,
                crag_iterations=0,
                crag_tools_used=[],
                sub_queries=[],
                retrieved_count=0,
                relevant_count=0,
            )

        else:
            # factual or comparative → run the CRAG self-correction loop
            crag_result = await self.crag.run(rewritten)

            relevant = DocumentGrader.filter_relevant(crag_result.graded_chunks)
            action = crag_result.final_decision.action

            if action == Action.REFUSE or not relevant:
                answer = await self._refuse(rewritten)
                citations = []
            else:
                answer, citations = await self._generate_with_context(
                    rewritten, relevant
                )

            diagnostics = PipelineDiagnostics(
                intent=intent.value,
                cache_hit=False,
                crag_iterations=crag_result.iterations,
                crag_tools_used=crag_result.tools_used,
                sub_queries=crag_result.sub_queries_used,
                retrieved_count=len(crag_result.graded_chunks),
                relevant_count=len(relevant),
            )

        # ---- Step 5: persist to cache + memory ----
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
            "Pipeline done: intent=%s crag_iter=%d tools=%s retrieved=%d relevant=%d",
            diagnostics.intent,
            diagnostics.crag_iterations,
            diagnostics.crag_tools_used,
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

    async def _generate_with_context(
        self,
        query: str,
        relevant: list[GradedChunk],
    ) -> tuple[str, list[Citation]]:
        """Build the prompt with numbered context, generate, parse citations."""
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

    async def _answer_without_retrieval(self, query: str) -> str:
        """Direct LLM answer for chitchat / no_retrieval intents."""
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
        """Graceful refusal when CRAG concluded no relevant info exists."""
        prompt = registry.get(PromptName.REFUSAL).format(query=query)
        response = await self.anthropic.messages.create(
            model=settings.model_generation,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    @staticmethod
    def _parse_cited_indices(answer: str, max_idx: int) -> list[int]:
        seen = set()
        ordered: list[int] = []
        for match in RE_CITATION_MARKER.finditer(answer):
            idx = int(match.group(1))
            if 1 <= idx <= max_idx and idx not in seen:
                seen.add(idx)
                ordered.append(idx)
        return ordered