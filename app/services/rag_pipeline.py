"""RAG pipeline orchestrator — runtime query flow with CRAG self-correction,
three security guard layers, and SSE streaming support.

Pipeline stages (both run() and stream() share the same logic):
    Query → input guard → memory rewrite → cache lookup → route →
            (if RAG) → CRAG.run() → content filter → generate →
            output guard → cache + memory → respond

run()    — non-streaming, returns final QueryResponse
stream() — yields events as they happen, for SSE clients

Phase 4 added cache + memory + routing + grading.
Phase 5 added the CRAG self-correction loop.
Phase 6 added the three security guards (input / content / output).
Phase 7 added the streaming variant.
Phase 8 (this commit): citation snippets bumped from 300 → 1500 chars
                       so eval-time judges have full context. Frontend should
                       truncate at display time if a shorter preview is wanted.
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
from app.security.content_filter import ContentFilter
from app.security.input_guard import InputGuard
from app.security.output_guard import OutputGuard
from app.services.conversation import ConversationMemory
from app.services.document_grader import DocumentGrader, GradedChunk
from app.services.query_decomposer import QueryDecomposer
from app.services.query_router import Intent, QueryRouter
from app.services.semantic_cache import SemanticCache

log = logging.getLogger(__name__)

RE_CITATION_MARKER = re.compile(r"\[#(\d+)\]")

# Maximum characters of chunk content stored on each Citation.
# Large enough that an LLM judge can verify faithfulness against the snippet
# alone, without needing the original chunk. Frontend should truncate further
# at display time if a shorter preview is desired.
CITATION_SNIPPET_CHARS = 1500


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
    guard_flags: list[str]


class RAGPipeline:
    """The runtime query orchestrator with CRAG + three-layer security."""

    def __init__(
        self,
        anthropic_client: AsyncAnthropic,
        crag_agent: CRAGAgent,
        memory: ConversationMemory,
        cache: SemanticCache,
        router: QueryRouter,
        decomposer: QueryDecomposer,
        grader: DocumentGrader,
        input_guard: InputGuard,
        content_filter: ContentFilter,
        output_guard: OutputGuard,
    ):
        self.anthropic = anthropic_client
        self.crag = crag_agent
        self.memory = memory
        self.cache = cache
        self.router = router
        self.decomposer = decomposer
        self.grader = grader
        self.input_guard = input_guard
        self.content_filter = content_filter
        self.output_guard = output_guard

    # ============================================================
    # Non-streaming entrypoint
    # ============================================================

    async def run(
        self,
        query: str,
        conversation_id: str | None = None,
    ) -> QueryResponse:
        """Run the full pipeline for one query. Returns answer + citations."""
        conversation_id = conversation_id or str(uuid.uuid4())

        # ---- Step 0: input guard — block injection attempts ----
        guard_check = await self.input_guard.check(query)
        if not guard_check.passed:
            log.warning(
                "Pipeline aborted by input guard: reason=%r flags=%s",
                guard_check.reason,
                guard_check.flags,
            )
            return QueryResponse(
                answer="I can't process that request.",
                citations=[],
                conversation_id=conversation_id,
                cache_hit=False,
            )

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
                guard_flags=guard_check.flags,
            )

        else:
            # factual or comparative → run the CRAG self-correction loop
            crag_result = await self.crag.run(rewritten)

            # Apply content filter to retrieved chunks before generation
            self._apply_content_filter(crag_result.graded_chunks)

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
                guard_flags=guard_check.flags,
            )

        # ---- Step 4.5: output guard — redact PII before responding ----
        output_result = await self.output_guard.check(answer)
        answer = output_result.sanitized_text
        if output_result.redactions:
            log.info("OutputGuard applied: %s", output_result.counts)

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
            "Pipeline done: intent=%s crag_iter=%d tools=%s retrieved=%d relevant=%d guard_flags=%s",
            diagnostics.intent,
            diagnostics.crag_iterations,
            diagnostics.crag_tools_used,
            diagnostics.retrieved_count,
            diagnostics.relevant_count,
            diagnostics.guard_flags,
        )

        return QueryResponse(
            answer=answer,
            citations=citations,
            conversation_id=conversation_id,
            cache_hit=False,
        )

    # ============================================================
    # Streaming entrypoint
    # ============================================================

    async def stream(
        self,
        query: str,
        conversation_id: str | None = None,
    ):
        """Stream events for one query.

        Yields dicts with `{"event": str, "data": dict}` shape, suitable for
        sse-starlette's EventSourceResponse. Mirrors run() but emits tokens
        and progress signals progressively.
        """
        conversation_id = conversation_id or str(uuid.uuid4())

        # ---- Step 0: input guard ----
        yield {"event": "status", "data": {"stage": "input_guard"}}
        guard_check = await self.input_guard.check(query)
        if not guard_check.passed:
            log.warning(
                "Pipeline aborted by input guard: reason=%r flags=%s",
                guard_check.reason, guard_check.flags,
            )
            yield {"event": "token", "data": {"text": "I can't process that request."}}
            yield {
                "event": "done",
                "data": {
                    "conversation_id": conversation_id,
                    "cache_hit": False,
                    "blocked": True,
                },
            }
            return

        # ---- Step 1: rewrite query using conversation history ----
        yield {"event": "status", "data": {"stage": "memory_rewrite"}}
        rewritten = await self.memory.rewrite_query(query, conversation_id)

        # ---- Step 2: semantic cache lookup ----
        yield {"event": "status", "data": {"stage": "cache_lookup"}}
        cached = await self.cache.lookup(rewritten)
        if cached:
            for citation_dict in cached.citations:
                yield {"event": "citation", "data": citation_dict}
            yield {"event": "token", "data": {"text": cached.answer}}
            yield {
                "event": "done",
                "data": {
                    "conversation_id": conversation_id,
                    "cache_hit": True,
                },
            }
            return

        # ---- Step 3: classify intent ----
        yield {"event": "status", "data": {"stage": "routing"}}
        intent = await self.router.classify(rewritten)

        # ---- Step 4: branch by intent ----
        full_answer_parts: list[str] = []
        citations: list[Citation] = []

        if intent in (Intent.CHITCHAT, Intent.NO_RETRIEVAL):
            yield {"event": "status", "data": {"stage": "generating"}}
            async for token_text in self._stream_without_retrieval(rewritten):
                full_answer_parts.append(token_text)
                yield {"event": "token", "data": {"text": token_text}}

        else:
            # factual or comparative → run CRAG
            yield {"event": "status", "data": {"stage": "crag_running"}}
            crag_result = await self.crag.run(rewritten)

            self._apply_content_filter(crag_result.graded_chunks)
            relevant = DocumentGrader.filter_relevant(crag_result.graded_chunks)
            action = crag_result.final_decision.action

            if action == Action.REFUSE or not relevant:
                yield {"event": "status", "data": {"stage": "refusing"}}
                async for token_text in self._stream_refuse(rewritten):
                    full_answer_parts.append(token_text)
                    yield {"event": "token", "data": {"text": token_text}}
            else:
                # Emit citations before the answer streams
                yield {"event": "status", "data": {"stage": "generating"}}
                citations = self._build_citations(relevant)
                for cit in citations:
                    yield {"event": "citation", "data": cit.model_dump()}

                async for token_text in self._stream_with_context(rewritten, relevant):
                    full_answer_parts.append(token_text)
                    yield {"event": "token", "data": {"text": token_text}}

        # ---- Step 4.5: output guard runs on the assembled answer ----
        # Streaming makes per-token redaction tricky (PII can span tokens).
        # We emit raw tokens during streaming, then send a "redacted" event
        # if the assembled answer contained PII. Frontend can replace the
        # displayed text with the sanitized version.
        full_answer = "".join(full_answer_parts)
        output_result = await self.output_guard.check(full_answer)
        if output_result.redactions:
            log.info("OutputGuard applied (post-stream): %s", output_result.counts)
            yield {
                "event": "redacted",
                "data": {
                    "sanitized_text": output_result.sanitized_text,
                    "categories": output_result.redactions,
                },
            }
            full_answer = output_result.sanitized_text

        # ---- Step 5: persist ----
        try:
            await self.cache.store(
                rewritten,
                full_answer,
                citations=[c.model_dump() for c in citations],
            )
            await self.memory.append(conversation_id, query, full_answer)
        except Exception as e:
            log.warning("Cache/memory store failed (non-fatal): %s", e)

        yield {
            "event": "done",
            "data": {
                "conversation_id": conversation_id,
                "cache_hit": False,
            },
        }

    # ============================================================
    # Internal helpers
    # ============================================================

    def _apply_content_filter(self, graded_chunks: list[GradedChunk]) -> None:
        """Filter retrieved chunks for instruction-like content. Mutates in place."""
        for graded in graded_chunks:
            original = graded.point.payload.get("content", "")
            if not original:
                continue
            filtered = self.content_filter.filter_chunk_text(original)
            if filtered.redactions:
                graded.point.payload["content"] = filtered.sanitized_content
                graded.point.payload["_filter_flags"] = filtered.redactions

    def _build_citations(self, relevant: list[GradedChunk]) -> list[Citation]:
        """Build citations up front. Used by stream() to emit them before tokens."""
        return [
            Citation(
                source=c.point.payload.get("source", "unknown"),
                chunk_id=str(c.point.id),
                score=c.rerank_score,
                snippet=(c.point.payload.get("content") or "")[:CITATION_SNIPPET_CHARS],
            )
            for c in relevant
        ]

    async def _generate_with_context(
        self,
        query: str,
        relevant: list[GradedChunk],
    ) -> tuple[str, list[Citation]]:
        """Non-streaming generation with retrieved context."""
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
                snippet=(relevant[i - 1].point.payload.get("content") or "")[
                    :CITATION_SNIPPET_CHARS
                ],
            )
            for i in cited_indices
        ]

        return answer, citations

    async def _stream_with_context(self, query: str, relevant: list[GradedChunk]):
        """Async generator yielding tokens from FACTUAL_QA generation."""
        context_lines: list[str] = []
        for idx, c in enumerate(relevant, start=1):
            content = c.point.payload.get("content", "")
            source = c.point.payload.get("source", "unknown")
            context_lines.append(f"[#{idx}] (source: {source})\n{content}")
        context = "\n\n---\n\n".join(context_lines)

        prompt = registry.get(PromptName.FACTUAL_QA).format(
            context=context, query=query
        )

        async with self.anthropic.messages.stream(
            model=settings.model_generation,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def _answer_without_retrieval(self, query: str) -> str:
        """Non-streaming chitchat / no_retrieval answer."""
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

    async def _stream_without_retrieval(self, query: str):
        """Async generator for chitchat / no_retrieval intents."""
        sys_prompt = (
            "You are a helpful assistant. Respond briefly and naturally. "
            "If the user is just chatting or asking a generic knowledge question, "
            "answer in 1-3 sentences without citations."
        )
        async with self.anthropic.messages.stream(
            model=settings.model_generation,
            max_tokens=300,
            system=sys_prompt,
            messages=[{"role": "user", "content": query}],
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def _refuse(self, query: str) -> str:
        """Non-streaming graceful refusal."""
        prompt = registry.get(PromptName.REFUSAL).format(query=query)
        response = await self.anthropic.messages.create(
            model=settings.model_generation,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    async def _stream_refuse(self, query: str):
        """Async generator for graceful refusals."""
        prompt = registry.get(PromptName.REFUSAL).format(query=query)
        async with self.anthropic.messages.stream(
            model=settings.model_generation,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text

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
