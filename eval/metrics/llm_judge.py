"""LLM-as-judge metrics — faithfulness and answer relevance scoring.

Uses Claude Haiku to evaluate generated answers. The judge sees the question,
the retrieved context, and the answer, then returns a verdict + reasoning.

Two metrics:
- Faithfulness:   does the answer make claims unsupported by the context?
- Answer Relevance: does the answer actually address the question asked?

Both return scores in [0.0, 1.0]. Multiple calls per answer would smooth
noise; for portfolio scale, single calls are fine and the relative comparison
across configurations is what we care about.
"""

import json
import logging
import re
from dataclasses import dataclass

from anthropic import AsyncAnthropic

from app.config import settings

log = logging.getLogger(__name__)

RE_OUTPUT_TAG = re.compile(r"<output>(.*?)</output>", re.DOTALL | re.IGNORECASE)


# ============================================================
# Judge prompts
# ============================================================

FAITHFULNESS_PROMPT = """You are evaluating whether an AI assistant's answer is faithful to the source context provided to it.

# Definitions
- FAITHFUL: every factual claim in the answer is supported by the context, or is a trivially obvious inference. Phrases like "the context does not mention X" are fully faithful.
- UNFAITHFUL: the answer contains at least one factual claim that the context does not support — a hallucination.

# Scoring
Output a score in {0.0, 0.5, 1.0}:
- 1.0 = fully faithful
- 0.5 = mostly faithful, with one minor unsupported detail (e.g. a date or name)
- 0.0 = unfaithful — at least one major hallucinated claim

# Important
- An answer that says "I don't know" or "the context doesn't cover this" is FAITHFUL (1.0). It's making no unsupported claims.
- Answers should be judged ONLY on what the context supports. Don't penalize the answer for missing facts that are not in the context.

# Context provided to the assistant
{context}

# Question
{query}

# Assistant's answer
{answer}

# Your judgment
Output a single JSON object inside <output> tags:
<output>{{"score": 0.0|0.5|1.0, "reasoning": "<one sentence explaining the score>"}}</output>
"""


ANSWER_RELEVANCE_PROMPT = """You are evaluating whether an AI assistant's answer is relevant to the user's question.

# Definitions
- RELEVANT: the answer directly addresses what the user asked, even if it acknowledges missing information.
- IRRELEVANT: the answer talks about a different topic, evades the question, or is gibberish.

# Scoring
Output a score in {0.0, 0.5, 1.0}:
- 1.0 = directly addresses the question
- 0.5 = partially relevant — touches the topic but misses the specific ask
- 0.0 = off-topic or evasive

# Important
- An honest "I don't have that information" is RELEVANT (1.0) when the question genuinely can't be answered from the context.
- An off-topic ramble is irrelevant even if individual sentences are factually true.
- Don't penalize for answer style or length — only judge whether the question was addressed.

# Question
{query}

# Assistant's answer
{answer}

# Your judgment
Output a single JSON object inside <output> tags:
<output>{{"score": 0.0|0.5|1.0, "reasoning": "<one sentence explaining the score>"}}</output>
"""


# ============================================================
# Result type
# ============================================================

@dataclass
class JudgeScore:
    """Result of one judge call."""

    metric: str           # "faithfulness" or "answer_relevance"
    score: float          # 0.0, 0.5, or 1.0
    reasoning: str
    raw_response: str     # for debugging if parsing fails


# ============================================================
# The judge
# ============================================================

class LLMJudge:
    """Haiku-based judge for faithfulness and answer relevance."""

    def __init__(
        self,
        anthropic_client: AsyncAnthropic,
        model: str | None = None,
    ):
        self.anthropic = anthropic_client
        self.model = model or settings.model_routing  # Haiku

    async def judge_faithfulness(
        self,
        query: str,
        context: str,
        answer: str,
    ) -> JudgeScore:
        """Score whether the answer is grounded in the context."""
        prompt = FAITHFULNESS_PROMPT.format(
            context=context or "(no context was retrieved)",
            query=query,
            answer=answer,
        )
        return await self._judge(prompt, metric="faithfulness")

    async def judge_answer_relevance(
        self,
        query: str,
        answer: str,
    ) -> JudgeScore:
        """Score whether the answer addresses the question."""
        prompt = ANSWER_RELEVANCE_PROMPT.format(
            query=query,
            answer=answer,
        )
        return await self._judge(prompt, metric="answer_relevance")

    async def _judge(self, prompt: str, metric: str) -> JudgeScore:
        """Send the prompt, parse the JSON verdict, return a JudgeScore."""
        try:
            response = await self.anthropic.messages.create(
                model=self.model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text
        except Exception as e:
            log.warning("Judge API call failed for %s: %s", metric, e)
            return JudgeScore(
                metric=metric,
                score=0.0,
                reasoning=f"Judge call failed: {e}",
                raw_response="",
            )

        score, reasoning = self._parse_verdict(raw)
        log.info("Judge[%s] = %.1f — %s", metric, score, reasoning[:80])
        return JudgeScore(
            metric=metric,
            score=score,
            reasoning=reasoning,
            raw_response=raw,
        )

    @staticmethod
    def _parse_verdict(raw: str) -> tuple[float, str]:
        """Extract {score, reasoning} JSON from the judge's response."""
        match = RE_OUTPUT_TAG.search(raw)
        json_str = match.group(1).strip() if match else raw.strip()

        try:
            verdict = json.loads(json_str)
        except json.JSONDecodeError:
            log.warning("Judge output was not valid JSON: %r", json_str[:200])
            return 0.0, "Could not parse judge response"

        score = verdict.get("score")
        reasoning = verdict.get("reasoning", "")

        # Normalize: accept floats, ints, or stringified numbers
        try:
            score = float(score)
        except (TypeError, ValueError):
            log.warning("Judge score not parseable: %r", score)
            return 0.0, f"Unparseable score: {score!r}"

        # Clamp to expected range
        if score not in (0.0, 0.5, 1.0):
            log.warning("Judge score out of expected set: %r — clamping", score)
            score = max(0.0, min(1.0, score))

        return score, str(reasoning)


# ============================================================
# Aggregation
# ============================================================

def aggregate_scores(scores: list[JudgeScore]) -> dict[str, float]:
    """Average judge scores by metric. Returns dict of metric → average."""
    by_metric: dict[str, list[float]] = {}
    for s in scores:
        by_metric.setdefault(s.metric, []).append(s.score)

    return {
        metric: sum(values) / len(values) if values else 0.0
        for metric, values in by_metric.items()
    }