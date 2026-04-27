"""End-to-end generation eval runner.

For each test case in generation_eval.json:
  1. POST /api/query → get the system's answer + citations
  2. Reconstruct the "context" from the citation snippets
  3. Score faithfulness via LLMJudge (does the answer hallucinate?)
     — skipped for cases with 0 citations (no_retrieval / refuse paths
       have no retrieved context to be faithful to)
  4. Score answer relevance via LLMJudge (does the answer address the question?)
  5. Compare actual vs expected behavior (answer / refuse / partial)

Sequential to respect Anthropic rate limits. Wall time ~3-5 minutes for 15 cases.

Usage:
    python -m eval.runners.e2e_runner

Output: eval/reports/generation_<timestamp>.md
"""

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import httpx
from anthropic import AsyncAnthropic

from app.config import settings
from eval.metrics.llm_judge import JudgeScore, LLMJudge

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

DATASET_PATH = Path("eval/dataset/generation_eval.json")
REPORTS_DIR = Path("eval/reports")

API_BASE = "http://localhost:8000"
HTTP_TIMEOUT = 60.0  # full pipeline can take ~10s on first run


# ============================================================
# Per-case result type
# ============================================================

@dataclass
class CaseResult:
    """Outcome of running and judging one test case."""

    id: str
    query: str
    expected_behavior: str

    # System output
    answer: str
    citations_count: int
    cache_hit: bool

    # Judge scores
    faithfulness: float
    faithfulness_reasoning: str
    faithfulness_judged: bool   # False when skipped (no_retrieval / refuse)
    answer_relevance: float
    answer_relevance_reasoning: str

    # Behavior classification
    actual_behavior: str   # answer / refuse / partial / error
    behavior_match: bool

    # Diagnostics
    error: str | None = None


# ============================================================
# Calling the live system
# ============================================================

async def call_query(
    client: httpx.AsyncClient,
    query: str,
) -> dict:
    """POST /api/query (non-streaming). Returns the JSON response."""
    resp = await client.post(
        f"{API_BASE}/api/query",
        json={"query": query, "stream": False},
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


# ============================================================
# Behavior classification
# ============================================================

def classify_behavior(response: dict) -> str:
    """Classify the system's response as answer / refuse / partial / error."""
    answer = (response.get("answer") or "").lower()
    citations = response.get("citations") or []

    refuse_phrases = [
        "i don't have",
        "i do not have",
        "i can't process",
        "no information",
        "not in the available documents",
        "not in the provided context",
        "the provided context does not",
        "cannot find",
    ]
    is_refusing = any(p in answer for p in refuse_phrases)

    partial_phrases = [
        "context only",
        "context does not",
        "limited details",
        "does not include further",
        "however, the context",
    ]
    has_partial_acknowledgment = any(p in answer for p in partial_phrases)

    if is_refusing and not citations:
        return "refuse"
    if has_partial_acknowledgment and citations:
        return "partial"
    if citations or len(answer) > 50:
        return "answer"
    return "answer"


# ============================================================
# Running one case
# ============================================================

async def run_case(
    http_client: httpx.AsyncClient,
    judge: LLMJudge,
    case: dict,
) -> CaseResult:
    """Run one test case end-to-end. Returns a CaseResult."""
    case_id = case["id"]
    log.info("Running %s: %r", case_id, case["query"][:80])

    # --- 1. Hit the pipeline ---
    try:
        response = await call_query(http_client, case["query"])
    except Exception as e:
        log.warning("Query failed for %s: %s", case_id, e)
        return CaseResult(
            id=case_id,
            query=case["query"],
            expected_behavior=case["expected_behavior"],
            answer="",
            citations_count=0,
            cache_hit=False,
            faithfulness=0.0,
            faithfulness_reasoning="N/A — query failed",
            faithfulness_judged=False,
            answer_relevance=0.0,
            answer_relevance_reasoning="N/A — query failed",
            actual_behavior="error",
            behavior_match=False,
            error=str(e),
        )

    answer = response.get("answer", "")
    citations = response.get("citations") or []
    cache_hit = response.get("cache_hit", False)

    # Reconstruct context from citation snippets for the judge.
    # The pipeline now stores up to ~1500 chars per snippet (Phase 8 fix),
    # which is enough for the judge to verify most claims.
    context = "\n\n---\n\n".join(
        f"[{c.get('source', 'unknown')}]\n{c.get('snippet', '')}"
        for c in citations
    )

    # --- 2. Judge: faithfulness ---
    # Skip when there are no citations: the no_retrieval / chitchat / refuse
    # paths have no context to be faithful to. Marking these 0.0 deflated the
    # headline number while measuring nothing meaningful.
    if not citations:
        faith_score = JudgeScore(
            metric="faithfulness",
            score=1.0,
            reasoning="N/A — no retrieval, no context to be faithful to",
            raw_response="",
        )
        faithfulness_judged = False
    else:
        faith_score = await judge.judge_faithfulness(
            query=case["query"],
            context=context,
            answer=answer,
        )
        faithfulness_judged = True

    # --- 3. Judge: answer relevance ---
    rel_score = await judge.judge_answer_relevance(
        query=case["query"],
        answer=answer,
    )

    # --- 4. Behavior classification ---
    actual = classify_behavior(response)
    expected = case["expected_behavior"]
    behavior_match = actual == expected

    log.info(
        "  faithfulness=%s relevance=%.1f behavior=%s/%s match=%s",
        f"{faith_score.score:.1f}" if faithfulness_judged else "N/A",
        rel_score.score,
        actual, expected, behavior_match,
    )

    return CaseResult(
        id=case_id,
        query=case["query"],
        expected_behavior=expected,
        answer=answer,
        citations_count=len(citations),
        cache_hit=cache_hit,
        faithfulness=faith_score.score,
        faithfulness_reasoning=faith_score.reasoning,
        faithfulness_judged=faithfulness_judged,
        answer_relevance=rel_score.score,
        answer_relevance_reasoning=rel_score.reasoning,
        actual_behavior=actual,
        behavior_match=behavior_match,
    )


# ============================================================
# Running the full suite
# ============================================================

async def run_e2e_eval() -> dict:
    """Run all generation test cases. Returns the result structure."""
    with DATASET_PATH.open() as f:
        dataset = json.load(f)
    cases = dataset["test_cases"]
    log.info("Running e2e eval on %d cases", len(cases))

    anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    judge = LLMJudge(anthropic_client=anthropic_client)

    results: list[CaseResult] = []
    async with httpx.AsyncClient() as http_client:
        for case in cases:
            result = await run_case(http_client, judge, case)
            results.append(result)

    successful = [r for r in results if r.error is None]
    n = len(successful)

    # Faithfulness is averaged ONLY over cases where we actually judged it
    # (i.e. cases with retrieved context). Cases with 0 citations are
    # tracked but not folded into the headline metric.
    judged_for_faith = [r for r in successful if r.faithfulness_judged]
    n_faith = len(judged_for_faith)

    if n > 0:
        avg_faith = (
            sum(r.faithfulness for r in judged_for_faith) / n_faith
            if n_faith > 0 else 1.0
        )
        avg_rel = sum(r.answer_relevance for r in successful) / n
        n_match = sum(1 for r in successful if r.behavior_match)
        behavior_accuracy = n_match / n
    else:
        avg_faith = avg_rel = behavior_accuracy = 0.0

    # By expected behavior — same logic, but per-bucket
    by_behavior: dict[str, list[CaseResult]] = defaultdict(list)
    for r in successful:
        by_behavior[r.expected_behavior].append(r)

    return {
        "n_cases": len(cases),
        "n_successful": len(successful),
        "n_failed": len(results) - len(successful),
        "overall": {
            "faithfulness": avg_faith,
            "faithfulness_n": n_faith,
            "answer_relevance": avg_rel,
            "behavior_accuracy": behavior_accuracy,
        },
        "by_behavior": {
            beh: _aggregate_bucket(rs)
            for beh, rs in by_behavior.items()
        },
        "per_case": [asdict(r) for r in results],
    }


def _aggregate_bucket(rs: list[CaseResult]) -> dict:
    """Aggregate metrics for one expected-behavior bucket."""
    if not rs:
        return {
            "count": 0,
            "faithfulness": 0.0,
            "faithfulness_n": 0,
            "answer_relevance": 0.0,
            "behavior_match_rate": 0.0,
        }

    judged = [r for r in rs if r.faithfulness_judged]
    n_judged = len(judged)

    return {
        "count": len(rs),
        "faithfulness": (
            sum(r.faithfulness for r in judged) / n_judged
            if n_judged > 0 else 1.0
        ),
        "faithfulness_n": n_judged,
        "answer_relevance": sum(r.answer_relevance for r in rs) / len(rs),
        "behavior_match_rate": sum(1 for r in rs if r.behavior_match) / len(rs),
    }


# ============================================================
# Reporting
# ============================================================

def _format_report(result: dict) -> str:
    overall = result["overall"]
    lines: list[str] = []

    lines.append("# End-to-End Generation Eval Report")
    lines.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_\n")

    lines.append("## Summary\n")
    lines.append(f"- Test cases: **{result['n_cases']}** total")
    lines.append(f"  - Successful: {result['n_successful']}")
    lines.append(f"  - Failed (errors): {result['n_failed']}")
    lines.append("")

    lines.append("## Overall metrics\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(
        f"| Faithfulness | **{overall['faithfulness']:.3f}** "
        f"(n={overall['faithfulness_n']}) |"
    )
    lines.append(f"| Answer relevance | **{overall['answer_relevance']:.3f}** |")
    lines.append(f"| Behavior accuracy | **{overall['behavior_accuracy']:.3f}** |")
    lines.append("")
    lines.append(
        "_Faithfulness is averaged only over cases where retrieval ran "
        "(cases with citations). Cases routed to no_retrieval / chitchat / "
        "refuse have no retrieved context, so faithfulness is undefined for "
        "them. Answer relevance is averaged across all successful cases._\n"
    )

    lines.append("## By expected behavior\n")
    lines.append(
        "| Expected | Count | Faithfulness | Faith n | Relevance | Match rate |"
    )
    lines.append("|---|---|---|---|---|---|")
    for beh, agg in result["by_behavior"].items():
        lines.append(
            f"| {beh} | {agg['count']} | {agg['faithfulness']:.3f} | "
            f"{agg['faithfulness_n']} | {agg['answer_relevance']:.3f} | "
            f"{agg['behavior_match_rate']:.3f} |"
        )
    lines.append("")

    lines.append("## Per-case results\n")
    lines.append("| ID | Expected | Actual | Match | Faith | Relev | Citations |")
    lines.append("|---|---|---|---|---|---|---|")
    for c in result["per_case"]:
        match = "✓" if c["behavior_match"] else "✗"
        faith_str = f"{c['faithfulness']:.1f}" if c["faithfulness_judged"] else "N/A"
        lines.append(
            f"| {c['id']} | {c['expected_behavior']} | {c['actual_behavior']} | "
            f"{match} | {faith_str} | {c['answer_relevance']:.1f} | "
            f"{c['citations_count']} |"
        )
    lines.append("")

    lines.append("## Diagnostics — cases worth a look\n")
    flagged = [
        c for c in result["per_case"]
        if (c["faithfulness_judged"] and c["faithfulness"] < 1.0)
        or c["answer_relevance"] < 1.0
        or not c["behavior_match"]
        or c.get("error")
    ]
    if not flagged:
        lines.append(
            "_All judged cases scored 1.0 on both metrics and matched expected "
            "behavior. Excellent._\n"
        )
    else:
        for c in flagged:
            lines.append(f"### {c['id']} — {c['query'][:80]}")
            lines.append(
                f"- **Expected behavior**: {c['expected_behavior']}, "
                f"**actual**: {c['actual_behavior']}"
            )
            faith_str = (
                f"{c['faithfulness']:.1f}" if c["faithfulness_judged"] else "N/A"
            )
            lines.append(
                f"- **Faithfulness**: {faith_str} — _{c['faithfulness_reasoning']}_"
            )
            lines.append(
                f"- **Answer relevance**: {c['answer_relevance']:.1f} — "
                f"_{c['answer_relevance_reasoning']}_"
            )
            lines.append(f"- Answer (truncated): {c['answer'][:300]}")
            if c.get("error"):
                lines.append(f"- ERROR: {c['error']}")
            lines.append("")

    return "\n".join(lines)


def write_report(result: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"generation_{timestamp}.md"
    path.write_text(_format_report(result), encoding="utf-8")
    return path


# ============================================================
# CLI entrypoint
# ============================================================

async def main() -> None:
    result = await run_e2e_eval()
    report_path = write_report(result)

    overall = result["overall"]
    print("\n" + "=" * 60)
    print("E2E generation eval complete")
    print(
        f"  Faithfulness:     {overall['faithfulness']:.3f} "
        f"(n={overall['faithfulness_n']})"
    )
    print(f"  Answer relevance: {overall['answer_relevance']:.3f}")
    print(f"  Behavior accuracy:{overall['behavior_accuracy']:.3f}")
    print(f"  Report: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())