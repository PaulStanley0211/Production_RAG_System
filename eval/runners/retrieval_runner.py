"""Retrieval eval runner — runs retrieval_eval.json against the live system.

Hits /api/search for each test case, computes Hit Rate@K, MRR, and nDCG@K,
and writes a markdown report.

Usage (from repo root):
    python -m eval.runners.retrieval_runner

Output: eval/reports/retrieval_<timestamp>.md
"""

import asyncio
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import httpx

from eval.metrics.retrieval import (
    RetrievalScore,
    aggregate,
    compute_all,
)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# Paths
DATASET_PATH = Path("eval/dataset/retrieval_eval.json")
REPORTS_DIR = Path("eval/reports")

# Config
API_BASE = "http://localhost:8000"
TOP_K = 5
HTTP_TIMEOUT = 30.0


# ============================================================
# Calling the live system
# ============================================================

async def call_search(client: httpx.AsyncClient, query: str, top_k: int) -> list[str]:
    """Hit /api/search and return retrieved chunk_ids in rank order."""
    resp = await client.get(
        f"{API_BASE}/api/search",
        params={"q": query, "top_k": top_k},
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    body = resp.json()

    # /api/search returns chunks under "results", each with "chunk_id"
    # Note: payload uses "doc_id" for path and chunk_id may be inside the
    # underlying point ID. We extract from the result item's chunk_id field.
    return [r["chunk_id"] for r in body.get("results", [])]


# ============================================================
# Running the suite
# ============================================================

async def run_retrieval_eval() -> dict:
    """Run all test cases. Returns the full result structure for reporting."""
    with DATASET_PATH.open() as f:
        dataset = json.load(f)

    cases = dataset["test_cases"]
    log.info("Running retrieval eval on %d cases", len(cases))

    in_corpus_scores: list[RetrievalScore] = []
    by_category: dict[str, list[RetrievalScore]] = {}
    out_of_corpus_results: list[dict] = []
    failed: list[dict] = []

    async with httpx.AsyncClient() as client:
        for case in cases:
            try:
                retrieved = await call_search(client, case["query"], TOP_K)
            except Exception as e:
                log.warning("Search failed for %r: %s", case["id"], e)
                failed.append({"id": case["id"], "error": str(e)})
                continue

            relevant = set(case["relevant_chunk_ids"])
            category = case["category"]

            # Out-of-corpus: track separately. We don't compute Hit Rate
            # because the metric is undefined; we measure whether the system
            # returned highly-ranked irrelevant chunks (which is bad).
            if category == "out_of_corpus":
                out_of_corpus_results.append({
                    "id": case["id"],
                    "query": case["query"],
                    "retrieved_count": len(retrieved),
                    "top_chunk_id": retrieved[0] if retrieved else None,
                })
                continue

            score = compute_all(
                query_id=case["id"],
                retrieved_ids=retrieved,
                relevant_ids=relevant,
                k=TOP_K,
            )
            in_corpus_scores.append(score)
            by_category.setdefault(category, []).append(score)

            log.info(
                "%s: HR@%d=%.2f MRR=%.2f nDCG@%d=%.2f",
                case["id"], TOP_K, score.hit_rate_at_k,
                score.mrr, TOP_K, score.ndcg_at_k,
            )

    return {
        "k": TOP_K,
        "n_cases": len(cases),
        "n_in_corpus": len(in_corpus_scores),
        "n_out_of_corpus": len(out_of_corpus_results),
        "n_failed": len(failed),
        "overall": aggregate(in_corpus_scores),
        "by_category": {
            cat: aggregate(scores) for cat, scores in by_category.items()
        },
        "per_query": [asdict(s) for s in in_corpus_scores],
        "out_of_corpus": out_of_corpus_results,
        "failed": failed,
    }


# ============================================================
# Reporting
# ============================================================

def _format_report(result: dict) -> str:
    """Build a markdown report from the run result dict."""
    k = result["k"]
    overall = result["overall"]

    lines: list[str] = []
    lines.append(f"# Retrieval Eval Report")
    lines.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_\n")

    lines.append("## Summary\n")
    lines.append(f"- Test cases: **{result['n_cases']}** total")
    lines.append(f"  - In-corpus: {result['n_in_corpus']}")
    lines.append(f"  - Out-of-corpus: {result['n_out_of_corpus']}")
    lines.append(f"  - Failed (HTTP errors): {result['n_failed']}")
    lines.append("")

    lines.append(f"## Overall metrics (in-corpus only, K={k})\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Hit Rate@{k} | **{overall['hit_rate_at_k']:.3f}** |")
    lines.append(f"| MRR | **{overall['mrr']:.3f}** |")
    lines.append(f"| nDCG@{k} | **{overall['ndcg_at_k']:.3f}** |")
    lines.append(f"| Cases | {overall['count']} |")
    lines.append("")

    lines.append("## By category\n")
    lines.append(f"| Category | Hit Rate@{k} | MRR | nDCG@{k} | Cases |")
    lines.append("|---|---|---|---|---|")
    for cat, agg in result["by_category"].items():
        lines.append(
            f"| {cat} | {agg['hit_rate_at_k']:.3f} | "
            f"{agg['mrr']:.3f} | {agg['ndcg_at_k']:.3f} | {agg['count']} |"
        )
    lines.append("")

    if result["out_of_corpus"]:
        lines.append("## Out-of-corpus behavior\n")
        lines.append(
            "These queries should retrieve nothing strongly relevant. "
            "We report what was returned for inspection — combine with "
            "the e2e generation eval to measure final REFUSE behavior.\n"
        )
        lines.append("| ID | Query | Returned | Top chunk |")
        lines.append("|---|---|---|---|")
        for r in result["out_of_corpus"]:
            top = r["top_chunk_id"][:24] + "..." if r["top_chunk_id"] else "—"
            lines.append(
                f"| {r['id']} | {r['query'][:60]} | {r['retrieved_count']} | {top} |"
            )
        lines.append("")

    if result["failed"]:
        lines.append("## Failed cases\n")
        for f in result["failed"]:
            lines.append(f"- `{f['id']}`: {f['error']}")
        lines.append("")

    lines.append("## Per-query results (in-corpus)\n")
    lines.append(f"| Query ID | Hit@{k} | MRR | nDCG@{k} |")
    lines.append("|---|---|---|---|")
    for s in result["per_query"]:
        lines.append(
            f"| {s['query_id']} | {s['hit_rate_at_k']:.2f} | "
            f"{s['mrr']:.2f} | {s['ndcg_at_k']:.2f} |"
        )

    return "\n".join(lines)


def write_report(result: dict) -> Path:
    """Write the markdown report. Returns the file path."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"retrieval_{timestamp}.md"
    path.write_text(_format_report(result), encoding="utf-8")
    return path


# ============================================================
# CLI entrypoint
# ============================================================

async def main() -> None:
    result = await run_retrieval_eval()
    report_path = write_report(result)

    print("\n" + "=" * 60)
    print(f"Retrieval eval complete")
    print(f"  Hit Rate@{result['k']}: {result['overall']['hit_rate_at_k']:.3f}")
    print(f"  MRR:              {result['overall']['mrr']:.3f}")
    print(f"  nDCG@{result['k']}:        {result['overall']['ndcg_at_k']:.3f}")
    print(f"  Report: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())