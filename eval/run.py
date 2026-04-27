"""Eval CLI — one-command entrypoint for reviewers.

Runs the eval suites and prints a summary. Use this instead of invoking
the runners directly.

Usage:
    python -m eval.run                 # default: runs both suites
    python -m eval.run --suite retrieval
    python -m eval.run --suite generation
    python -m eval.run --suite all

Each suite writes a detailed report to eval/reports/. The CLI itself
prints just the headline numbers so you can scan results quickly.

Prerequisites:
    - Backend stack running (docker compose up)
    - Documents ingested into Qdrant
    - ANTHROPIC_API_KEY set in .env (for the generation eval's LLM judge)
"""

import argparse
import asyncio
import logging
import sys

from eval.runners.e2e_runner import run_e2e_eval, write_report as write_e2e_report
from eval.runners.retrieval_runner import (
    run_retrieval_eval,
    write_report as write_retrieval_report,
)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


# ============================================================
# Headline printing
# ============================================================

def _print_retrieval_summary(result: dict) -> None:
    overall = result["overall"]
    k = result["k"]
    print()
    print("-" * 60)
    print(f"  RETRIEVAL EVAL")
    print("-" * 60)
    print(f"  Cases scored      : {overall['count']} (in-corpus)")
    print(f"  Hit Rate@{k}        : {overall['hit_rate_at_k']:.3f}")
    print(f"  MRR               : {overall['mrr']:.3f}")
    print(f"  nDCG@{k}            : {overall['ndcg_at_k']:.3f}")


def _print_generation_summary(result: dict) -> None:
    overall = result["overall"]
    print()
    print("-" * 60)
    print(f"  GENERATION EVAL")
    print("-" * 60)
    print(f"  Cases run         : {result['n_successful']}/{result['n_cases']}")
    print(
        f"  Faithfulness      : {overall['faithfulness']:.3f} "
        f"(n={overall['faithfulness_n']})"
    )
    print(f"  Answer relevance  : {overall['answer_relevance']:.3f}")
    print(f"  Behavior accuracy : {overall['behavior_accuracy']:.3f}")


# ============================================================
# Suite runners
# ============================================================

async def run_retrieval_only() -> dict:
    """Run retrieval eval; write report; print summary."""
    log.info("Running retrieval eval...")
    result = await run_retrieval_eval()
    report_path = write_retrieval_report(result)
    _print_retrieval_summary(result)
    print(f"\n  Report: {report_path}")
    return result


async def run_generation_only() -> dict:
    """Run e2e generation eval; write report; print summary."""
    log.info("Running generation eval...")
    result = await run_e2e_eval()
    report_path = write_e2e_report(result)
    _print_generation_summary(result)
    print(f"\n  Report: {report_path}")
    return result


async def run_all() -> tuple[dict, dict]:
    """Run both suites in sequence."""
    retrieval_result = await run_retrieval_only()
    generation_result = await run_generation_only()

    # Combined headline at the end
    print()
    print("=" * 60)
    print("  EVAL COMPLETE")
    print("=" * 60)
    print(
        f"  Retrieval:  HR@{retrieval_result['k']}={retrieval_result['overall']['hit_rate_at_k']:.3f}  "
        f"MRR={retrieval_result['overall']['mrr']:.3f}  "
        f"nDCG@{retrieval_result['k']}={retrieval_result['overall']['ndcg_at_k']:.3f}"
    )
    print(
        f"  Generation: Faithfulness={generation_result['overall']['faithfulness']:.3f}  "
        f"Relevance={generation_result['overall']['answer_relevance']:.3f}  "
        f"Behavior={generation_result['overall']['behavior_accuracy']:.3f}"
    )
    print("=" * 60)
    return retrieval_result, generation_result


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation suites. Reports written to eval/reports/.",
    )
    parser.add_argument(
        "--suite",
        choices=["retrieval", "generation", "all"],
        default="all",
        help="Which eval suite(s) to run. Default: all.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    if args.suite == "retrieval":
        await run_retrieval_only()
    elif args.suite == "generation":
        await run_generation_only()
    else:  # "all"
        await run_all()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)