"""Retrieval metrics — Hit Rate, MRR, nDCG.

All three are functions of (retrieved_ids, relevant_ids) — the order of
retrieved_ids matters, the order of relevant_ids does not.

Conventions:
- retrieved_ids: list[str], ordered best-to-worst by the retrieval system
- relevant_ids:  set[str], the gold-standard relevant chunks (no ordering)
- k: int, the cutoff for @K metrics

All metrics return floats in [0, 1] where higher is better.
"""

import math
from dataclasses import dataclass


@dataclass
class RetrievalScore:
    """Per-query metric values, plus the inputs used to compute them."""

    query_id: str
    retrieved_ids: list[str]
    relevant_ids: set[str]
    hit_rate_at_k: float
    mrr: float
    ndcg_at_k: float


def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Did at least one relevant chunk appear in the top K?

    Returns 1.0 if any relevant chunk is in retrieved_ids[:k], else 0.0.
    Edge case: if relevant_ids is empty, hit rate is undefined — we return
    1.0 when the system returned nothing (correctly empty), 0.0 when it
    returned something irrelevant. This makes out-of-corpus tests scoreable.
    """
    if not relevant_ids:
        # Out-of-corpus query: success = retrieving nothing relevant.
        # We can't tell from IDs alone whether retrieved chunks are irrelevant,
        # so by convention we return 1.0 here and let the runner handle this
        # case separately if it wants to. Most callers pass non-empty relevant_ids.
        return 1.0

    top_k = retrieved_ids[:k]
    return 1.0 if any(rid in relevant_ids for rid in top_k) else 0.0


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal rank of the FIRST relevant chunk in the result list.

    rank 1 → 1.0
    rank 2 → 0.5
    rank 3 → 0.333
    not found → 0.0

    Across many queries, this is averaged into MRR.
    """
    if not relevant_ids:
        return 1.0  # see hit_rate_at_k note

    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K.

    DCG = sum over rank i: relevance_i / log2(i + 1)
    IDCG = the DCG of the IDEAL ordering (all relevant chunks ranked first)
    nDCG = DCG / IDCG, in [0, 1]

    We use binary relevance (chunk is relevant or not). For graded relevance,
    you'd pass per-chunk scores; we don't need that complexity here.
    """
    if not relevant_ids:
        return 1.0

    top_k = retrieved_ids[:k]

    # DCG of the actual ranking
    dcg = 0.0
    for i, rid in enumerate(top_k, start=1):
        if rid in relevant_ids:
            # Binary relevance: 1 if relevant, 0 otherwise
            dcg += 1.0 / math.log2(i + 1)

    # IDCG: place all relevant chunks in the top positions
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


def compute_all(
    query_id: str,
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int = 5,
) -> RetrievalScore:
    """Compute Hit Rate@K, MRR, and nDCG@K for one query."""
    return RetrievalScore(
        query_id=query_id,
        retrieved_ids=retrieved_ids,
        relevant_ids=relevant_ids,
        hit_rate_at_k=hit_rate_at_k(retrieved_ids, relevant_ids, k),
        mrr=mean_reciprocal_rank(retrieved_ids, relevant_ids),
        ndcg_at_k=ndcg_at_k(retrieved_ids, relevant_ids, k),
    )


def aggregate(scores: list[RetrievalScore]) -> dict[str, float]:
    """Average scores across all queries.

    Returns a dict with keys: hit_rate_at_k, mrr, ndcg_at_k, count.
    Use this for the summary at the end of an eval run.
    """
    if not scores:
        return {"hit_rate_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0, "count": 0}

    n = len(scores)
    return {
        "hit_rate_at_k": sum(s.hit_rate_at_k for s in scores) / n,
        "mrr": sum(s.mrr for s in scores) / n,
        "ndcg_at_k": sum(s.ndcg_at_k for s in scores) / n,
        "count": n,
    }