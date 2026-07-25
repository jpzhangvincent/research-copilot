"""NDCG@k evaluation (paper's primary metric, Tables 2/3).

Pure-Python, dependency-free implementation of (n)DCG so the reproduction can be
scored without pulling in pytrec_eval. Graded relevance is supported (BEIR qrels
are graded), using the standard exponential gain 2^rel - 1.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    """Discounted Cumulative Gain over an already-ranked list of relevances."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        if rel <= 0:
            continue
        dcg += (2.0 ** rel - 1.0) / math.log2(i + 2)  # i+2 => rank starts at 1
    return dcg


def ndcg_at_k(ranked_doc_ids: Sequence[str], qrels: Dict[str, int], k: int = 5) -> float:
    """NDCG@k for one query.

    ranked_doc_ids: document ids in predicted rank order (best first).
    qrels:          {doc_id: graded_relevance} for this query (missing => 0).
    """
    gains = [float(qrels.get(doc_id, 0)) for doc_id in ranked_doc_ids]
    dcg = dcg_at_k(gains, k)
    ideal = dcg_at_k(sorted(qrels.values(), reverse=True), k)
    if ideal == 0.0:
        return 0.0
    return dcg / ideal


def mean_ndcg_at_k(
    runs: Dict[str, List[str]], qrels: Dict[str, Dict[str, int]], k: int = 5
) -> float:
    """Mean NDCG@k across all queries that have judgments.

    runs:  {query_id: [ranked doc_ids]}
    qrels: {query_id: {doc_id: relevance}}
    """
    scores = []
    for qid, ranked in runs.items():
        judged = qrels.get(qid)
        if not judged:
            continue
        scores.append(ndcg_at_k(ranked, judged, k))
    return sum(scores) / len(scores) if scores else 0.0
