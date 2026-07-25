"""Query-Likelihood Model reranker (QLM baseline, paper Tables 2/3).

Scores each document by the log-likelihood of the query under a Dirichlet-smoothed
unigram language model estimated from the document (Zhai & Lafferty, 2004). This
is a classic, dependency-free reranker — a fair, reproducible baseline that LCR is
layered on top of.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import List

from retrievers.bm25_retriever import RetrievedDoc, simple_tokenize


class QLMReranker:
    def __init__(self, mu: float = 2000.0):
        self.mu = mu  # Dirichlet smoothing strength

    def rerank(self, query: str, docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
        q_tokens = simple_tokenize(query)
        # Background model from the candidate set (collection proxy).
        bg = Counter()
        for d in docs:
            bg.update(simple_tokenize(d.text))
        bg_total = sum(bg.values()) or 1

        for d in docs:
            d_tokens = simple_tokenize(d.text)
            d_counts = Counter(d_tokens)
            d_len = len(d_tokens) or 1
            log_lik = 0.0
            for term in q_tokens:
                p_bg = bg.get(term, 0) / bg_total
                numer = d_counts.get(term, 0) + self.mu * p_bg
                denom = d_len + self.mu
                prob = numer / denom if denom > 0 else 1e-12
                log_lik += math.log(prob + 1e-12)
            d.score = log_lik
            d.extra["reranker"] = "qlm"

        ordered = sorted(docs, key=lambda d: d.score, reverse=True)
        for rank, d in enumerate(ordered, start=1):
            d.rank = rank
        return ordered
