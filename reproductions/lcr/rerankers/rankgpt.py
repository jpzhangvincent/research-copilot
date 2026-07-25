"""RankGPT listwise reranker baseline (Sun et al., 2023; paper Tables 2/3).

Asks an LLM to output a permutation of candidate passages by relevance, then maps
that permutation back to descending PrevScores. Uses any `LLMClient`; the parser is
robust to the common "[3] > [1] > [2]" and "3 1 2" output styles.
"""
from __future__ import annotations

import re
from typing import List

from confidence.sampling import LLMClient
from retrievers.bm25_retriever import RetrievedDoc


def _build_prompt(query: str, docs: List[RetrievedDoc]) -> str:
    passages = "\n".join(f"[{i + 1}] {d.text[:400]}" for i, d in enumerate(docs))
    return (
        f"The following are {len(docs)} passages, each with an identifier in [].\n"
        f"{passages}\n\n"
        f"Search query: {query}\n"
        f"Rank the passages by relevance to the query, most relevant first. "
        f"Output only the identifiers in order, e.g. [2] > [1] > [3]."
    )


def parse_permutation(text: str, n: int) -> List[int]:
    """Extract a 0-based permutation from the model output; pad/dedupe defensively."""
    ids = [int(x) - 1 for x in re.findall(r"\[?(\d+)\]?", text)]
    seen, order = set(), []
    for i in ids:
        if 0 <= i < n and i not in seen:
            seen.add(i)
            order.append(i)
    for i in range(n):  # append any missing indices in original order
        if i not in seen:
            order.append(i)
    return order


class RankGPTReranker:
    def __init__(self, llm: LLMClient, max_tokens: int = 128):
        self.llm = llm
        self.max_tokens = max_tokens

    def rerank(self, query: str, docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
        if not docs:
            return docs
        prompt = _build_prompt(query, docs)
        out = self.llm.generate([prompt], n=1, temperature=0.0,
                                max_tokens=self.max_tokens, stop=None)[0][0]
        order = parse_permutation(out, len(docs))
        n = len(docs)
        reranked: List[RetrievedDoc] = []
        for rank, idx in enumerate(order):
            d = docs[idx]
            d.score = float(n - rank)  # descending PrevScore from rank position
            d.rank = rank + 1
            d.extra["reranker"] = "rankgpt"
            reranked.append(d)
        return reranked
