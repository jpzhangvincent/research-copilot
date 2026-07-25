"""Upstream rerankers LCR can be layered on top of (paper Tables 2/3).

LCR is plug-and-play: it re-sorts the output (query, docs, PrevScore) of *any*
of these rerankers. Every reranker here implements the same `Reranker` protocol:

    rerank(query, docs) -> List[RetrievedDoc]   # re-scored (PrevScore) + re-ordered

`RetrieverOnlyReranker` is the identity baseline (PrevScore = retriever score),
i.e. the "Retriever-Only" rows in the paper. Heavier model-backed rerankers use
lazy imports so importing this package never requires GPU/torch dependencies.
"""
from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from retrievers.bm25_retriever import RetrievedDoc


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
        ...


def _reorder(docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
    """Sort by score desc and refresh 1-based ranks (stable)."""
    ordered = sorted(docs, key=lambda d: d.score, reverse=True)
    for rank, d in enumerate(ordered, start=1):
        d.rank = rank
    return ordered


class RetrieverOnlyReranker:
    """Identity reranker: keep the base retriever's PrevScore and order."""

    def rerank(self, query: str, docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
        return _reorder(list(docs))


def get_reranker(name: str, **kwargs) -> Reranker:
    """Factory: map a config name to a reranker instance (lazy-imports heavy ones)."""
    name = (name or "retriever_only").lower()
    if name in ("retriever_only", "none", "identity"):
        return RetrieverOnlyReranker()
    if name == "qlm":
        from .qlm import QLMReranker
        return QLMReranker(**kwargs)
    if name == "yesno":
        from .yesno import YesNoReranker
        return YesNoReranker(**kwargs)
    if name == "rankgpt":
        from .rankgpt import RankGPTReranker
        return RankGPTReranker(**kwargs)
    if name == "cross_encoder":
        from .cross_encoder import CrossEncoderReranker
        return CrossEncoderReranker(**kwargs)
    if name == "rankt5":
        from .rankt5 import RankT5Reranker
        return RankT5Reranker(**kwargs)
    if name == "colbert":
        from .colbert import ColBERTReranker
        return ColBERTReranker(**kwargs)
    raise ValueError(f"Unknown reranker: {name!r}")


__all__ = ["Reranker", "RetrieverOnlyReranker", "get_reranker"]
