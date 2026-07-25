"""Cross-encoder reranker baseline (MiniLM cross-encoder; paper Tables 2/3).

Scores each (query, document) pair jointly with a sentence-transformers
CrossEncoder. Torch/sentence-transformers are imported lazily so the package
imports cleanly on CPU-only / dependency-light environments.
"""
from __future__ import annotations

from typing import List, Optional

from retrievers.bm25_retriever import RetrievedDoc


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 max_length: int = 512, device: Optional[str] = None):
        from sentence_transformers import CrossEncoder  # lazy
        self._model = CrossEncoder(model_name, max_length=max_length, device=device)

    def rerank(self, query: str, docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
        if not docs:
            return docs
        pairs = [(query, d.text) for d in docs]
        scores = self._model.predict(pairs)
        for d, s in zip(docs, scores):
            d.score = float(s)
            d.extra["reranker"] = "cross_encoder"
        ordered = sorted(docs, key=lambda d: d.score, reverse=True)
        for rank, d in enumerate(ordered, start=1):
            d.rank = rank
        return ordered
