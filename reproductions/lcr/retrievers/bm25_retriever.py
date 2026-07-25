"""BM25 sparse retriever (paper §3.3 / Table 2 setup).

Produces top-K candidate documents + PrevScore(q,d) per query, using the
classic BM25 ranking function. Prefers `pyserini` (Lucene-backed, matches the
paper's likely production setup) and falls back to the pure-Python
`rank_bm25` package when a Lucene/JVM index is not available, so the rest of
the pipeline (rerankers/, lcr/) can run without extra system dependencies.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def simple_tokenize(text: str) -> List[str]:
    """Lowercase alnum tokenizer used by the rank_bm25 fallback backend."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class CorpusDoc:
    """A single corpus document as loaded from a BEIR/TREC collection."""

    doc_id: str
    text: str
    title: str = ""

    @property
    def full_text(self) -> str:
        if self.title:
            return f"{self.title} {self.text}".strip()
        return self.text


@dataclass
class RetrievedDoc:
    """One retrieved candidate with its retriever score (PrevScore)."""

    doc_id: str
    text: str
    score: float
    rank: int
    extra: dict = field(default_factory=dict)


class BM25Retriever:
    """BM25 retriever with a pyserini backend and a rank_bm25 fallback.

    Usage:
        retriever = BM25Retriever(corpus)  # corpus: List[CorpusDoc]
        results = retriever.retrieve(query, top_k=10)
    """

    def __init__(
        self,
        corpus: Optional[Sequence[CorpusDoc]] = None,
        index_dir: Optional[str] = None,
        backend: str = "auto",
        k1: float = 0.9,
        b: float = 0.4,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.index_dir = index_dir
        self._corpus_by_id: Dict[str, CorpusDoc] = {}
        self._backend = backend
        self._searcher = None
        self._bm25 = None
        self._doc_ids: List[str] = []

        if backend == "pyserini" or (backend == "auto" and index_dir is not None):
            self._init_pyserini(index_dir)
        elif corpus is not None:
            self._init_rank_bm25(corpus)
        else:
            raise ValueError(
                "BM25Retriever requires either `corpus` (rank_bm25 backend) "
                "or `index_dir` (pyserini backend)."
            )

    def _init_pyserini(self, index_dir: Optional[str]) -> None:
        if not index_dir:
            raise ValueError("pyserini backend requires `index_dir`.")
        from pyserini.search.lucene import LuceneSearcher  # lazy import

        self._searcher = LuceneSearcher(index_dir)
        self._searcher.set_bm25(self.k1, self.b)
        self._backend = "pyserini"

    def _init_rank_bm25(self, corpus: Sequence[CorpusDoc]) -> None:
        from rank_bm25 import BM25Okapi  # lazy import

        self._corpus_by_id = {doc.doc_id: doc for doc in corpus}
        self._doc_ids = [doc.doc_id for doc in corpus]
        tokenized = [simple_tokenize(doc.full_text) for doc in corpus]
        self._bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        self._backend = "rank_bm25"

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievedDoc]:
        if self._backend == "pyserini":
            return self._retrieve_pyserini(query, top_k)
        return self._retrieve_rank_bm25(query, top_k)

    def retrieve_batch(
        self, queries: Sequence[str], top_k: int = 10
    ) -> List[List[RetrievedDoc]]:
        return [self.retrieve(query, top_k=top_k) for query in queries]

    def _retrieve_pyserini(self, query: str, top_k: int) -> List[RetrievedDoc]:
        hits = self._searcher.search(query, k=top_k)
        results: List[RetrievedDoc] = []
        for rank, hit in enumerate(hits, start=1):
            doc_id = str(hit.docid)
            text = self._corpus_by_id.get(doc_id, CorpusDoc(doc_id, "")).text
            if not text:
                raw = getattr(hit, "raw", None)
                text = raw if isinstance(raw, str) else ""
            results.append(
                RetrievedDoc(doc_id=doc_id, text=text, score=float(hit.score), rank=rank)
            )
        return results

    def _retrieve_rank_bm25(self, query: str, top_k: int) -> List[RetrievedDoc]:
        tokenized_query = simple_tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        ranked_idx = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        results: List[RetrievedDoc] = []
        for rank, idx in enumerate(ranked_idx, start=1):
            doc_id = self._doc_ids[idx]
            doc = self._corpus_by_id[doc_id]
            results.append(
                RetrievedDoc(
                    doc_id=doc_id, text=doc.text, score=float(scores[idx]), rank=rank
                )
            )
        return results
