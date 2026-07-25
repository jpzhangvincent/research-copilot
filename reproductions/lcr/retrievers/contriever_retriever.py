"""Contriever dense retriever (facebook/contriever) producing top-K candidates with PrevScore.

Implements the dense retrieval baseline used to build Table 3 of the paper: documents and
queries are embedded with the contrastively-pretrained Contriever checkpoint (mean-pooled
token embeddings) and ranked by dot-product similarity.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .bm25_retriever import CorpusDoc, RetrievedDoc


class ContrieverRetriever:
    """Dense retriever using the HF `facebook/contriever` checkpoint.

    Encodes queries and corpus documents with mean-pooled Contriever embeddings and ranks
    candidates by dot-product similarity, matching the Contriever paper's contrastive
    pretraining objective (no supervised fine-tuning is applied here).
    """

    def __init__(
        self,
        corpus: Optional[Sequence[CorpusDoc]] = None,
        model_name: str = "facebook/contriever",
        device: str = "cuda",
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._tokenizer = None
        self._model = None
        self._torch = None
        self.corpus: List[CorpusDoc] = list(corpus) if corpus else []
        self._doc_embeddings: Optional[np.ndarray] = None
        if self.corpus:
            self.index(self.corpus)

    def _lazy_load_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self._model.eval()
        self._torch = torch

    @staticmethod
    def _mean_pooling(token_embeddings, attention_mask):
        mask = attention_mask[..., None].to(token_embeddings.dtype)
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        self._lazy_load_model()
        torch = self._torch
        embeddings = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                inputs = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self._model(**inputs)
                token_embeddings = outputs[0]
                pooled = self._mean_pooling(token_embeddings, inputs["attention_mask"])
                embeddings.append(pooled.cpu().numpy())
        return np.concatenate(embeddings, axis=0)

    def index(self, corpus: Sequence[CorpusDoc]) -> None:
        """(Re)build the dense index over a corpus of documents."""
        self.corpus = list(corpus)
        texts = [doc.full_text for doc in self.corpus]
        self._doc_embeddings = self._encode(texts)

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievedDoc]:
        """Retrieve the top_k documents for a single query, ranked by dot-product similarity."""
        return self.retrieve_batch([query], top_k=top_k)[0]

    def retrieve_batch(self, queries: Sequence[str], top_k: int = 10) -> List[List[RetrievedDoc]]:
        """Batched retrieval across multiple queries."""
        if self._doc_embeddings is None or len(self.corpus) == 0 or self._doc_embeddings.size == 0:
            return [[] for _ in queries]
        query_embeddings = self._encode(list(queries))
        scores = query_embeddings @ self._doc_embeddings.T
        results: List[List[RetrievedDoc]] = []
        for row in scores:
            k = min(top_k, len(self.corpus))
            order = np.argsort(-row)[:k]
            hits = [
                RetrievedDoc(
                    doc_id=self.corpus[idx].doc_id,
                    text=self.corpus[idx].text,
                    score=float(row[idx]),
                    rank=rank,
                    extra={},
                )
                for rank, idx in enumerate(order)
            ]
            results.append(hits)
        return results
