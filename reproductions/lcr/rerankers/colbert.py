"""ColBERT late-interaction reranker baseline (Khattab & Zaharia, 2020; Tables 2/3).

Scores (query, document) by MaxSim late interaction over token embeddings from a
ColBERT checkpoint. The ColBERT/torch stack is imported lazily; if unavailable,
a clear error tells the user how to install it.
"""
from __future__ import annotations

from typing import List, Optional

from retrievers.bm25_retriever import RetrievedDoc


class ColBERTReranker:
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0",
                 device: Optional[str] = None, max_length: int = 512):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:  # pragma: no cover - optional heavy dep
            raise ImportError(
                "ColBERTReranker needs torch + transformers "
                "(`pip install torch transformers`)."
            ) from e
        self._torch = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(self._device).eval()
        self.max_length = max_length

    def _embed(self, text: str):
        torch = self._torch
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True,
                                 max_length=self.max_length).to(self._device)
        with torch.no_grad():
            out = self._model(**inputs).last_hidden_state[0]  # (seq, dim)
        return torch.nn.functional.normalize(out, dim=-1)

    def _maxsim(self, q_emb, d_emb) -> float:
        # sum over query tokens of max over doc tokens of cosine similarity
        sim = q_emb @ d_emb.T  # (q_len, d_len)
        return float(sim.max(dim=1).values.sum().item())

    def rerank(self, query: str, docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
        q_emb = self._embed(query)
        for d in docs:
            d.score = self._maxsim(q_emb, self._embed(d.text))
            d.extra["reranker"] = "colbert"
        ordered = sorted(docs, key=lambda d: d.score, reverse=True)
        for rank, d in enumerate(ordered, start=1):
            d.rank = rank
        return ordered
