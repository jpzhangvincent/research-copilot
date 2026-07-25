"""RankT5 pointwise reranker baseline (Zhuang et al., 2023; paper Tables 2/3).

Scores each (query, document) pair with a T5-based ranking model. Uses the
encoder-decoder logit of the ranking token as the relevance score. Transformers/
torch are imported lazily.
"""
from __future__ import annotations

from typing import List, Optional

from retrievers.bm25_retriever import RetrievedDoc


class RankT5Reranker:
    def __init__(self, model_name: str = "castorini/monot5-base-msmarco",
                 device: Optional[str] = None, max_length: int = 512):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._torch = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self._device).eval()
        self.max_length = max_length
        # monoT5 emits 'true'/'false'; use the 'true' token logit as the score.
        self._true_id = self._tokenizer.encode("true", add_special_tokens=False)[0]

    def _score(self, query: str, text: str) -> float:
        torch = self._torch
        prompt = f"Query: {query} Document: {text} Relevant:"
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True,
                                 max_length=self.max_length).to(self._device)
        decoder_start = self._model.config.decoder_start_token_id
        decoder_input_ids = torch.tensor([[decoder_start]], device=self._device)
        with torch.no_grad():
            logits = self._model(**inputs, decoder_input_ids=decoder_input_ids).logits
        return float(logits[0, 0, self._true_id].item())

    def rerank(self, query: str, docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
        for d in docs:
            d.score = self._score(query, d.text)
            d.extra["reranker"] = "rankt5"
        ordered = sorted(docs, key=lambda d: d.score, reverse=True)
        for rank, d in enumerate(ordered, start=1):
            d.rank = rank
        return ordered
