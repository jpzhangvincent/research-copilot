"""YesNo (relevance-generation) reranker baseline (paper Tables 2/3).

Prompts an LLM to judge whether a document is relevant to the query and scores
each document by the model's propensity to answer "yes" (estimated over K samples).
Works with any `LLMClient` from `confidence.sampling` (the same backend used for MSCP).
"""
from __future__ import annotations

from typing import List, Optional

from confidence.sampling import LLMClient
from retrievers.bm25_retriever import RetrievedDoc

YESNO_PROMPT = (
    "Document: {document}\n"
    "Question: {query}\n"
    "Is the document relevant to answering the question? "
    "Answer with only 'yes' or 'no'."
)


class YesNoReranker:
    def __init__(self, llm: LLMClient, samples: int = 5, max_tokens: int = 4,
                 temperature: float = 1.0):
        self.llm = llm
        self.samples = samples
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _yes_fraction(self, completions: List[str]) -> float:
        if not completions:
            return 0.0
        yes = sum(1 for c in completions if c.strip().lower().startswith("y"))
        return yes / len(completions)

    def rerank(self, query: str, docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
        prompts = [YESNO_PROMPT.format(document=d.text, query=query) for d in docs]
        batched = self.llm.generate(
            prompts, n=self.samples, temperature=self.temperature,
            max_tokens=self.max_tokens, stop=None,
        )
        for d, comps in zip(docs, batched):
            d.score = self._yes_fraction(comps)
            d.extra["reranker"] = "yesno"

        ordered = sorted(docs, key=lambda d: d.score, reverse=True)
        for rank, d in enumerate(ordered, start=1):
            d.rank = rank
        return ordered
