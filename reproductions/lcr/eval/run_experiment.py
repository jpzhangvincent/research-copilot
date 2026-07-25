"""End-to-end LCR experiment (paper §4): retrieve -> rerank -> LCR_SORT -> NDCG@k.

Compares the base ranking (retriever/reranker PrevScore order) against the
LCR-resorted ranking on the same candidates, reporting NDCG@k for both plus the
absolute and relative improvement — the headline result of Tables 2/3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from confidence.sampling import LLMClient, SamplingConfig
from confidence.nli_cluster import NLIConfig
from lcr.binning import LCRThresholds
from lcr.lcr_sort import ScoredDocument, lcr_sort
from retrievers.bm25_retriever import CorpusDoc, RetrievedDoc
from rerankers import get_reranker
from eval.ndcg import mean_ndcg_at_k


@dataclass
class ExperimentConfig:
    dataset: str = "nfcorpus"
    data_dir: str = "./datasets"
    split: str = "test"
    retriever: str = "bm25"
    reranker: str = "retriever_only"
    top_k: int = 10
    ndcg_k: int = 5
    max_queries: Optional[int] = None
    thresholds: LCRThresholds = field(default_factory=LCRThresholds)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    nli: NLIConfig = field(default_factory=NLIConfig)


def _to_scored(docs: List[RetrievedDoc]) -> List[ScoredDocument]:
    return [ScoredDocument(doc_id=d.doc_id, text=d.text, prev_score=d.score) for d in docs]


def run_experiment(
    config: ExperimentConfig,
    llm: LLMClient,
    corpus: List[CorpusDoc],
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    retriever=None,
) -> Dict[str, object]:
    """Run the base-vs-LCR comparison and return a metrics dict."""
    if retriever is None:
        from retrievers.bm25_retriever import BM25Retriever
        retriever = BM25Retriever(corpus)

    reranker = get_reranker(config.reranker, llm=llm) if config.reranker not in (
        "retriever_only", "none", "identity"
    ) else get_reranker(config.reranker)

    qids = list(queries.keys())
    if config.max_queries:
        qids = qids[: config.max_queries]

    base_run: Dict[str, List[str]] = {}
    lcr_run: Dict[str, List[str]] = {}

    for qid in qids:
        query = queries[qid]
        candidates = retriever.retrieve(query, top_k=config.top_k)
        if not candidates:
            continue
        reranked = reranker.rerank(query, candidates)
        base_run[qid] = [d.doc_id for d in reranked]

        scored = _to_scored(reranked)
        resorted = lcr_sort(
            llm, query, scored, config.thresholds,
            sampling_config=config.sampling, nli_config=config.nli,
        )
        lcr_run[qid] = [d.doc_id for d in resorted]

    base_ndcg = mean_ndcg_at_k(base_run, qrels, config.ndcg_k)
    lcr_ndcg = mean_ndcg_at_k(lcr_run, qrels, config.ndcg_k)
    abs_gain = lcr_ndcg - base_ndcg
    rel_gain = (abs_gain / base_ndcg * 100.0) if base_ndcg > 0 else 0.0

    return {
        "dataset": config.dataset,
        "reranker": config.reranker,
        "num_queries": len(base_run),
        "ndcg_k": config.ndcg_k,
        "base_ndcg": round(base_ndcg, 4),
        "lcr_ndcg": round(lcr_ndcg, 4),
        "abs_gain": round(abs_gain, 4),
        "rel_gain_pct": round(rel_gain, 2),
    }
