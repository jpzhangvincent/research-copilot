"""LCR_SORT (Algorithm 1, lines 10-18): confidence-based reranking wrapper.

Wraps ANY upstream reranker output (Retriever-Only, QLM, RankGPT, YesNo,
Cross-Encoder, ColBERT, RankT5) and re-sorts it using MSCP-derived confidence,
falling back to the original PrevScore ranking whenever the query-level
confidence C(q) is high enough (the "safe lower bound" property: T_query=0
reproduces the original ranker output exactly).
"""

from dataclasses import dataclass, field
from typing import List, Optional

from confidence.sampling import LLMClient, SamplingConfig
from confidence.nli_cluster import NLIConfig
from lcr.binning import LCRThresholds, compute_confidence, binned_confidence_score


@dataclass
class ScoredDocument:
    """A single retrieved/reranked document plus its LCR bookkeeping fields."""

    doc_id: str
    text: str
    prev_score: float
    confidence: Optional[float] = None
    binned_score: Optional[int] = None
    extra: dict = field(default_factory=dict)


def _sort_by_prev_score(documents: List[ScoredDocument]) -> List[ScoredDocument]:
    """Baseline ranking: PrevScore(q,d) descending only (stable)."""
    return sorted(documents, key=lambda d: d.prev_score, reverse=True)


def _sort_by_binned_then_prev(documents: List[ScoredDocument]) -> List[ScoredDocument]:
    """Confidence-sharpened ranking: BinnedConfidenceScore desc, then PrevScore desc (stable)."""
    return sorted(
        documents,
        key=lambda d: (d.binned_score, d.prev_score),
        reverse=True,
    )


def lcr_sort(
    llm: LLMClient,
    query: str,
    documents: List[ScoredDocument],
    thresholds: LCRThresholds,
    sampling_config: Optional[SamplingConfig] = None,
    nli_config: Optional[NLIConfig] = None,
    use_incremental: bool = True,
) -> List[ScoredDocument]:
    """End-to-end LCR_SORT: computes C(q) and C(q,d) via MSCP, then re-sorts.

    Steps (Algorithm 1, lines 10-18):
      1. C(q) = MSCP(q; phi, K) (query-only confidence).
      2. For each d in documents: C(q,d) = MSCP(q,d; phi, K); PrevScore retained as-is.
      3. If C(q) < T_query: stable-sort by (BinnedConfidenceScore desc, PrevScore desc).
         Else: stable-sort by PrevScore desc only (reverts to baseline ranking).
    """
    query_confidence = compute_confidence(
        llm,
        query,
        document=None,
        sampling_config=sampling_config,
        nli_config=nli_config,
        use_incremental=use_incremental,
    )

    if query_confidence >= thresholds.t_query:
        return _sort_by_prev_score(documents)

    for doc in documents:
        doc.confidence = compute_confidence(
            llm,
            query,
            document=doc.text,
            sampling_config=sampling_config,
            nli_config=nli_config,
            use_incremental=use_incremental,
        )
        doc.binned_score = binned_confidence_score(doc.confidence, thresholds)

    return _sort_by_binned_then_prev(documents)


def lcr_sort_from_confidences(
    query_confidence: float,
    documents: List[ScoredDocument],
    thresholds: LCRThresholds,
) -> List[ScoredDocument]:
    """Pure, LLM-free variant of LCR_SORT for unit tests / offline analysis.

    Callers must have already populated `doc.confidence` (i.e. C(q,d)) on every
    document. This is the entry point used to test the safe-lower-bound property
    (T_query=0 => query_confidence >= 0 always holds => baseline ranking exactly)
    and the three-way binning boundaries with synthetic PrevScore/confidence values.
    """
    if query_confidence >= thresholds.t_query:
        return _sort_by_prev_score(documents)

    for doc in documents:
        if doc.confidence is None:
            raise ValueError(
                f"Document {doc.doc_id!r} is missing a precomputed confidence value; "
                "set doc.confidence before calling lcr_sort_from_confidences()."
            )
        doc.binned_score = binned_confidence_score(doc.confidence, thresholds)

    return _sort_by_binned_then_prev(documents)
