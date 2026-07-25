"""LCR (LLM-Confidence Reranker) package.

Implements Algorithm 1 from the paper:
  - BinnedConfidenceScore (Algorithm 1, lines 1-9): see `binning.py`.
  - LCR_SORT (Algorithm 1, lines 10-18): see `lcr_sort.py`.

This package is a post-hoc, plug-and-play reranking layer that wraps any
upstream retriever/reranker output (query, document list, PrevScore per
document) and re-sorts it using MSCP-derived confidence scores.
"""

from .binning import (
    LCRThresholds,
    compute_confidence,
    binned_confidence_score,
    binned_confidence_score_for_doc,
    binned_confidence_scores,
)
from .lcr_sort import (
    ScoredDocument,
    lcr_sort,
    lcr_sort_from_confidences,
)

__all__ = [
    "LCRThresholds",
    "compute_confidence",
    "binned_confidence_score",
    "binned_confidence_score_for_doc",
    "binned_confidence_scores",
    "ScoredDocument",
    "lcr_sort",
    "lcr_sort_from_confidences",
]
