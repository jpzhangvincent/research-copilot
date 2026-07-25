"""BinnedConfidenceScore: three-way confidence binning (paper Algorithm 1, lines 1-9, Sec 3.2).

Wraps the MSCP confidence metric (confidence/mscp.py) to produce C(q) (query-only
confidence) and C(q,d) (joint query+document confidence), then bins C(q,d) into
{1 (high), 0 (medium), -1 (low)} using two tunable thresholds T_upper/T_lower.
"""

from dataclasses import dataclass
from typing import List, Optional

from confidence.mscp import mscp_score
from confidence.nli_cluster import NLIConfig
from confidence.sampling import LLMClient, SamplingConfig


@dataclass
class LCRThresholds:
    """Tunable LCR hyperparameters (T_query, T_upper, T_lower).

    Paper Sec 4.2 states these are grid-searched per (dataset, reranker) combo
    and reports only the best-after-tuning values in Tables 2/3; exact grids are
    not disclosed, so callers should treat these defaults as a starting point
    for their own sweep (see implementation_strategy ambiguity handling: coarse
    grid {0.4,0.5,0.6,0.7,0.8}).
    """

    t_query: float = 0.5
    t_upper: float = 0.7
    t_lower: float = 0.3


def compute_confidence(
    llm: LLMClient,
    query: str,
    document: Optional[str] = None,
    sampling_config: Optional[SamplingConfig] = None,
    nli_config: Optional[NLIConfig] = None,
    use_incremental: bool = True,
) -> float:
    """C(q) when document is None, else C(q,d). Both are MSCP(x; phi, K) (Eq. 1)."""
    return mscp_score(
        llm,
        query,
        document=document,
        sampling_config=sampling_config,
        nli_config=nli_config,
        use_incremental=use_incremental,
    )


def binned_confidence_score(confidence: float, thresholds: LCRThresholds) -> int:
    """BinnedConfidenceScore(q,d) given a precomputed C(q,d) value.

    if C(q,d) >= T_upper: return 1   (high confidence)
    elif C(q,d) <= T_lower: return -1 (low confidence)
    else: return 0                    (medium confidence)
    """
    if confidence >= thresholds.t_upper:
        return 1
    if confidence <= thresholds.t_lower:
        return -1
    return 0


def binned_confidence_score_for_doc(
    llm: LLMClient,
    query: str,
    document: str,
    thresholds: LCRThresholds,
    sampling_config: Optional[SamplingConfig] = None,
    nli_config: Optional[NLIConfig] = None,
    use_incremental: bool = True,
) -> int:
    """End-to-end BinnedConfidenceScore(q,d): computes C(q,d) via MSCP then bins it."""
    confidence = compute_confidence(
        llm,
        query,
        document=document,
        sampling_config=sampling_config,
        nli_config=nli_config,
        use_incremental=use_incremental,
    )
    return binned_confidence_score(confidence, thresholds)


def binned_confidence_scores(
    confidences: List[float], thresholds: LCRThresholds
) -> List[int]:
    """Vectorized binning over a list of precomputed C(q,d) values."""
    return [binned_confidence_score(c, thresholds) for c in confidences]
