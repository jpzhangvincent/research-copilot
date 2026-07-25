"""MSCP(x; phi, K) computation (paper Sec 3.1, Eq. 1).

Wires together K-sample generation (sampling.py) and semantic clustering
(nli_cluster.py) into the Maximum Semantic Cluster Proportion metric:

    MSCP(x; phi, K) = max_{m=1..M} |s_m| / K

where x is either just the query q (query-only confidence C(q)) or the pair
(q, d) (joint confidence C(q, d)), and {s_1..s_M} are the connected
components of the bidirectional-entailment graph over K sampled answers.
"""
from dataclasses import dataclass
from typing import List, Optional

from .sampling import LLMClient, SamplingConfig, sample_answers, sample_answers_batch
from .nli_cluster import (
    NLIConfig,
    cluster_answers_bruteforce,
    cluster_answers_incremental,
)


@dataclass
class MSCPResult:
    """Result of an MSCP computation: score plus the underlying evidence.

    Keeping the raw ``clusters`` and ``answers`` (rather than just the
    scalar score) is required for debugging and for reproducing paper
    illustrations such as Figure 2 (President/Trump/Musk toy example).
    """

    score: float
    clusters: List[List[int]]
    answers: List[str]

    @property
    def K(self) -> int:
        return len(self.answers)

    @property
    def M(self) -> int:
        return len(self.clusters)


def _mscp_from_clusters(clusters: List[List[int]], k: int) -> float:
    if k == 0:
        return 0.0
    largest = max((len(c) for c in clusters), default=0)
    return largest / k


def compute_mscp(
    llm: LLMClient,
    query: str,
    document: Optional[str] = None,
    sampling_config: Optional[SamplingConfig] = None,
    nli_config: Optional[NLIConfig] = None,
    use_incremental: bool = True,
) -> MSCPResult:
    """Compute MSCP(x; phi, K) for x = query, or x = (query, document).

    Steps (Eq. 1):
      1. Draw K i.i.d. samples t_1..t_K from phi via multinomial sampling
         (sampling.py, temperature T=1).
      2. Cluster the samples into semantic equivalence classes {s_1..s_M}
         via bidirectional-entailment connected components
         (nli_cluster.py); incremental clustering is O(K*M), brute force
         is O(K^2) and used for correctness checks.
      3. Return max_m |s_m| / K.
    """
    sampling_config = sampling_config or SamplingConfig()
    answers = sample_answers(llm, query, document=document, config=sampling_config)

    cluster_fn = cluster_answers_incremental if use_incremental else cluster_answers_bruteforce
    clusters = cluster_fn(llm, query, answers, config=nli_config)

    score = _mscp_from_clusters(clusters, len(answers))
    return MSCPResult(score=score, clusters=clusters, answers=answers)


def mscp_score(
    llm: LLMClient,
    query: str,
    document: Optional[str] = None,
    sampling_config: Optional[SamplingConfig] = None,
    nli_config: Optional[NLIConfig] = None,
    use_incremental: bool = True,
) -> float:
    """Convenience wrapper returning only the scalar MSCP(x; phi, K).

    This is the primary entry point used by lcr/binning.py to obtain
    C(q) (document=None) and C(q, d) (document=d).
    """
    return compute_mscp(
        llm,
        query,
        document=document,
        sampling_config=sampling_config,
        nli_config=nli_config,
        use_incremental=use_incremental,
    ).score


def compute_mscp_batch(
    llm: LLMClient,
    queries: List[str],
    documents: Optional[List[Optional[str]]] = None,
    sampling_config: Optional[SamplingConfig] = None,
    nli_config: Optional[NLIConfig] = None,
    use_incremental: bool = True,
) -> List[MSCPResult]:
    """Batched MSCP computation across many (query[, document]) pairs.

    Sampling is batched via sample_answers_batch (a single LLM.generate
    call across all pairs, critical for vLLM throughput per the paper's
    ~11*10 generations/query MSCP budget). Clustering is performed
    per-pair since the entailment graph is query-specific.
    """
    sampling_config = sampling_config or SamplingConfig()
    answers_batch = sample_answers_batch(
        llm, queries, documents=documents, config=sampling_config
    )

    cluster_fn = cluster_answers_incremental if use_incremental else cluster_answers_bruteforce

    results: List[MSCPResult] = []
    for query, answers in zip(queries, answers_batch):
        clusters = cluster_fn(llm, query, answers, config=nli_config)
        score = _mscp_from_clusters(clusters, len(answers))
        results.append(MSCPResult(score=score, clusters=clusters, answers=answers))
    return results
