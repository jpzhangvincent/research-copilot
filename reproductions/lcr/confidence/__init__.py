"""Confidence package: MSCP(x; phi, K) pipeline (paper Sec 3.1, Eq. 1).

Re-exports the public API from sampling.py, nli_cluster.py, and mscp.py so
downstream code (lcr/, eval/) can simply do:

    from paper_000aa76f_repro.confidence import mscp_score, LCRThresholds
"""

from .sampling import (
    QUERY_ONLY_PROMPT,
    QUERY_WITH_DOC_PROMPT,
    LLMClient,
    SamplingConfig,
    VLLMClient,
    HFTransformersClient,
    build_sampling_prompt,
    sample_answers,
    sample_answers_batch,
)
from .nli_cluster import (
    ENTAILMENT,
    CONTRADICTION,
    NEUTRAL,
    NLI_PROMPT_TEMPLATE,
    NLIConfig,
    UnionFind,
    build_nli_prompt,
    parse_entailment_label,
    query_entailment,
    query_entailment_batch,
    is_bidirectional_entailment,
    cluster_answers_bruteforce,
    cluster_answers_incremental,
)
from .mscp import (
    MSCPResult,
    compute_mscp,
    mscp_score,
    compute_mscp_batch,
)

__all__ = [
    # sampling.py
    "QUERY_ONLY_PROMPT",
    "QUERY_WITH_DOC_PROMPT",
    "LLMClient",
    "SamplingConfig",
    "VLLMClient",
    "HFTransformersClient",
    "build_sampling_prompt",
    "sample_answers",
    "sample_answers_batch",
    # nli_cluster.py
    "ENTAILMENT",
    "CONTRADICTION",
    "NEUTRAL",
    "NLI_PROMPT_TEMPLATE",
    "NLIConfig",
    "UnionFind",
    "build_nli_prompt",
    "parse_entailment_label",
    "query_entailment",
    "query_entailment_batch",
    "is_bidirectional_entailment",
    "cluster_answers_bruteforce",
    "cluster_answers_incremental",
    # mscp.py
    "MSCPResult",
    "compute_mscp",
    "mscp_score",
    "compute_mscp_batch",
]
