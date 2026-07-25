"""LCR reproduction: LLM-Confidence Reranker (training-free reranking for RAG).

Packages:
  confidence/ — MSCP(x; phi, K): sampling + pairwise-NLI clustering + score.
  lcr/        — Algorithm 1: BinnedConfidenceScore + LCR_SORT.
  retrievers/ — BM25 (sparse) and Contriever (dense) base retrievers.
  rerankers/  — optional upstream rerankers LCR can be layered on top of.
  eval/       — NDCG@k and the end-to-end experiment runner.
  data/       — BEIR / TREC dataset loaders.
"""

__version__ = "0.1.0"
