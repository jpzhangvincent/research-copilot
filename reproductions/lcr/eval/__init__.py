"""Evaluation utilities (NDCG@k) and the end-to-end experiment runner."""

from .ndcg import dcg_at_k, ndcg_at_k, mean_ndcg_at_k

__all__ = ["dcg_at_k", "ndcg_at_k", "mean_ndcg_at_k"]
