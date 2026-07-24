# Reproduction Spec — LLM-Confidence Reranker (LCR)

Locked target for the demo Reproduction. Source: arXiv:2602.13571 (Song et al., Feb 2026; ESWA 2026).

## Method (what the Engineer must build)

LCR is a **training-free, plug-and-play reranker** inserted after a base retriever. Two stages:

1. **Confidence assessment** — for a query (and per candidate document), derive a black-box LLM confidence via **Maximum Semantic Cluster Proportion (MSCP)**: sample the LLM multiple times (multinomial sampling), cluster the samples semantically, and take the proportion of the largest cluster as the confidence signal.
2. **Binning + multi-level sort** — bin candidates by query- and document-confidence thresholds, then multi-level sort so relevant docs rise while high-confidence queries preserve the base ranking (robustness / no-degradation guarantee).

- Base retrievers: **BM25** (sparse) and **Contriever** (dense).
- Confidence LLM: a **7–9B pretrained** model in the paper. For the demo we substitute **OpenAI** (per ADR: standardize on one provider) as the black-box confidence source.
- No training. Pure inference → fast to pre-bake (ADR-0002).

## Dataset

- **NFCorpus** (BEIR) — ~3.6K docs, small test query set. Chosen for speed (default per shared-understanding).
- Fallback if NFCorpus setup is fiddly: SciFact or any tiny BEIR set with published NDCG@5.

## Reproduction Goal (Goal Loop success criterion)

- Metric: **NDCG@5** on the chosen BEIR set.
- Target: **BM25 + LCR** beats **BM25 alone** by a margin consistent with the paper's reported gains (paper reports up to +20.6% NDCG@5).
- Tolerance: land within a set band of the paper's reported improvement; Goal Loop caps at N iterations (N TBD at build) and falls back to the cached known-good run.

## Open items to confirm during build

- Exact NFCorpus NDCG@5 baseline for BM25 (to set the tolerance band).
- Sampling count + clustering method for MSCP that is cheap enough to pre-bake yet reproduces the effect.
- Whether DeepCode's paper2code, given the PDF, generates this pipeline unaided or needs a nudge in the plan-review gate (Clarification).
