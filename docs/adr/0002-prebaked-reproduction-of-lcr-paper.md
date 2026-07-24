# Pre-baked Reproduction of the LLM-Confidence Reranker paper

The demo Reproduction targets the **LLM-Confidence Reranker (LCR)** paper (arXiv 2602.13571, Feb 2026) in the LLM-agents/RAG domain. The full run is pre-baked offline; the live demo kicks off paper2code and shows the streaming UI, but the finished repo and results are cached as the guaranteed payoff.

## Considered Options

- **LiR3AG / Online-Optimized RAG for Tool Use**: on-theme but more moving parts (multi-module, reasoning models, online-learning loops) — riskier to reproduce.
- **Classic method (HyDE, etc.)**: reliable but not "fresh", weakens the Discovery narrative.
- **Fully live reproduction**: authentic but can fail on stage.

## Why LCR

- **Training-free** — reproduction is pure inference code (BM25 retrieval + LLM-confidence reranking), no GPU training to pre-bake.
- **Tiny corpus** — evaluates on small BEIR datasets (e.g. NFCorpus), so it runs fast.
- **Crisp metric** — NDCG@5 gives a clean, visible payoff number in the demo.
- **Recent (Feb 2026)** — Discovery surfacing it looks genuinely fresh.

## Consequences

- We must cache a known-good generated repo + result numbers before the demo; the live run is a veneer with a guaranteed fallback.
- Reproduction quality is judged against the paper's reported NDCG@5 on the chosen BEIR subset.
