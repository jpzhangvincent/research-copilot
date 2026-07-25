# LCR — LLM-Confidence Reranker (reproduction)

A training-free reranking layer for retrieval-augmented generation. LCR wraps the
output of **any** retriever/reranker and re-sorts the candidates using an LLM's
**confidence**, estimated with **MSCP** (Maximum Semantic Cluster Proportion).
Because it reverts to the base ranking whenever the query is already
high-confidence, LCR has a **safe lower bound**: it can only help, never hurt.

This repo reproduces the method (paper Algorithm 1) end-to-end and evaluates it
with **NDCG@5** on BEIR-style datasets.

## How it works

```
query ─┬─► retriever (BM25 / Contriever) ─► top-K docs + PrevScore
       │                                        │
       │                            optional reranker (QLM, RankGPT,
       │                            CrossEncoder, RankT5, ColBERT, YesNo)
       │                                        │
       └─► MSCP confidence ──► LCR_SORT ────────┘──► re-ranked docs ─► NDCG@5
```

1. **MSCP (`confidence/`)** — sample `K` answers from the LLM (temperature 1),
   cluster them by pairwise **bidirectional NLI entailment**, and score
   `MSCP = |largest cluster| / K` (Eq. 1). Computed both query-only `C(q)` and
   joint `C(q,d)`.
2. **LCR (`lcr/`)** — Algorithm 1:
   - If `C(q) ≥ T_query`, keep the base ranking (safe lower bound).
   - Otherwise bin each `C(q,d)` into `{+1, 0, −1}` (`T_upper`/`T_lower`) and
     stable-sort by `(bin, PrevScore)`.

## Layout

| Path | What |
|------|------|
| `confidence/sampling.py` | K-sample multinomial generation (vLLM / HF backends) |
| `confidence/nli_cluster.py` | pairwise-NLI clustering (incremental + brute force) |
| `confidence/mscp.py` | MSCP(x; φ, K) score |
| `lcr/binning.py` | `C(q)` / `C(q,d)` + three-way `BinnedConfidenceScore` |
| `lcr/lcr_sort.py` | `LCR_SORT` (Algorithm 1) |
| `retrievers/` | BM25 (sparse) and Contriever (dense) base retrievers |
| `rerankers/` | QLM, RankGPT, CrossEncoder, RankT5, ColBERT, YesNo baselines |
| `eval/ndcg.py` | NDCG@k (graded, pure-Python) |
| `eval/run_experiment.py` | base-vs-LCR comparison |
| `data/` | BEIR / TREC loaders |
| `run.py` | entry point (`--smoke` and `--config`) |

## Quickstart

```bash
pip install numpy rank_bm25          # minimal deps for the smoke run

# 1) End-to-end smoke test — toy corpus + deterministic mock LLM ($0, no GPU):
python run.py --smoke

# 2) Full reproduction — real BEIR dataset + a real confidence LLM:
pip install -r requirements.txt      # uncomment the full-reproduction section
python run.py --config configs/default.yaml
```

The smoke run exercises the entire pipeline (retrieve → LCR_SORT → NDCG@5) and
confirms the safe-lower-bound property. It uses a **mock** confidence model, so it
reports **no gain** by design; the paper's improvements (up to **+20.6%** NDCG@5)
require a real 7–9B model — configure it in `configs/default.yaml`.

## Configuration

Key knobs in `configs/default.yaml`: `dataset`, `retriever`, `reranker`,
`top_k`, `ndcg_k`, MSCP `sampling.K`, and the LCR `thresholds`
(`t_query`, `t_upper`, `t_lower`). Thresholds are grid-searched per
(dataset, reranker) in the paper (§4.2); the defaults are a reasonable start.

## Notes / known gaps

- `T_query = 0` reproduces the base ranker exactly (a unit-testable invariant of
  `lcr_sort_from_confidences`).
- MSCP cost is ~`(1 + top_k) · K` generations per query; batch via the vLLM backend.
- Exact tuning grids and per-dataset thresholds are not disclosed in the paper, so
  absolute numbers depend on your sweep and LLM choice.
